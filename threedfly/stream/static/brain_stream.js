// 3dfly brain-stream protocol v1 decoder (ES module, no dependencies).
// See docs/stream_protocol.md. Zero-copy typed-array views on the message.
export const MAGIC = "3DFY";
export const VERSION = 1;
export const MsgType = { HELLO: 0, FRAME: 1 };
export const Section = {
  META_JSON: 0, POSE: 1, DN: 2, CELLTYPE: 3, ACTIVITY: 4,
  EYE_L: 5, EYE_R: 6, CONTROL: 7, STATS: 8,
};
const DTYPES = {
  1: Uint8Array, 2: Uint16Array, 3: Float32Array, 4: Float64Array, 5: Int32Array,
  6: Uint16Array, // f16 raw bits
};

export function decodeMessage(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== MAGIC) throw new Error("bad magic " + magic);
  const version = dv.getUint16(4, true);
  if (version !== VERSION) throw new Error("unsupported version " + version);
  const msgType = dv.getUint8(6), nSec = dv.getUint8(7);
  const frameIndex = dv.getUint32(8, true), total = dv.getUint32(12, true);
  if (total !== buf.byteLength) throw new Error("length mismatch");
  const simTime = dv.getFloat64(16, true), wallTime = dv.getFloat64(24, true);
  const sections = {};
  let off = 32;
  for (let i = 0; i < nSec; i++) {
    const sid = dv.getUint16(off, true), code = dv.getUint8(off + 2);
    const len = dv.getUint32(off + 4, true);
    off += 8;
    const T = DTYPES[code];
    if (T) sections[sid] = new T(buf, off, len / T.BYTES_PER_ELEMENT);
    off += len + ((8 - (len % 8)) % 8);
  }
  const out = { msgType, version, frameIndex, simTime, wallTime, sections, nbytes: total };
  if (msgType === MsgType.HELLO && sections[Section.META_JSON]) {
    out.meta = JSON.parse(new TextDecoder().decode(sections[Section.META_JSON]));
  }
  return out;
}

// uint8 log1p code → Hz lookup table (256 entries) from HELLO meta.activity.
export function activityLUT(meta) {
  const { floor, cap } = meta.activity;
  const k = Math.log1p(cap / floor) / 255;
  const lut = new Float32Array(256);
  for (let q = 0; q < 256; q++) lut[q] = floor * Math.expm1(q * k);
  return lut;
}

export class BrainStream {
  constructor(url, { onHello, onFrame, onStatus } = {}) {
    this.url = url; this.onHello = onHello; this.onFrame = onFrame; this.onStatus = onStatus;
    this.meta = null; this.nFrames = 0; this.nBytes = 0; this.t0 = null;
    this.connect();
  }
  connect() {
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => this.onStatus && this.onStatus("connected");
    ws.onclose = () => { this.onStatus && this.onStatus("closed"); setTimeout(() => this.connect(), 1000); };
    ws.onmessage = (ev) => {
      const m = decodeMessage(ev.data);
      if (m.msgType === MsgType.HELLO) { this.meta = m.meta; this.lut = activityLUT(m.meta); this.onHello && this.onHello(m.meta); return; }
      if (!this.meta) return;
      const now = performance.now();
      if (this.t0 === null) this.t0 = now;
      this.nFrames++; this.nBytes += m.nbytes;
      this.onFrame && this.onFrame(m, this);
    };
    this.ws = ws;
  }
  rates() { // fps & bytes/s since first frame
    const dt = (performance.now() - this.t0) / 1000;
    return { fps: this.nFrames > 1 ? (this.nFrames - 1) / dt : 0, bps: dt > 0 ? this.nBytes / dt : 0 };
  }
}
