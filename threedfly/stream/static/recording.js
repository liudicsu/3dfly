// Protocol-v1 recording I/O (browser). Format matches threedfly.stream.client:
// length-prefixed (u32 LE) concatenation of HELLO then FRAME messages.
import { decodeMessage, MsgType } from "./brain_stream.js";

/** Parse an ArrayBuffer produced by write_recording / stream_client --dump. */
export function parseRecording(buf) {
  const ab = buf instanceof ArrayBuffer ? buf : buf.buffer;
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(ab);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const msgs = [];
  let off = 0;
  while (off + 4 <= u8.byteLength) {
    const n = dv.getUint32(off, true);
    off += 4;
    if (n < 0 || off + n > u8.byteLength) throw new Error("truncated recording at offset " + (off - 4));
    msgs.push(u8.buffer.slice(u8.byteOffset + off, u8.byteOffset + off + n));
    off += n;
  }
  if (!msgs.length) throw new Error("empty recording");
  return { helloRaw: msgs[0], framesRaw: msgs.slice(1) };
}

/**
 * Play a protocol-v1 recording into the same callbacks BrainStream uses.
 * Does not open a websocket — pure local file / fetched ArrayBuffer.
 */
export class RecordingPlayer {
  constructor(arrayBuffer, { onHello, onFrame, onStatus, hz = 10, loop = true } = {}) {
    this.onHello = onHello;
    this.onFrame = onFrame;
    this.onStatus = onStatus;
    this.hz = Math.max(0.1, +hz || 10);
    this.loop = !!loop;
    this._timer = null;
    this._i = 0;
    this.nFrames = 0;
    this.nBytes = 0;
    this.t0 = null;
    this.meta = null;
    this.lut = null;
    const { helloRaw, framesRaw } = parseRecording(arrayBuffer);
    this._helloRaw = helloRaw;
    this._frames = framesRaw;
    const hello = decodeMessage(helloRaw);
    if (hello.msgType !== MsgType.HELLO) throw new Error("first message is not HELLO");
    this.meta = hello.meta;
  }

  start() {
    this.onStatus && this.onStatus("recording");
    this.onHello && this.onHello(this.meta);
    this._i = 0;
    this.nFrames = 0;
    this.nBytes = 0;
    this.t0 = null;
    this._tick();
  }

  _tick() {
    if (this._i >= this._frames.length) {
      if (this.loop && this._frames.length) {
        this._i = 0;
        this.onStatus && this.onStatus("loop");
      } else {
        this.onStatus && this.onStatus("ended");
        return;
      }
    }
    const raw = this._frames[this._i++];
    const m = decodeMessage(raw);
    if (m.msgType !== MsgType.FRAME) {
      this._schedule();
      return;
    }
    const now = performance.now();
    if (this.t0 === null) this.t0 = now;
    this.nFrames++;
    this.nBytes += m.nbytes;
    this.onFrame && this.onFrame(m, this);
    this._schedule();
  }

  _schedule() {
    const dt = 1000 / this.hz;
    this._timer = setTimeout(() => this._tick(), dt);
  }

  pause() {
    if (this._timer) { clearTimeout(this._timer); this._timer = null; }
    this.onStatus && this.onStatus("paused");
  }

  resume() {
    if (this._timer) return;
    this.onStatus && this.onStatus("recording");
    this._tick();
  }

  stop() {
    this.pause();
    this._i = 0;
    this.onStatus && this.onStatus("stopped");
  }

  rates() {
    const dt = (performance.now() - (this.t0 || performance.now())) / 1000;
    return { fps: this.nFrames > 1 ? (this.nFrames - 1) / Math.max(dt, 1e-9) : 0, bps: dt > 0 ? this.nBytes / dt : 0 };
  }
}
