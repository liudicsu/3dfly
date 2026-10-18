// PR5 dashboard panels: L/R hex eyes, signal-flow, DN traces, heatmap, timeline.
// Consumes decoded protocol-v1 frames (sections + HELLO meta).

const FLOW_NODES = [
  { id: "eyes", label: "eyes", x: 0.06, y: 0.5 },
  { id: "T4/T5", label: "T4/T5", x: 0.24, y: 0.5 },
  { id: "LPLC2", label: "LPLC2", x: 0.44, y: 0.28 },
  { id: "LC4", label: "LC4", x: 0.44, y: 0.72 },
  { id: "DNp01", label: "DNp01", x: 0.66, y: 0.18 },
  { id: "DNp03", label: "DNp03", x: 0.66, y: 0.42 },
  { id: "DNa02", label: "DNa02", x: 0.66, y: 0.62 },
  { id: "DNg02", label: "DNg02", x: 0.66, y: 0.84 },
  { id: "wings", label: "wings", x: 0.9, y: 0.5 },
];
const FLOW_EDGES = [
  [0, 1], [1, 2], [1, 3], [2, 4], [3, 4], [2, 5], [3, 5], [1, 6], [1, 7],
  [4, 8], [5, 8], [6, 8], [7, 8],
];

const DN_SERIES = [
  { key: "dnp01", label: "DNp01", color: "#ff5d5d", L: "dnp01_L", R: "dnp01_R" },
  { key: "dnp03", label: "DNp03", color: "#ffb454", L: "dnp03_L", R: "dnp03_R" },
  { key: "dna02", label: "DNa02", color: "#6fb6ff", L: "dna02_L", R: "dna02_R" },
  { key: "dng02", label: "DNg02", color: "#59d38a", L: "dng02_L", R: "dng02_R" },
  { key: "lplc2", label: "LPLC2", color: "#c792ea", L: "lplc2_L", R: "lplc2_R" },
];

const HEAT_GROUPS = ["T4a", "T4b", "T5a", "T5b", "LPLC2", "LC4", "DNp01", "DNp03", "DNa02", "DNg02", "SApp"];

function fit(cv) {
  const r = cv.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width));
  const h = Math.max(1, Math.round(r.height));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  return cv.getContext("2d");
}

/** Build 721 hex axial coords matching flyvis extent=15. */
export function hexLattice(extent = 15) {
  const out = [];
  for (let u = -extent; u <= extent; u++) {
    for (let v = Math.max(-extent, -u - extent); v <= Math.min(extent, -u + extent); v++) {
      out.push([u, v]);
    }
  }
  return out;
}

export class DashboardPanels {
  constructor({ eyeL, eyeR, flow, dn, heat, timeline } = {}) {
    this.eyeL = eyeL;
    this.eyeR = eyeR;
    this.flow = flow;
    this.dn = dn;
    this.heat = heat;
    this.timeline = timeline;
    this.hexes = hexLattice(15);
    this.dnHist = Object.fromEntries(DN_SERIES.map((s) => [s.key, []]));
    this.heatHist = [];
    this.events = []; // {t, kind}
    this.meta = null;
    this.dnIndex = {};
    this.ctIndex = {};
    this.lastSim = 0;
    this.histLen = 300;
    this._lastDraw = 0;
  }

  onHello(meta) {
    this.meta = meta;
    this.dnIndex = Object.fromEntries((meta.dn_names || []).map((n, i) => [n, i]));
    this.ctIndex = Object.fromEntries((meta.celltype_names || []).map((n, i) => [n, i]));
  }

  onFrame(msg, sections) {
    const dn = sections.DN;
    const ct = sections.CELLTYPE;
    const eyeL = sections.EYE_L;
    const eyeR = sections.EYE_R;
    const stats = sections.STATS;
    this.lastSim = msg.simTime;

    // DN history (mean L/R)
    for (const s of DN_SERIES) {
      let v = 0, n = 0;
      if (dn) {
        for (const side of [s.L, s.R]) {
          const i = this.dnIndex[side];
          if (i != null) { v += dn[i]; n++; }
        }
      }
      const hz = n ? v / n : 0;
      const h = this.dnHist[s.key];
      h.push(hz);
      if (h.length > this.histLen) h.shift();
    }

    // Escape event: DNp01 mean above threshold
    const dnp = this.dnHist.dnp01;
    const cur = dnp[dnp.length - 1] || 0;
    if (cur > 5 && (!this.events.length || msg.simTime - this.events[this.events.length - 1].t > 0.4)) {
      this.events.push({ t: msg.simTime, kind: "GF", v: cur });
    }

    // Heatmap column
    const col = HEAT_GROUPS.map((name) => {
      const i = this.ctIndex[name];
      return i != null && ct ? Math.min(1, Math.log1p(ct[i]) / Math.log1p(50)) : 0;
    });
    this.heatHist.push(col);
    if (this.heatHist.length > this.histLen) this.heatHist.shift();

    const now = performance.now();
    if (now - this._lastDraw < 33) return;
    this._lastDraw = now;

    const flowVals = this._flowValues(dn, ct);
    if (this.eyeL && eyeL) this.drawEye(this.eyeL, eyeL, "L");
    if (this.eyeR && eyeR) this.drawEye(this.eyeR, eyeR, "R");
    if (this.flow) this.drawFlow(flowVals);
    if (this.dn) this.drawDN();
    if (this.heat) this.drawHeat();
    if (this.timeline) this.drawTimeline(stats);
  }

  _flowValues(dn, ct) {
    const getCt = (name) => {
      const i = this.ctIndex[name];
      return i != null && ct ? ct[i] : 0;
    };
    const getDn = (a, b) => {
      let s = 0, n = 0;
      for (const k of [a, b]) {
        const i = this.dnIndex[k];
        if (i != null && dn) { s += dn[i]; n++; }
      }
      return n ? s / n : 0;
    };
    const norm = (hz, cap = 30) => Math.min(1, Math.log1p(hz) / Math.log1p(cap));
    const t4 = 0.25 * (getCt("T4a") + getCt("T4b") + getCt("T4c") + getCt("T4d"));
    const t5 = 0.25 * (getCt("T5a") + getCt("T5b") + getCt("T5c") + getCt("T5d"));
    return {
      eyes: 0.4,
      "T4/T5": norm(0.5 * (t4 + t5), 20),
      LPLC2: norm(getDn("lplc2_L", "lplc2_R"), 20),
      LC4: norm(getDn("lc4_L", "lc4_R"), 20),
      DNp01: norm(getDn("dnp01_L", "dnp01_R"), 40),
      DNp03: norm(getDn("dnp03_L", "dnp03_R"), 20),
      DNa02: norm(getDn("dna02_L", "dna02_R"), 20),
      DNg02: norm(getDn("dng02_L", "dng02_R") || (this.dnIndex.dng02_amp != null && dn ? dn[this.dnIndex.dng02_amp] : 0), 20),
      wings: norm(getDn("dng02_L", "dng02_R"), 20),
    };
  }

  drawEye(cv, u8, side) {
    const c = fit(cv), W = cv.width, H = cv.height;
    c.fillStyle = "#0b1220"; c.fillRect(0, 0, W, H);
    const extent = 15;
    const s = Math.min(W, H) / (2 * extent + 3) / 1.75;
    const n = Math.min(this.hexes.length, u8.length);
    for (let j = 0; j < n; j++) {
      const [u, v] = this.hexes[j];
      const x = W / 2 + s * 1.5 * u;
      const y = H / 2 + s * Math.sqrt(3) * (v + u / 2);
      const g = u8[j] / 255;
      c.fillStyle = `hsl(${210 - 190 * g},80%,${12 + 50 * g}%)`;
      c.beginPath();
      for (let a = 0; a < 6; a++) {
        const th = Math.PI / 3 * a;
        c.lineTo(x + s * Math.cos(th), y + s * Math.sin(th));
      }
      c.closePath(); c.fill();
    }
    c.fillStyle = "#8fb3ff";
    c.fillText(`${side} eye · ${n} hex`, 4, 12);
  }

  drawFlow(vals) {
    const cv = this.flow, c = fit(cv), W = cv.width, H = cv.height;
    c.fillStyle = "#0b1220"; c.fillRect(0, 0, W, H);
    FLOW_EDGES.forEach(([a, b]) => {
      const A = FLOW_NODES[a], B = FLOW_NODES[b];
      const v = vals[A.id] || 0;
      c.strokeStyle = `rgba(255,${120 + 100 * v},60,${0.15 + 0.85 * v})`;
      c.lineWidth = 1 + 4 * v;
      c.beginPath(); c.moveTo(A.x * W, A.y * H); c.lineTo(B.x * W, B.y * H); c.stroke();
    });
    FLOW_NODES.forEach((node) => {
      const v = vals[node.id] || 0;
      c.fillStyle = `hsl(${30 + 20 * v},90%,${15 + 50 * v}%)`;
      c.beginPath(); c.arc(node.x * W, node.y * H, 7 + 7 * v, 0, 7); c.fill();
      c.fillStyle = "#e6edf3";
      c.fillText(node.label, node.x * W - 16, node.y * H - 12 - 6 * v);
    });
  }

  drawDN() {
    const cv = this.dn, c = fit(cv), W = cv.width, H = cv.height;
    c.fillStyle = "#0b1220"; c.fillRect(0, 0, W, H);
    const rows = DN_SERIES.length;
    DN_SERIES.forEach((s, row) => {
      const h = this.dnHist[s.key];
      const y0 = (row + 1) * H / rows - 4;
      const rowH = H / rows - 8;
      const vmax = Math.max(1, ...h, 1);
      c.strokeStyle = s.color; c.beginPath();
      h.forEach((v, j) => c.lineTo(j / Math.max(this.histLen - 1, 1) * W, y0 - (v / vmax) * rowH));
      c.stroke();
      c.fillStyle = s.color;
      const last = h[h.length - 1] || 0;
      c.fillText(`${s.label} ${last.toFixed(2)} Hz`, 4, y0 - rowH + 12);
    });
    this.events.forEach((e) => {
      const age = this.lastSim - e.t;
      const x = W - age / 10 * W;
      if (x > 0 && x < W) {
        c.strokeStyle = "#ff5d5d55"; c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke();
      }
    });
  }

  drawHeat() {
    const cv = this.heat, c = fit(cv), W = cv.width, H = cv.height;
    c.fillStyle = "#0b1220"; c.fillRect(0, 0, W, H);
    const G = HEAT_GROUPS.length, lw = 56, rh = (H - 16) / G;
    const cw = (W - lw) / Math.max(this.histLen, 1);
    this.heatHist.forEach((col, j) => col.forEach((v, g) => {
      c.fillStyle = `hsl(${240 - 220 * v},85%,${8 + 52 * v}%)`;
      c.fillRect(lw + j * cw, g * rh, cw + 0.5, rh - 1);
    }));
    c.fillStyle = "#cfd8e3";
    HEAT_GROUPS.forEach((g, i) => c.fillText(g, 2, i * rh + rh * 0.7));
    this.events.forEach((e) => {
      const age = this.lastSim - e.t;
      const x = lw + (W - lw) - age / 10 * (W - lw);
      if (x > lw) { c.fillStyle = "#ff5d5d"; c.fillText("▲", x - 4, H - 3); }
    });
  }

  drawTimeline(stats) {
    const cv = this.timeline, c = fit(cv), W = cv.width, H = cv.height;
    c.fillStyle = "#0b1220"; c.fillRect(0, 0, W, H);
    c.fillStyle = "#8fb3ff";
    c.fillText(`t = ${this.lastSim.toFixed(2)} s`, 6, 14);
    if (stats) {
      c.fillStyle = "#cfd8e3";
      c.fillText(`active ${(100 * stats[2]).toFixed(1)}% · mean ${stats[0].toFixed(2)} Hz · max ${stats[1].toFixed(1)} Hz`, 6, 30);
    }
    // event ticks
    const tMax = 10;
    this.events.forEach((e) => {
      const age = this.lastSim - e.t;
      if (age < 0 || age > tMax) return;
      const x = 6 + (1 - age / tMax) * (W - 12);
      c.fillStyle = "#ff5d5d";
      c.fillRect(x, H - 18, 2, 12);
      c.fillText(e.kind, x - 6, H - 4);
    });
    c.strokeStyle = "#1e2a3a";
    c.beginPath(); c.moveTo(6, H - 20); c.lineTo(W - 6, H - 20); c.stroke();
  }
}
