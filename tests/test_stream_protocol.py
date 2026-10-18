"""Brain-stream protocol v1: encode/decode, builder, websocket server + client (PR2)."""

import json
import struct
import time
import urllib.request

import numpy as np
import pytest

from threedfly.brain.readouts import DNReadout
from threedfly.stream.builder import DN_FIELDS, FrameBuilder
from threedfly.stream.protocol import (
    ALIGN,
    HEADER_SIZE,
    MAGIC,
    PROTOCOL_VERSION,
    MsgType,
    ProtocolError,
    Section,
    decode_frame_values,
    decode_message,
    dequantize_log1p,
    encode_frame,
    encode_hello,
    encode_message,
    quantize_log1p,
)


def test_header_layout_and_alignment():
    buf = encode_frame(7, 1.25, {Section.DN: np.arange(3, dtype=np.float32),
                                 Section.ACTIVITY: np.arange(5, dtype=np.uint8)}, wall_time=10.0)
    assert buf[:4] == MAGIC
    ver, mtype, nsec, idx, total = struct.unpack_from("<HBBII", buf, 4)
    assert (ver, mtype, nsec, idx, total) == (PROTOCOL_VERSION, MsgType.FRAME, 2, 7, len(buf))
    assert struct.unpack_from("<dd", buf, 16) == (1.25, 10.0)
    assert HEADER_SIZE == 32
    # payloads 8-byte aligned (JS typed-array views without copying)
    assert (HEADER_SIZE + 8) % ALIGN == 0
    assert len(buf) % ALIGN == 0
    fr = decode_message(buf)
    np.testing.assert_array_equal(fr.sections[Section.DN], [0, 1, 2])
    np.testing.assert_array_equal(fr.sections[Section.ACTIVITY], [0, 1, 2, 3, 4])
    assert fr.frame_index == 7 and fr.sim_time == 1.25


def test_unknown_section_is_skipped_and_errors():
    buf = encode_message(MsgType.FRAME, {999: np.ones(3, np.float32), Section.DN: np.ones(2, np.float32)})
    fr = decode_message(buf)
    assert 999 in fr.sections and Section.DN in fr.sections
    with pytest.raises(ProtocolError):
        decode_message(b"XXXX" + buf[4:])
    with pytest.raises(ProtocolError):
        decode_message(buf[:-8])
    bad = bytearray(buf)
    struct.pack_into("<H", bad, 4, 99)
    with pytest.raises(ProtocolError):
        decode_message(bytes(bad))


def test_log1p_quantisation_error():
    r = np.array([0, 1e-4, 1e-3, 0.01, 0.1, 1, 5, 20, 50, 199, 500], dtype=np.float32)
    q = quantize_log1p(r)
    assert q[0] == 0 and q[-1] == 255
    assert np.all(np.diff(q.astype(int)) >= 0)
    back = dequantize_log1p(q)
    mid = (r >= 0.01) & (r <= 200)
    rel = np.abs(back[mid] - r[mid]) / r[mid]
    assert rel.max() < 0.03  # ≤ ~2.4 % per code step at cap 200 Hz


def test_hello_roundtrip():
    meta = {"n_neurons": 3, "dn_names": ["a"]}
    fr = decode_message(encode_hello(meta))
    assert fr.msg_type == MsgType.HELLO
    assert fr.meta["n_neurons"] == 3 and fr.meta["version"] == PROTOCOL_VERSION


def _builder(n=120):
    types = np.array(["T4a"] * 40 + ["T5a"] * 40 + ["LPLC2"] * 20 + ["DNp01"] * 2 + ["x"] * (n - 102))
    sides = np.array(["L", "R"] * (n // 2))
    return FrameBuilder(types=types, sides=sides, body_ids=np.arange(n))


def test_builder_window_max_and_values():
    b = _builder()
    meta = decode_message(b.hello()).meta
    assert meta["n_neurons"] == 120 and "DNp01" in meta["celltype_names"]
    assert len(meta["dn_names"]) == len(DN_FIELDS)
    assert len(meta["eye_layout"]["L"]["az_deg"]) == 721
    r1 = np.zeros(120, np.float32)
    r1[100] = 30.0  # brief DNp01 burst on an intermediate step
    r2 = np.zeros(120, np.float32)
    r2[:40] = 2.0
    b.accumulate(r1)
    buf = b.build(
        sim_time=0.5, rates=r2, readout=DNReadout(dnp01_L=30.0, dng02_amp=5.0),
        position=[1, 2, 3], orientation=[1, 0, 0, 0], velocity=[0.1, 0, 0],
        wing_angles=[0.3, -0.3], hex_L=np.full(721, 0.5), hex_R=np.linspace(0, 1, 721),
        control=[0.1, 0.2, 0, 0, -0.1],
    )
    v = decode_frame_values(decode_message(buf), meta)
    assert v["activity"][100] == pytest.approx(30.0, rel=0.03)  # window max kept the burst
    assert v["activity"][:40] == pytest.approx(2.0, rel=0.03)
    assert v["celltype"]["T4a"] == pytest.approx(2.0)
    assert v["celltype"]["DNp01"] == pytest.approx(15.0)
    assert v["dn"]["dnp01_L"] == pytest.approx(30.0) and v["dn"]["dng02_amp"] == pytest.approx(5.0)
    assert v["pose"]["z"] == pytest.approx(3.0) and v["pose"]["wing_L"] == pytest.approx(0.3)
    assert v["eye_L"].mean() == pytest.approx(0.5, abs=0.01)
    assert v["eye_R"][-1] == pytest.approx(1.0)
    assert v["control"]["yaw"] == pytest.approx(-0.1)
    assert v["stats"]["window_steps"] == 2 and v["stats"]["n_active"] == 41
    # window resets after build
    v2 = decode_frame_values(decode_message(b.build(sim_time=0.52, rates=np.zeros(120))), meta)
    assert v2["activity"].max() == 0 and v2["frame"] == 1
    with pytest.raises(ValueError):
        b.accumulate(np.zeros(5))


def test_frame_size_for_core_tplus():
    n = 45_434
    b = FrameBuilder(n_neurons=n, include_eye_layout=False)
    buf = b.build(0.0, rates=np.zeros(n), hex_L=np.zeros(721), hex_R=np.zeros(721), control=np.zeros(5))
    assert n + 2 * 721 < len(buf) < n + 2 * 721 + 512


def test_server_client_roundtrip():
    pytest.importorskip("websockets")
    from threedfly.stream.client import receive
    from threedfly.stream.server import BrainStreamServer

    b = _builder()
    srv = BrainStreamServer(host="127.0.0.1", port=0, max_hz=200).start()
    try:
        srv.set_hello(b.hello())
        import threading

        stop = threading.Event()

        def produce():
            i = 0
            while not stop.is_set():
                r = np.zeros(120, np.float32)
                r[i % 120] = 10.0
                if srv.wants_frame():
                    srv.publish(b.build(sim_time=0.02 * i, rates=r, readout=DNReadout(dnp01_L=float(i))))
                i += 1
                time.sleep(0.002)

        th = threading.Thread(target=produce, daemon=True)
        th.start()
        rep = receive(srv.url, max_frames=20, duration=10)
        stop.set()
        th.join()
        assert rep.meta["n_neurons"] == 120
        assert rep.n == 20
        idx = [f.frame_index for f in rep.frames]
        assert idx == sorted(idx)
        last = rep.latest_values()
        assert last["activity"].max() == pytest.approx(10.0, rel=0.03)
        assert last["dn"]["dnp01_L"] > 0
        assert rep.fps > 0 and rep.bytes_per_s > 0
        # HTTP side: inspector page + hello.json + decoder module
        base = srv.url.replace("ws://", "http://")
        html = urllib.request.urlopen(base + "/").read().decode()
        assert "brain_stream.js" in html
        js = urllib.request.urlopen(base + "/brain_stream.js").read().decode()
        assert "decodeMessage" in js
        meta = json.loads(urllib.request.urlopen(base + "/hello.json").read())
        assert meta["n_neurons"] == 120
        st = json.loads(urllib.request.urlopen(base + "/stats.json").read())
        assert st["frames_sent"] >= 20
        with pytest.raises(Exception):
            urllib.request.urlopen(base + "/../protocol.py")
    finally:
        srv.stop()


def test_wants_frame_rate_limit_and_no_clients():
    from threedfly.stream.server import BrainStreamServer

    srv = BrainStreamServer(port=0, max_hz=10)
    assert not srv.wants_frame()  # no clients → don't encode
    srv.n_clients = 1
    assert srv.wants_frame()
    srv.publish(b"x" * 40)
    assert not srv.wants_frame()
    assert srv.wants_frame(now=time.perf_counter() + 0.11)


def test_recording_roundtrip(tmp_path):
    from threedfly.stream.client import read_recording, write_recording

    b = _builder()
    frames = [b.build(0.02 * i, rates=np.full(120, float(i), np.float32)) for i in range(3)]
    path = tmp_path / "rec.bin"
    write_recording(path, b.hello(), frames)
    hello, back = read_recording(path)
    assert decode_message(hello).msg_type == MsgType.HELLO
    assert back == frames


def test_brain_panel_and_assets_http_routes():
    """PR4: /brain panel page + /brain-assets/* from assets/brain."""
    from pathlib import Path
    import urllib.request

    from threedfly.stream.builder import FrameBuilder
    from threedfly.stream.server import BrainStreamServer
    from threedfly.viz.brain_assets import ASSET_NAME, default_asset_dir

    asset_dir = default_asset_dir()
    if not (asset_dir / ASSET_NAME).is_file():
        import pytest
        pytest.skip("brain assets not present")

    fb = FrameBuilder(n_neurons=16, include_eye_layout=False)
    srv = BrainStreamServer(host="127.0.0.1", port=0, max_hz=30).start()
    try:
        srv.set_hello(fb.hello())
        base = f"http://127.0.0.1:{srv.port}"
        panel = urllib.request.urlopen(base + "/brain", timeout=5).read()
        assert b"brain panel" in panel.lower() or b"coreTplus" in panel
        assert b"three.module.js" in panel
        meta = urllib.request.urlopen(base + "/brain-assets/meta.json", timeout=5).read()
        assert b"body_ids_sha1" in meta
        pos = urllib.request.urlopen(base + "/brain-assets/positions_u16.bin", timeout=5).read()
        assert len(pos) > 1000
        # path traversal rejected
        import urllib.error
        try:
            urllib.request.urlopen(base + "/brain-assets/../../pyproject.toml", timeout=5)
            raised = False
        except urllib.error.HTTPError as exc:
            raised = True
            assert exc.code == 404
        assert raised, "traversal should 404"
    finally:
        srv.stop()


def test_dashboard_panels_static_served():
    import urllib.request
    from threedfly.stream.builder import FrameBuilder
    from threedfly.stream.server import BrainStreamServer

    fb = FrameBuilder(n_neurons=8, include_eye_layout=False)
    srv = BrainStreamServer(host="127.0.0.1", port=0, max_hz=30).start()
    try:
        srv.set_hello(fb.hello())
        base = f"http://127.0.0.1:{srv.port}"
        js = urllib.request.urlopen(base + "/dashboard_panels.js", timeout=5).read()
        assert b"DashboardPanels" in js
        assert b"hexLattice" in js
        html = urllib.request.urlopen(base + "/brain", timeout=5).read()
        assert b"dashboard_panels.js" in html
        assert b"eyeL" in html and b"PATHWAY" in html
    finally:
        srv.stop()


def test_hex_lattice_extent15_is_721():
    """Mirror dashboard_panels.js hexLattice(15) — flyvis hex count."""
    extent = 15
    n = 0
    for u in range(-extent, extent + 1):
        for v in range(max(-extent, -u - extent), min(extent, -u + extent) + 1):
            n += 1
    assert n == 721
    assert n == 3 * extent * (extent + 1) + 1


def test_dashboard_panels_js_hex_and_flow_constants():
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "threedfly/stream/static/dashboard_panels.js").read_text()
    assert "export function hexLattice" in js
    assert "export class DashboardPanels" in js
    assert "FLOW_NODES" in js and "DNp01" in js and "wings" in js
    assert "DN_SERIES" in js and "HEAT_GROUPS" in js
    assert "drawEye" in js and "drawFlow" in js and "drawDN" in js
    assert "drawHeat" in js and "drawTimeline" in js


def test_demo_and_recording_js_routes(tmp_path):
    """PR6: /demo page, recording.js, and /recording.bin for browser replay."""
    import struct
    import urllib.request

    from threedfly.stream.builder import FrameBuilder
    from threedfly.stream.client import write_recording
    from threedfly.stream.server import BrainStreamServer

    fb = FrameBuilder(n_neurons=8, include_eye_layout=False)
    hello = fb.hello()
    frames = [fb.build(0.02 * i, rates=np.full(8, float(i + 1), np.float32)) for i in range(3)]
    rec = tmp_path / "sample.bin"
    write_recording(rec, hello, frames)

    srv = BrainStreamServer(
        host="127.0.0.1", port=0, max_hz=30, recording_path=rec
    ).start()
    try:
        srv.set_hello(hello)
        base = f"http://127.0.0.1:{srv.port}"
        demo = urllib.request.urlopen(base + "/demo", timeout=5).read()
        assert b"local demo" in demo.lower() or b"PR6" in demo
        assert b"recording.js" in demo
        js = urllib.request.urlopen(base + "/recording.js", timeout=5).read()
        assert b"parseRecording" in js and b"RecordingPlayer" in js
        raw = urllib.request.urlopen(base + "/recording.bin", timeout=5).read()
        assert raw == rec.read_bytes()
        # length-prefixed: first u32 is hello length
        (n,) = struct.unpack_from("<I", raw, 0)
        assert n == len(hello)
        brain = urllib.request.urlopen(base + "/brain", timeout=5).read()
        assert b"recording.js" in brain
        assert b"REC_URL" in brain or b"recFile" in brain
    finally:
        srv.stop()


def test_stream_record_write_roundtrip(tmp_path):
    """write_recording format is what browser parseRecording expects."""
    import struct

    from threedfly.stream.builder import FrameBuilder
    from threedfly.stream.client import read_recording, write_recording

    fb = FrameBuilder(n_neurons=4, include_eye_layout=False)
    hello = fb.hello()
    frames = [fb.build(0.1 * i, rates=np.arange(4, dtype=np.float32) + i) for i in range(5)]
    path = tmp_path / "out.bin"
    write_recording(path, hello, frames)
    data = path.read_bytes()
    # Browser parseRecording walks u32 lengths
    off, msgs = 0, []
    while off + 4 <= len(data):
        (n,) = struct.unpack_from("<I", data, off)
        off += 4
        msgs.append(data[off : off + n])
        off += n
    assert len(msgs) == 1 + len(frames)
    assert msgs[0] == hello
    assert msgs[1:] == frames
    h2, f2 = read_recording(path)
    assert h2 == hello and f2 == frames
