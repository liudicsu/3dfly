# 3dfly brain stream — protocol v1

A binary websocket stream of closed-loop brain + body state. It is designed
so the planned three.js brain panel (UI PR4/PR5), replay (PR6) and a fully
custom front end can all consume the same frames.

* Server: `threedfly.stream.server.BrainStreamServer` (started by
  `threedfly run-interactive`, `--stream-port 8765` default, `0` = off,
  `--stream-hz 30` max display rate).
* Same port, plain HTTP: `/` inspector page, `/brain_stream.js` ES-module
  decoder, `/hello.json` (the HELLO metadata), `/stats.json`.
* Python: `threedfly.stream.protocol` (encode/decode),
  `threedfly.stream.client.receive`, `scripts/stream_client.py`,
  `scripts/stream_bench.py`.

## Messages

Little-endian. One websocket binary message = one protocol message. The
first message a client receives is `HELLO`, then `FRAME`s. Each client
always gets the **latest** frame, so a slow client drops frames instead of
building a backlog.

### Header (32 bytes)

| off | type | field |
|---|---|---|
| 0 | 4s | magic `3DFY` |
| 4 | u16 | version (= 1) |
| 6 | u8 | msg_type: 0 HELLO, 1 FRAME |
| 7 | u8 | n_sections |
| 8 | u32 | frame_index (0 for HELLO) |
| 12 | u32 | total_length (bytes, incl. header) |
| 16 | f64 | sim_time (s) |
| 24 | f64 | wall_time (unix s) |

### Section (8-byte header + payload)

| off | type | field |
|---|---|---|
| 0 | u16 | section_id |
| 2 | u8 | dtype: 1 u8, 2 u16, 3 f32, 4 f64, 5 i32, 6 f16 |
| 3 | u8 | flags (0) |
| 4 | u32 | payload_len (bytes, unpadded) |
| 8 | … | payload, zero-padded to a multiple of 8 |

Payloads start on 8-byte boundaries, so JS can do
`new Float32Array(buf, off, n)` without copying. **Unknown section ids must
be skipped** (use `payload_len`). You can add new sections without
changing the version. If you change the meaning of an existing section,
bump the version.

### Sections

| id | name | dtype | length | content |
|---|---|---|---|---|
| 0 | META_JSON | u8 | – | HELLO only: UTF-8 JSON (below) |
| 1 | POSE | f32 | 12 | x y z (m), qw qx qy qz, vx vy vz (m/s), wing_L wing_R (rad) |
| 2 | DN | f32 | `len(dn_names)` (21) | DN / loom readouts (Hz), `DNReadout` fields |
| 3 | CELLTYPE | f32 | `len(celltype_names)` (24) | mean rate per cell-type group (Hz) |
| 4 | ACTIVITY | u8 | `n_neurons` (45,434) | every coreTplus neuron, `RateCore` index order, log1p-quantised |
| 5 | EYE_L | u8 | 721 | left-eye hex luminance, linear 0..1 → 0..255 |
| 6 | EYE_R | u8 | 721 | right eye |
| 7 | CONTROL | f32 | 5 | fwd, up, roll, pitch, yaw command |
| 8 | STATS | f32 | 5 | mean_rate, max_rate, active_fraction, n_active, window_steps |

**ACTIVITY quantisation:** `q = round(255·log1p(r/floor)/log1p(cap/floor))`
with `floor = 1e-3 Hz` and `cap = 200 Hz`. Decode with
`r = floor·expm1(q/255·log1p(cap/floor))`. A JS client can use a 256-entry
LUT (`activityLUT` in `brain_stream.js`). `q = 0` means silent. One code
step is about 4.9% of the rate, so the round-trip error is ≤ about 2.4%
in the 0.01–200 Hz range.

**Decimation:** the sim runs at `sim_dt` = 20 ms, and the stream sends at
most `stream_hz` frames per second. `ACTIVITY`/`CELLTYPE`/`STATS` carry
the **maximum over the sim steps since the previous frame**
(`window_steps` = how many). This way a short DNp01 burst between display
frames still shows up. `DN`, `POSE`, `EYE_*` and `CONTROL` are the latest
values.

### HELLO metadata (JSON)

`protocol`, `version`, `sections` (id → name), `n_neurons`, `neuron_order`,
`body_ids_sha1` (SHA-1 of the int64 body ids in neuron order; the PR3 asset
file will carry the same hash), `pose_fields`, `dn_names`,
`celltype_names`, `celltype_sizes`, `control_names`, `stats_fields`,
`activity` {dtype, scheme, floor, cap, decode, reduction}, `eye` {dtype,
n_hex, scheme, min, max}, `eye_layout` {L, R: {az_deg[721], el_deg[721]}}
(body-frame azimuth/elevation of each flyvis hex, +az = left), and `units`.

## Size and bandwidth (measured on the box, 2 CPU cores)

* Frame: **47,272 B** (45,434 activity + 2×721 eyes + about 400 B
  header/pose/DN/cell types). zlib-6 shrinks it to about 31.6 KB (0.67×).
  The websocket uses permessage-deflate by default.
* Server cost: `accumulate` 0.006 ms per sim step. `build` + encode is
  0.3 ms per frame with real cell types. HELLO takes about 3.6 s once at
  startup, mostly for the eye layout.
* Python client decode: about 0.05–0.1 ms per frame.
* Replay of real closed-loop frames through the server to a Python client
  on localhost (`scripts/stream_bench.py`):
  * 10 Hz → 9.97 fps, 460 KiB/s
  * 30 Hz → 29.8 fps, 1.37 MiB/s
  * 60 Hz → 59.2 fps, 2.7 MiB/s
  * 120 Hz → 117.7 fps, 5.4 MiB/s

  These are uncompressed payload rates. The wire rate with deflate is
  about 0.67× of these.
* Live, inside `run-interactive` with the real coreTplus brain + flyvis:
  the stream follows the simulation loop. On this 2-core box the loop runs
  about 4.5–5 steps/s (about 0.1× real time, including viser and the
  matplotlib dashboard), so the client sees about 4.7–4.9 fps at
  about 220 KiB/s. The 30 Hz cap never limits it here.

## Recording & local replay (PR6)

Length-prefixed (``u32`` LE) concatenation of raw HELLO then FRAME messages —
same bytes the websocket carries. Helpers:

* Python: ``threedfly.stream.client.write_recording`` / ``read_recording``
* CLI dump: ``scripts/stream_client.py --dump PATH``
* Live sim: ``threedfly run-interactive --stream-record PATH`` (records even with 0 clients)
* Server fan-out: ``scripts/stream_replay.py PATH --port 8767``
* Browser: ``threedfly/stream/static/recording.js`` (``parseRecording`` + ``RecordingPlayer``)
* Local demo page: ``http://localhost:PORT/demo`` (not for GitHub Pages)
* In-panel: ``/brain?rec=/recording.bin&hz=10`` or the file picker under Controls

``stream_replay`` also serves the file at ``/recording.bin`` so the browser can
replay without consuming the websocket.

