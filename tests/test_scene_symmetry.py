"""Open-space yaw bias: room and eyes are L/R-even (mirrored-scene control)."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.brain.eyes import hex_camera_uv, raw_stereo_hex, reset_luminance_rest
from threedfly.sim.mjcf import get_fly_mjcf, write_one_over_f_png


def test_one_over_f_texture_is_even():
    from threedfly.sim.mjcf import ASSETS_ROOT

    path = write_one_over_f_png(ASSETS_ROOT / "room_natural.png")
    from pathlib import Path

    raw = Path(path).read_bytes()
    # Decode the gray PNG we just wrote (8-bit, no interlace).
    import struct
    import zlib

    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    i = 8
    pixels = None
    n = None
    while i < len(raw):
        ln = struct.unpack(">I", raw[i : i + 4])[0]
        tag = raw[i + 4 : i + 8]
        data = raw[i + 8 : i + 8 + ln]
        i += 12 + ln
        if tag == b"IHDR":
            n = struct.unpack(">I", data[:4])[0]
        if tag == b"IDAT":
            rows = zlib.decompress(data)
            pixels = np.frombuffer(b"".join(rows[k * (n + 1) + 1 : (k + 1) * (n + 1)] for k in range(n)), dtype=np.uint8).reshape(n, n)
    assert pixels is not None
    assert np.allclose(pixels, np.fliplr(pixels), atol=1)
    assert np.allclose(pixels, np.flipud(pixels), atol=1)


def _hex_lr(env, yaw: float):
    env.set_pose((0.0, 0.0, 1.5), yaw=yaw)
    env.hide_loom_object()
    left, right = env.get_eye_images()
    ls, rs = env.last_side_images
    reset_luminance_rest()
    hL, hR = raw_stereo_hex(left, right, ls, rs)
    _, okL = hex_camera_uv("L", left.shape[0], left.shape[1])
    _, okR = hex_camera_uv("R", right.shape[0], right.shape[1])
    return float(hL[okL].mean() - hR[okR].mean())


def test_room_hex_laterality_is_small_and_flips_under_mirror():
    """Body-frame L−R should be ~0; a Y-mirrored room must not keep the same sign."""
    pytest.importorskip("mujoco")
    from threedfly.sim import FlyEnvironment

    env = FlyEnvironment(render_mode="rgb_array", camera_width=160, camera_height=160)
    d0 = _hex_lr(env, 0.0)
    dpi = _hex_lr(env, np.pi)
    env.close()
    env_m = FlyEnvironment(
        render_mode="rgb_array",
        camera_width=160,
        camera_height=160,
        mjcf_xml=get_fly_mjcf(mirror_y=True),
    )
    d0m = _hex_lr(env_m, 0.0)
    env_m.close()
    assert abs(d0) < 0.008, f"heading 0 hex L−R={d0:.4f}"
    assert abs(dpi) < 0.008, f"heading π hex L−R={dpi:.4f}"
    # Official room is already paired; mirror residual should also be small
    # and must not reproduce a body-frame (same-sign) bias.
    assert abs(d0m) < 0.008, f"mirrored heading 0 hex L−R={d0m:.4f}"


def test_physics_has_no_open_space_yaw_bias():
    from threedfly.control.exploration import heading_from_quaternion
    from threedfly.sim import FlyEnvironment

    env = FlyEnvironment(render_mode=None)
    env.reset(seed=0)
    env.set_pose((0.0, 0.0, 1.5), yaw=0.0)
    hover = env.hover_action
    h0 = heading_from_quaternion(env.get_fly_orientation())
    for _ in range(220):
        env.step(np.array([0.38, hover, 0.0, 0.0, 0.0]), wing_strokes=(hover, hover))
    h1 = heading_from_quaternion(env.get_fly_orientation())
    env.close()
    d = float(np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0)))
    assert abs(d) < 0.01, f"physics-only drift {d:.4f} rad"
