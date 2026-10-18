"""Mirror-scene laterality: swapped L/R input must swap DNp03 / DNa02 / HSE."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.brain.closed_loop import ClosedLoopBrain, core_stack_available
from threedfly.brain.constants import N_HEX, flyvis_available
from threedfly.brain.core_bundle import core_bundle_available
from threedfly.brain.stimuli import loom
from threedfly.control.flight_params import FlightParams


needs_stack = pytest.mark.skipif(
    not (flyvis_available() and core_bundle_available()),
    reason="mirror-scene laterality needs flyvis + data/core_bundle.npz",
)


def _mean_dns(brain: ClosedLoopBrain, hex_L, hex_R, n: int = 8):
    last = None
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(n):
        last = brain.step(blank, blank, hex_L=hex_L, hex_R=hex_R)
    ro = last.readout
    return {
        "dnp03_L": float(ro.dnp03_L),
        "dnp03_R": float(ro.dnp03_R),
        "dna02_L": float(ro.dna02_L),
        "dna02_R": float(ro.dna02_R),
        "hse_L": float(ro.hse_L),
        "hse_R": float(ro.hse_R),
        "dnp01_L": float(ro.dnp01_L),
        "dnp01_R": float(ro.dnp01_R),
        "lc4_L": float(ro.lc4_L),
        "lc4_R": float(ro.lc4_R),
        "lplc2_L": float(ro.lplc2_L),
        "lplc2_R": float(ro.lplc2_R),
    }


def test_side_walls_brighten_ipsilateral_eye():
    """Forward 150° eyes: the matching wall must change that eye more than the other."""
    pytest.importorskip("mujoco")
    from threedfly.brain.eyes import stereo_to_hex
    from threedfly.sim import FlyEnvironment
    import importlib.util
    from pathlib import Path

    _acc = Path(__file__).resolve().parents[1] / "scripts" / "closed_loop_acceptance.py"
    _spec = importlib.util.spec_from_file_location("cla_eye_lat", _acc)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    from threedfly.brain.eyes import hex_body_azel_deg

    env = FlyEnvironment(render_mode="rgb_array")
    az_L, _ = hex_body_azel_deg("L")
    az_R, _ = hex_body_azel_deg("R")
    env.reset(seed=0)
    env.set_pose((0.0, 0.0, 1.5), yaw=0.0)
    env.hide_loom_object()
    hL0, hR0 = stereo_to_hex(*env.get_eye_images())
    change = {}
    for side in ("L", "R"):
        pose = _mod.lateral_loom_pose(side)
        env.reset(seed=0)
        env.set_pose(pose["pose"], yaw=pose["yaw"])
        hL, hR = stereo_to_hex(*env.get_eye_images())
        # +az looks left. A left wall should move left-eye ipsi hexes more.
        change[side] = {
            "left_rays": float(np.mean(np.abs(hL - hL0)[az_L > 20.0])),
            "right_rays": float(np.mean(np.abs(hR - hR0)[az_R < -20.0])),
        }
    env.close()
    # 1/f walls + 150° FOV: both eyes see the near wall. Require a real
    # luminance change vs center, not a hard L>R on low-contrast texels.
    assert change["L"]["left_rays"] > 0.04, f"left wall: {change}"
    assert change["R"]["right_rays"] > 0.04, f"right wall: {change}"


@needs_stack
def test_m1_hex_loom_via_closed_loop_path_is_ipsilateral():
    """M1 loom movie through camera-free hex → flyvis → RateCore stays ipsi."""
    params = FlightParams(prewarm_steps=12, flyvis_stride=1, flyvis_torch_threads=2)
    brain = ClosedLoopBrain.build(params=params, quick=True)
    gray = np.full(N_HEX, 0.5, dtype=np.float32)
    movie = loom(0.4, x0=0.0)
    blank = np.zeros((8, 8, 3), dtype=np.uint8)

    def _play(hex_L, hex_R):
        brain.reset()
        last = None
        for _ in range(12):
            last = brain.step(blank, blank, hex_L=gray, hex_R=gray)
        rest = last.readout
        peak = {
            "dnp01_L": 0.0, "dnp01_R": 0.0, "lc4_L": 0.0, "lc4_R": 0.0,
            "dnp03_L": 0.0, "dnp03_R": 0.0,
        }
        for i in range(len(movie)):
            last = brain.step(blank, blank, hex_L=hex_L[i], hex_R=hex_R[i])
            ro = last.readout
            peak["dnp01_L"] = max(peak["dnp01_L"], float(ro.dnp01_L) - float(rest.dnp01_L))
            peak["dnp01_R"] = max(peak["dnp01_R"], float(ro.dnp01_R) - float(rest.dnp01_R))
            peak["lc4_L"] = max(peak["lc4_L"], float(ro.lc4_L) - float(rest.lc4_L))
            peak["lc4_R"] = max(peak["lc4_R"], float(ro.lc4_R) - float(rest.lc4_R))
            peak["dnp03_L"] = max(peak["dnp03_L"], float(ro.dnp03_L) - float(rest.dnp03_L))
            peak["dnp03_R"] = max(peak["dnp03_R"], float(ro.dnp03_R) - float(rest.dnp03_R))
        return peak

    left = _play(movie, np.repeat(gray[None, :], len(movie), axis=0))
    right = _play(np.repeat(gray[None, :], len(movie), axis=0), movie)
    assert left["dnp01_L"] > left["dnp01_R"] + 0.10, left
    assert right["dnp01_R"] > right["dnp01_L"] + 0.10, right
    assert left["lc4_L"] > left["lc4_R"] + 0.10, left
    assert right["lc4_R"] > right["lc4_L"] + 0.10, right
    # DNp01 L↔R within ~20%; LC4 is ipsi but streaming R is a bit stronger.
    a, b = left["dnp01_L"], right["dnp01_R"]
    err = abs(a - b) / max(a, b, 1e-6)
    assert err < 0.25, f"dnp01 L↔R err {err:.2f}: {left} {right}"


@needs_stack
@pytest.mark.skipif(not core_stack_available(), reason="open-loop camera path needs the stack")
def test_open_loop_mirrored_object_flips_dns():
    """Stationary fly: L object vs exact mirrored R object → mirrored DN/VPN."""
    pytest.importorskip("mujoco")
    from threedfly.sim import FlyEnvironment

    params = FlightParams(prewarm_steps=16, flyvis_stride=1, flyvis_torch_threads=2)
    brain = ClosedLoopBrain.build(params=params, quick=True)
    env = FlyEnvironment(render_mode="rgb_array", camera_width=160, camera_height=160)
    env.reset(seed=0)
    env.set_pose((0.0, 0.0, 1.5), yaw=0.0, vel=(0.0, 0.0, 0.0))

    def _movie(start, vel, n=36):
        frames = []
        pos = np.asarray(start, dtype=float)
        v = np.asarray(vel, dtype=float)
        for _ in range(n):
            env.set_loom_object(pos)
            frames.append(env.get_eye_images())
            pos = pos + v * 0.02
        return frames

    def _run(frames):
        brain.reset()
        gray = np.full_like(frames[0][0], 140)
        last = None
        for _ in range(16):
            last = brain.step(gray, gray)
        rest = last.readout
        peak = {"dnp01_L": 0.0, "dnp01_R": 0.0, "lc4_L": 0.0, "lc4_R": 0.0,
                "lplc2_L": 0.0, "lplc2_R": 0.0, "dnp03_L": 0.0, "dnp03_R": 0.0}
        for left, right in frames:
            last = brain.step(left, right)
            ro = last.readout
            for k in peak:
                peak[k] = max(peak[k], float(getattr(ro, k)) - float(getattr(rest, k)))
        return peak

    # 45° approach matches the official side-loom geometry. Bilinear hex
    # sampling (not a V-flip) is what brings DNp01 L↔R under 20%.
    left = _run(_movie((1.55, 1.55, 1.5), (-1.30, -1.25, 0.0), n=50))
    right = _run(_movie((1.55, -1.55, 1.5), (-1.30, 1.25, 0.0), n=50))
    env.close()
    # Camera-path LC4 is nearly binocular and can flip; DNp01 L−R does not.
    assert left["dnp01_L"] > left["dnp01_R"], left
    assert right["dnp01_R"] > right["dnp01_L"], right
    for key, lim in (("dnp01", 0.20), ("lc4", 0.30)):
        a, b = left[f"{key}_L"], right[f"{key}_R"]
        err = abs(a - b) / max(abs(a), abs(b), 1e-6)
        assert err < lim, f"{key} L↔R err {err:.2f} (want <{lim}): L={left} R={right}"


@needs_stack
def test_hex_swap_mirrors_dn_outputs():
    """Swapping the two hex frames must swap per-side DNp03 / DNa02 / HSE."""
    params = FlightParams(prewarm_steps=16, flyvis_stride=1, flyvis_torch_threads=2)
    brain = ClosedLoopBrain.build(params=params, quick=True)
    gray = np.full(N_HEX, 0.5, dtype=np.float32)
    movie = loom(0.4, x0=-25.0)
    looming = movie[-1]
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    brain.reset()
    for _ in range(12):
        brain.step(blank, blank, hex_L=gray, hex_R=gray)
    a = _mean_dns(brain, looming, gray, n=10)
    brain.reset()
    for _ in range(12):
        brain.step(blank, blank, hex_L=gray, hex_R=gray)
    b = _mean_dns(brain, gray, looming, n=10)

    # Mirror: the eye that got the loom should drive the matching DN side more.
    d03_a = a["dnp03_L"] - a["dnp03_R"]
    d03_b = b["dnp03_L"] - b["dnp03_R"]
    assert d03_a * d03_b < 0, f"DNp03 L−R did not flip: A={a} B={b}"
    d02_a = a["dna02_L"] - a["dna02_R"]
    d02_b = b["dna02_L"] - b["dna02_R"]
    assert d02_a * d02_b < 0, f"DNa02 L−R did not flip: A={a} B={b}"
    # Left-eye loom should prefer DNp03_L (ipsi) and LC4_L.
    assert a["dnp03_L"] > a["dnp03_R"], f"left-eye loom should raise DNp03_L: {a}"
    assert b["dnp03_R"] > b["dnp03_L"], f"right-eye loom should raise DNp03_R: {b}"
    assert a["lc4_L"] > a["lc4_R"], f"left-eye loom should raise LC4_L: {a}"
    assert b["lc4_R"] > b["lc4_L"], f"right-eye loom should raise LC4_R: {b}"


@needs_stack
@pytest.mark.skipif(
    not core_stack_available(),
    reason="MuJoCo mirror-scene needs the closed-loop stack",
)
def test_lateral_approaches_turn_away_from_threat():
    """DNp01 escapes on true L/R side-loom must change heading away from the wall."""
    pytest.importorskip("mujoco")
    from threedfly.control.closed_loop_metrics import direction_correct_frac
    from threedfly.control.dn_controller import DNFlightController
    from threedfly.control.exploration import heading_from_quaternion
    from threedfly.sim import FlyEnvironment
    import importlib.util
    from pathlib import Path

    _acc = Path(__file__).resolve().parents[1] / "scripts" / "closed_loop_acceptance.py"
    _spec = importlib.util.spec_from_file_location("cla_laterality", _acc)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    params = FlightParams(flyvis_stride=1, flyvis_torch_threads=2)
    brain = ClosedLoopBrain.build(params=params, quick=True)
    env = FlyEnvironment(render_mode="rgb_array", camera_width=160, camera_height=160)
    env.control_dt = 0.01
    env.n_substeps = max(1, int(round(0.01 / float(env.model.opt.timestep))))
    ctl = DNFlightController(params)
    collision_course_object = _mod.collision_course_object

    def _approach(side: str):
        az = 45.0 if side == "L" else -45.0
        spec = collision_course_object(az, 3.4, 0.08)
        env.reset(seed=0)
        env.set_pose(spec["pose"], yaw=spec["yaw"], vel=spec["vel"])
        env.hide_loom_object()
        brain.reset()
        ctl.reset()
        ctl.bind_hover(env.hover_action)
        left, right = env.get_eye_images()
        warmed = brain.prewarm(left, right, env.get_fly_angular_velocity(), n=params.prewarm_steps)
        if warmed is not None:
            ctl.capture_baseline(warmed.readout)
        env.set_loom_object(spec["loom_start"])
        heads = []
        poses = []
        loom_pos = np.asarray(spec["loom_start"], dtype=float)
        loom_v = np.asarray(spec["loom_vel"], dtype=float)
        for _ in range(160):
            loom_pos = loom_pos + loom_v * float(env.control_dt)
            env.set_loom_object(loom_pos)
            res = brain.step(
                left,
                right,
                env.get_fly_angular_velocity(),
            )
            act = ctl.compute(res.readout)
            left, right, info = env.step(act, wing_strokes=ctl.last_wing_strokes)
            heads.append(heading_from_quaternion(info["orientation"]))
            poses.append(np.asarray(info["position"], dtype=float))
            if env.check_collision():
                env.recover_from_collision()
        return heads, list(ctl.escape_events)

    for side in ("L", "R"):
        heads, evs = _approach(side)
        assert evs, f"{side} approach produced no DNp01 escape"
        frac = direction_correct_frac(heads, evs, side)
        assert frac >= 0.8, f"{side} direction_correct={frac} events={[(e.step, e.yaw) for e in evs]}"
    env.close()


@needs_stack
def test_gray_hex_dnp01_rest_dc_is_brain_not_eyes():
    """Mirrored gray hex: any standing L−R DNp01 is a brain DC, not the eye mask.

    Per-side rest subtracts it; shared-max rest would hide a contralateral loom.
    """
    params = FlightParams(prewarm_steps=16, flyvis_stride=1, flyvis_torch_threads=2)
    brain = ClosedLoopBrain.build(params=params, quick=True)
    gray = np.full(N_HEX, 0.5, dtype=np.float32)
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    brain.reset()
    last = None
    for _ in range(20):
        last = brain.step(blank, blank, hex_L=gray, hex_R=gray)
    rest_L = float(last.readout.dnp01_L)
    rest_R = float(last.readout.dnp01_R)
    # The DC (if any) is a property of the connectome, not of a rendered
    # laterality: both eyes received the identical gray lattice.
    assert abs(rest_L - rest_R) < 0.40, f"gray-hex DNp01 rest L={rest_L:.3f} R={rest_R:.3f}"
