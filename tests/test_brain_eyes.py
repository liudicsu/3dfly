"""BoxEye-equivalent hex sampling (no flyvis / connectome)."""

from __future__ import annotations

import numpy as np
import pytest

from threedfly.brain.constants import N_HEX
from threedfly.brain.eyes import (
    BIO_AZ_CONTRA_DEG,
    EYE_CANT_DEG,
    EYE_DOWN_DEG,
    EYE_FOVY_DEG,
    box_eye_uv,
    hex_body_azel_deg,
    hex_coverage,
    hex_in_bio_envelope,
    image_to_hex,
    stereo_to_hex,
)


def test_box_eye_has_721_cells():
    uv = box_eye_uv()
    assert uv.shape == (N_HEX, 2)
    assert uv.min() >= 0.0 and uv.max() <= 1.0


def test_image_to_hex_samples_blob_laterality():
    h = w = 64
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Bright patch on the left half of the image (small u).
    img[:, : w // 4] = 255
    hex_frame = image_to_hex(img)
    assert hex_frame.shape == (N_HEX,)
    assert hex_frame.max() > 0.5
    uv = box_eye_uv()
    left = hex_frame[uv[:, 0] < 0.35].mean()
    right = hex_frame[uv[:, 0] > 0.65].mean()
    assert left > right + 0.2


def test_right_eye_looks_forward_not_backward():
    """Both cameras look +X. Laterality is hex rays, not a 40° cant."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "threedfly" / "sim" / "mjcf.py"
    spec = importlib.util.spec_from_file_location("threedfly_mjcf_camera", path)
    mjcf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mjcf)
    look_L = mjcf.camera_look_direction(mjcf.parse_eye_xyaxes(mjcf.FLY_MJCF, "left_eye"))
    look_R = mjcf.camera_look_direction(mjcf.parse_eye_xyaxes(mjcf.FLY_MJCF, "right_eye"))
    assert look_L[0] > 0.95, f"left eye should look +X, got {look_L}"
    assert look_R[0] > 0.95, f"right eye should look +X, got {look_R}"
    assert abs(look_L[1]) < 0.1 and abs(look_R[1]) < 0.1
    assert 'fovy="150"' in mjcf.FLY_MJCF
    assert 'type="skybox"' in mjcf.FLY_MJCF
    xml_L = mjcf.parse_eye_xyaxes(mjcf.FLY_MJCF, "left_eye")
    xml_R = mjcf.parse_eye_xyaxes(mjcf.FLY_MJCF, "right_eye")
    assert xml_L == pytest.approx(mjcf.eye_xyaxes("L"), abs=0.002)
    assert xml_R == pytest.approx(mjcf.eye_xyaxes("R"), abs=0.002)
    assert mjcf.EYE_FOVY_DEG == pytest.approx(150.0)
    assert mjcf.EYE_CANT_DEG == pytest.approx(0.0)


def test_walls_use_natural_one_over_f():
    from threedfly.sim import mjcf as mjcf_mod

    xml = mjcf_mod.FLY_MJCF
    assert "wall_nat" in xml
    assert "wall_check" not in xml
    assert "room_natural" in xml or "{wall_tex}" in xml
    assert "rgba=\"0.8 0.8 0.9 1\"" not in xml
    assert 'name="loom_target"' in xml
    assert 'mocap="true"' in xml
    assert 'size="0.58"' in xml
    assert "loom_mat" in xml
    assert 'rgba="0 0 0 1"' in xml
    assert "specular=\"0\"" in xml
    assert "loom_check" not in xml
    path = mjcf_mod.write_one_over_f_png(mjcf_mod.ASSETS_ROOT / "room_natural.png")
    assert path.is_file() and path.stat().st_size > 100


def test_mirror_stereo_images_helper_removed():
    """Swap+fliplr is not a rendered mirror (DNp01 err 0.33 vs 0.03)."""
    from threedfly.brain import eyes as eyes_mod

    assert not hasattr(eyes_mod, "mirror_stereo_images")


def test_hex_fovy_matches_camera():
    from threedfly.sim import mjcf as mjcf_mod

    assert EYE_FOVY_DEG == pytest.approx(mjcf_mod.EYE_FOVY_DEG)
    assert EYE_FOVY_DEG == pytest.approx(150.0)
    assert EYE_CANT_DEG == pytest.approx(mjcf_mod.EYE_CANT_DEG)
    assert EYE_DOWN_DEG == pytest.approx(mjcf_mod.EYE_DOWN_DEG)


def test_legacy_pack_put_centre_on_optical_axis():
    """The old 40°/20°/120° pack is why 3D looms missed LPLC2."""
    from threedfly.brain.eyes import legacy_packed_azel_deg

    az, el, _ = legacy_packed_azel_deg("L")
    az0, el0 = hex_body_azel_deg("L")
    i0 = int(np.argmin(az0**2 + el0**2))
    assert abs(float(az[i0]) - 40.0) < 3.0
    assert abs(float(el[i0]) + 20.0) < 3.0


def test_hex_rays_put_lattice_centre_on_body_frontal():
    """flyvis (0,0) must look at az≈0, not the old 40° optical axis."""
    az_L, el_L = hex_body_azel_deg("L")
    az_R, el_R = hex_body_azel_deg("R")
    i0 = int(np.argmin(az_L**2 + el_L**2))
    assert abs(float(az_L[i0])) < 4.0
    assert abs(float(el_L[i0])) < 4.0
    assert float(az_R[i0]) == pytest.approx(-float(az_L[i0]), abs=1e-4)
    # Ipsilateral frontolat 20–70° is on-eye, not clipped to the nasal edge.
    cov = hex_coverage("L")
    in_fl = (cov["az_deg"] >= 20.0) & (cov["az_deg"] <= 70.0) & (np.abs(cov["el_deg"]) < 15.0)
    assert int(np.count_nonzero(in_fl)) > 20
    assert float(np.mean(cov["valid"][in_fl])) > 0.9
    # Biological envelope: ipsilateral hexes stay under 170°, some cross midline.
    assert float(np.max(cov["az_deg"])) < BIO_AZ_CONTRA_DEG + 160.0
    assert float(np.min(cov["az_deg"])) < 0.0
    # Contra beyond ~15° is unseen on that eye (real overlap is 10–20°).
    contra = cov["az_deg"] < -BIO_AZ_CONTRA_DEG
    assert int(np.count_nonzero(contra)) > 10
    assert not np.any(cov["valid"][contra])


def test_stereo_to_hex_pair():
    from threedfly.brain.eyes import hex_camera_uv

    left = np.full((32, 32, 3), 128, dtype=np.uint8)
    right = np.zeros((32, 32, 3), dtype=np.uint8)
    hL, hR = stereo_to_hex(left, right)
    assert hL.shape == hR.shape == (N_HEX,)
    _, ok_L = hex_camera_uv("L", 32, 32)
    _, ok_R = hex_camera_uv("R", 32, 32)
    # Uniform images sit at flyvis gray after median luminance shift.
    assert hL[ok_L].mean() == pytest.approx(0.5, abs=0.05)
    assert hR[ok_R].mean() == pytest.approx(0.5, abs=0.05)


def test_right_eye_hex_mirrors_body_azimuth():
    """A bright patch on the body-right (−Y) side of the image hits +xd on the right eye."""
    h = w = 64
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, w // 2 :] = 255  # image-right half = body −Y
    left = np.zeros_like(img)
    hL, hR = stereo_to_hex(left, img)
    az_R, _ = hex_body_azel_deg("R")
    from threedfly.brain.eyes import hex_camera_uv

    _, ok_R = hex_camera_uv("R", h, w)
    # Right-eye +xd looks to −az (body right / image-right). Those hexes
    # should be brighter than the nasal overlap (image-left of centre).
    ipsi = hR[ok_R & (az_R < -15.0)].mean()
    nasal = hR[ok_R & (az_R > 5.0)].mean()
    assert ipsi > nasal + 0.15, f"ipsi={ipsi:.3f} nasal={nasal:.3f}"
    # Left image is uniform: luminance shift puts in-view hexes at gray.
    from threedfly.brain.eyes import hex_camera_uv

    _, ok_L = hex_camera_uv("L", h, w)
    assert float(hL[ok_L].mean()) == pytest.approx(0.5, abs=0.05)


def test_frontolat_body_az_is_monocular():
    """+35° is on the left eye only (right contra cut is +15°)."""
    az_L, _ = hex_body_azel_deg("L")
    az_R, _ = hex_body_azel_deg("R")
    assert bool(hex_in_bio_envelope("L", np.array([35.0]))[0])
    assert not bool(hex_in_bio_envelope("R", np.array([35.0]))[0])
    assert bool(hex_in_bio_envelope("L", np.array([0.0]))[0])
    assert bool(hex_in_bio_envelope("R", np.array([0.0]))[0])
    # Lattice sites that look at +35° exist on both eyes; only L is valid.
    near_L = np.abs(az_L - 35.0) < 8.0
    near_R = np.abs(az_R - 35.0) < 8.0
    cov_L = hex_coverage("L")
    cov_R = hex_coverage("R")
    assert int(np.count_nonzero(near_L)) > 0
    assert int(np.count_nonzero(near_R)) > 0
    assert float(np.mean(cov_L["valid"][near_L])) > 0.8
    assert float(np.mean(cov_R["valid"][near_R])) < 0.2


def test_camera_disc_lands_on_hex_disc_site():
    """3D disc at body az 35° darkens the same left-eye hexes as a hex disc.

    Rendered pixels only — no object-pose paint. Gaussian acceptance +
    luminance shift keep the disc darker than the background, not 0.00.
    """
    pytest.importorskip("mujoco")
    from threedfly.brain.stimuli import hex_degrees
    from threedfly.control.closed_loop_metrics import LOOM_RADIUS_M
    from threedfly.sim import FlyEnvironment

    ang = 40.0
    d = float(LOOM_RADIUS_M) / max(float(np.tan(np.radians(0.5 * ang))), 1e-6)
    az = np.radians(35.0)
    env = FlyEnvironment(render_mode="rgb_array", camera_width=160, camera_height=160)
    env.reset(seed=0)
    env.set_pose((0.0, 0.0, 1.5), yaw=0.0, vel=(0.0, 0.0, 0.0))
    env.set_loom_object((d * np.cos(az), d * np.sin(az), 1.5))
    hL, hR = stereo_to_hex(*env.get_eye_images())
    env.close()
    _, _, xd, yd = hex_degrees()
    disc_L = np.hypot(xd - 35.0, yd) <= 0.5 * ang
    # Right-eye xd=−35 looks at body +35° — outside the 15° contra envelope.
    disc_R_frontolat = np.hypot(xd + 35.0, yd) <= 0.5 * ang
    assert int(np.count_nonzero(disc_L)) > 10
    az_L, el_L = hex_body_azel_deg("L")
    bg = (~disc_L) & hex_in_bio_envelope("L", az_L) & (np.abs(el_L) < 40.0)
    assert float(hL[disc_L].mean()) < 0.35
    assert float(hR[disc_R_frontolat].mean()) > 0.35
    assert float(hL[bg].mean()) > float(hL[disc_L].mean()) + 0.12


def test_stereo_to_hex_has_no_loom_oracle():
    """Eye input must not accept sim object pose/radius."""
    import inspect

    from threedfly.brain import closed_loop as cl
    from threedfly.brain import eyes as eyes_mod
    from threedfly.sim import environment as env_mod

    sig = inspect.signature(eyes_mod.stereo_to_hex)
    assert "loom_body" not in sig.parameters
    assert "loom_radius" not in sig.parameters
    step = inspect.signature(cl.ClosedLoopBrain.step)
    assert "loom_body" not in step.parameters
    assert "loom_radius" not in step.parameters
    assert not hasattr(eyes_mod, "paint_loom_rays")
    assert not hasattr(eyes_mod, "angular_loom_mask")
    assert not hasattr(eyes_mod, "rays_hit_sphere")
    assert not hasattr(env_mod.FlyEnvironment, "loom_centers_body")
    assert not hasattr(env_mod.FlyEnvironment, "loom_radius")


def test_empty_room_hex_is_left_right_symmetric():
    """Mirrored empty room: L/R hex means match, so L DC is not an eye-mask artifact."""
    pytest.importorskip("mujoco")
    from threedfly.sim import FlyEnvironment

    env = FlyEnvironment(render_mode="rgb_array", camera_width=160, camera_height=160)
    env.reset(seed=0)
    env.set_pose((0.0, 0.0, 1.5), yaw=0.0, vel=(0.0, 0.0, 0.0))
    env.hide_loom_object()
    left, right = env.get_eye_images()
    hL, hR = stereo_to_hex(left, right)
    env.close()
    from threedfly.brain.eyes import hex_camera_uv

    _, ok_L = hex_camera_uv("L", 160, 160)
    _, ok_R = hex_camera_uv("R", 160, 160)
    mean_L = float(hL[ok_L].mean())
    mean_R = float(hR[ok_R].mean())
    std_L = float(hL[ok_L].std())
    std_R = float(hR[ok_R].std())
    # After luminance-norm both sit near 0.5; L vs R should match.
    assert abs(mean_L - mean_R) < 0.04, f"empty-room hex L={mean_L:.3f} R={mean_R:.3f}"
    assert abs(std_L - std_R) < 0.05, f"empty-room hex std L={std_L:.3f} R={std_R:.3f}"
