"""Brain panel shows real, non-empty activity (PR1)."""

import socket

import numpy as np
import pytest

from threedfly.brain.readouts import DNReadout
from threedfly.viz.brain_panel import (
    BrainPanelModel,
    log_bar_fraction,
    population_stats,
    top_active_indices,
)


def _sparse_rates(n=45_000, n_active=40, seed=0):
    rng = np.random.default_rng(seed)
    r = np.zeros(n, dtype=np.float32)
    stride = set(np.linspace(0, n - 1, 200, dtype=int).tolist())
    pool = np.array([i for i in range(n) if i not in stride])
    hot = rng.choice(pool, n_active, replace=False)
    r[hot] = rng.uniform(1, 20, n_active)
    return r, np.sort(hot)


def test_stride_sample_misses_sparse_activity_but_top_k_does_not():
    r, hot = _sparse_rates()
    stride = np.linspace(0, len(r) - 1, 200, dtype=int)
    assert r[stride].max() == 0.0  # the old panel's view: nothing
    top = top_active_indices(r, 200)
    assert set(hot) <= set(top.tolist())
    assert np.all(np.diff(top) > 0)


def test_population_stats():
    r, _ = _sparse_rates()
    s = population_stats(r)
    assert s.n == r.size and s.n_active == 40
    assert s.max == pytest.approx(float(r.max()))
    assert 0 < s.active_fraction < 0.01
    assert population_stats(np.zeros(0)).n == 0


def test_log_bar_fraction_monotone_and_visible_for_small_rates():
    vals = [0, 1e-4, 1e-3, 1e-2, 1, 20]
    fr = [log_bar_fraction(v, 20) for v in vals]
    assert fr[0] == 0.0 and fr[-1] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(fr, fr[1:]))
    assert fr[2] > 0.05  # 1e-3 Hz still a visible bar


def test_group_means_top_types_and_traces():
    types = np.array(["T4a"] * 4 + ["T5b"] * 4 + ["LPLC2"] * 2 + ["DNp01"] * 2 + ["DNg02_a"] * 2 + ["x"] * 6)
    m = BrainPanelModel(types=types, history=5)
    r = np.zeros(len(types), dtype=np.float32)
    r[0:4] = 2.0
    r[10:12] = 7.0
    r[12:14] = 5.0
    for i in range(8):
        m.update(0.02 * i, r, DNReadout(dnp01_L=7.0, dnp01_R=7.0, dng02_L=5.0, dng02_R=5.0))
    gm = m.group_means()
    assert gm["T4 (ON motion)"] == pytest.approx(2.0)
    assert gm["DNp01 (giant fiber)"] == pytest.approx(7.0)
    assert gm["DNg02 (wing amp)"] == pytest.approx(5.0)
    assert gm["T5 (OFF motion)"] == 0.0
    assert m.top_types(2)[0][0] == "DNp01"
    t, *series = m.trace_arrays("dn")
    assert len(t) == 5  # ring buffer
    assert series[0][-1] == pytest.approx(7.0)
    html = m.groups_html()
    assert "DNp01" in html and "7.00e+00" in html
    assert "DNp01" in m.top_types_html()
    assert "active" in m.stats_markdown()
    m.reset_history()
    assert len(m.t) == 0


def test_legacy_model_without_types():
    m = BrainPanelModel(types=None)
    assert m.group_means() == {}
    m.update(0.0, np.ones(10))
    assert m.group_means() == {}
    assert m.top_types() == []
    assert m.last_stats.n_active == 10


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_interactive_visualizer_brain_panel_updates():
    pytest.importorskip("viser")
    from threedfly.viz.interactive import InteractiveVisualizer

    viz = InteractiveVisualizer(host="127.0.0.1", port=_free_port())
    try:
        types = np.array(["T4a"] * 50 + ["DNp01"] * 2 + ["LPLC2"] * 8)
        viz.set_brain_metadata(types=types)
        r = np.zeros(len(types), dtype=np.float32)
        r[50:52] = 3.0
        for i in range(3):
            viz.update(
                fly_position=np.array([0.0, 0.0, 1.0]),
                fly_orientation=np.array([1.0, 0, 0, 0]),
                brain_activity=r,
                control_action=np.zeros(5),
                brain_readout=DNReadout(dnp01_L=3.0, dnp01_R=3.0),
                sim_time=0.02 * i,
            )
        assert "60 neurons" in viz._brain_stats_md.content
        assert "DNp01" in viz._brain_groups_html.content
        assert viz._brain_dn_plot.data[1][-1] == pytest.approx(3.0)
        assert len(viz._brain_dn_plot.data[0]) == 3
    finally:
        viz.server.stop()
