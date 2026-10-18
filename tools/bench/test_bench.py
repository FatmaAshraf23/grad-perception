#!/usr/bin/env python3
"""Self-tests for the bench tools with SYNTHETIC data whose true answers we know.
Run:  python3 test_bench.py        (or: python3 -m pytest -q test_bench.py)
The idea: if the tool cannot recover numbers we PUT IN, it cannot be trusted on the real sensor."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imu_static_check as imu          # noqa: E402
import lidar_range_check as lid         # noqa: E402


# ----------------------------------------------------------------------------- LiDAR
def fake_wall_scans(D, offset=0.03, scale=0.0, sigma=0.02, phi_deg=7.0, n_scans=60, seed=1,
                    dropout=0.03, box=True, beam_bias=0.004):
    """360 beams (1 deg), flat wall at perpendicular distance D whose normal is phi_deg away
    from the LiDAR's zero direction.  reading = (D/cos(a-phi))*(1+scale) + offset + noise."""
    rng = np.random.default_rng(seed)
    ang = np.deg2rad(np.arange(-180, 180, 1.0))
    phi = np.deg2rad(phi_deg)
    delta = lid.wrap(ang - phi)
    wall = np.abs(delta) < np.deg2rad(75)
    base = np.where(wall, D / np.cos(delta), rng.uniform(3.5, 7.0, ang.size))     # elsewhere: far room
    bias_beam = beam_bias * rng.standard_normal(ang.size)                         # per-beam systematic
    R = np.empty((n_scans, ang.size))
    for s in range(n_scans):
        r = base * (1 + scale) + offset + bias_beam + sigma * rng.standard_normal(ang.size)
        if box:                                                                    # object in front of the wall
            hit = wall & (np.abs(delta - np.deg2rad(20)) < np.deg2rad(4))
            r[hit] = 0.7 * D / np.cos(delta[hit]) + 0.0
        r[rng.random(ang.size) < dropout] = np.inf
        R[s] = r
    A = np.tile(ang, (n_scans, 1))
    return lid.Scans(R, A, 0.15, 12.0)


def test_lidar_recovers_offset_and_noise():
    sc = fake_wall_scans(2.0, offset=0.030, sigma=0.02)
    r, _ = lid.analyse_wall(sc, 2.0)
    assert abs(r["offset"] - 0.030) < 0.003, r["offset"]
    assert abs(r["sigma_repeat"] - 0.02) < 0.003, r["sigma_repeat"]
    assert abs(r["wall_normal_deg"] - 7.0) < 0.3, r["wall_normal_deg"]
    assert r["dropout"] > 0.0


def test_lidar_plusminus_is_honest():
    """over many random recordings the true offset must lie within ~2 of the reported '+-' most of the time"""
    inside = 0
    for seed in range(20):
        r, _ = lid.analyse_wall(fake_wall_scans(2.0, offset=0.022, seed=100 + seed, n_scans=40), 2.0)
        inside += abs(r["offset"] - 0.022) < 2.0 * r["offset_se"]
    assert inside >= 15, f"true value inside 2*se in only {inside}/20 recordings"


def test_lidar_negative_offset_and_other_orientation():
    sc = fake_wall_scans(3.0, offset=-0.015, phi_deg=-25.0, seed=4)
    r, _ = lid.analyse_wall(sc, 3.0)
    assert abs(r["offset"] + 0.015) < 0.003, r["offset"]


def test_lidar_scale_error_over_distance():
    res = []
    for D in (1.0, 2.0, 3.0, 4.0, 5.0):
        r, _ = lid.analyse_wall(fake_wall_scans(D, offset=0.02, scale=0.01, seed=int(D * 10)), D)
        res.append(r)
    m = lid.fit_distance_model(res)
    assert abs(m["a"] - 0.02) < 0.004, m
    assert abs(m["e"] - 0.01) < 0.0015, m
    assert m["scale_matters"] is True
    res2 = [lid.analyse_wall(fake_wall_scans(D, offset=0.02, scale=0.0, seed=int(D * 10)), D)[0] for D in (1.0, 2.0, 3.0, 4.0, 5.0)]
    m2 = lid.fit_distance_model(res2)
    assert m2["scale_matters"] is False, m2
    assert abs(m2["mean_offset"] - 0.02) < 0.003


def test_lidar_wrong_distance_is_caught():
    sc = fake_wall_scans(2.0, offset=0.0)
    try:
        r, _ = lid.analyse_wall(sc, 3.0)
    except ValueError:
        return                                  # refused: fine
    assert any("differs" in w for w in r["warnings"]), r["warnings"]


def test_lidar_end_to_end_cli():
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for D in (1.0, 2.0, 4.0):
            sc = fake_wall_scans(D, offset=0.022, seed=int(D * 7))
            p = os.path.join(d, f"s{D}.npz")
            np.savez(p, ranges=sc.ranges, angle_min=sc.angles[0, 0], angle_increment=1.0 * np.pi / 180,
                     range_min=0.15, range_max=12.0)
            paths.append(f"{D}:{p}")
        args = []
        for p in paths:
            args += ["--run", p]
        code = lid.main(args + ["--out", os.path.join(d, "out")])
        assert code in (0, 3)
        assert os.path.exists(os.path.join(d, "out", "result.json"))
        assert os.path.exists(os.path.join(d, "out", "report.txt"))


# ----------------------------------------------------------------------------- IMU
def fake_imu(hours=1.0, fs=100.0, sample_std=0.0015, K=2e-5, bias=0.01, seed=0, jitter=0.0):
    """gz with white noise (std per sample), constant bias, bias random walk K [rad/s/sqrt(s)]."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    n = int(hours * 3600 * fs)
    walk = np.cumsum(K * np.sqrt(dt) * rng.standard_normal(n))
    gz = bias + walk + sample_std * rng.standard_normal(n)
    gx = 0.002 + sample_std * rng.standard_normal(n)
    gy = -0.003 + sample_std * rng.standard_normal(n)
    t = np.arange(n) * dt + 100.0 + jitter * rng.standard_normal(n)
    t = np.maximum.accumulate(t)
    return t, np.stack([gx, gy, gz], axis=1)


def test_allan_recovers_noise_terms():
    t, g = fake_imu(hours=1.0, sample_std=0.0015, K=2e-5)
    r = imu.analyse_gyro(t, g)
    z = r["axes"][2]
    N_true = 0.0015 * np.sqrt(0.01)
    assert abs(z["N"] / N_true - 1) < 0.10, (z["N"], N_true)
    assert abs(z["white_per_sample"] / 0.0015 - 1) < 0.10
    assert abs(z["K"] / 2e-5 - 1) < 0.30, z["K"]
    assert abs(z["bias"] - 0.01) < 5e-4
    assert not r["warnings"], r["warnings"]


def test_allan_no_drift_gives_tiny_K():
    t, g = fake_imu(hours=0.5, K=0.0, seed=3)
    z = imu.analyse_gyro(t, g)["axes"][2]
    assert z["K"] < 5e-6, z["K"]


def test_timestamp_jitter_is_fine_but_gaps_are_flagged():
    t, g = fake_imu(hours=0.2, seed=5, jitter=0.0005)
    r = imu.analyse_gyro(t, g)
    assert r["timing"]["gaps"] == 0
    keep = np.ones(len(t), bool); keep[5000:5060] = False        # 0.6 s dropped
    r2 = imu.analyse_gyro(t[keep], g[keep])
    assert r2["timing"]["gaps"] == 1 and any("gap" in w for w in r2["warnings"])


def test_motion_is_flagged():
    t, g = fake_imu(hours=0.3, seed=6)
    i0 = int(500 / 0.01)
    g[i0:i0 + 200, 2] += np.deg2rad(5.0)                           # turned for 2 s
    r = imu.analyse_gyro(t, g)
    assert r["motion_windows"] >= 1 and any("still" in w for w in r["warnings"])


def test_csv_loading_units_and_turn_on_scatter():
    with tempfile.TemporaryDirectory() as d:
        paths, biases = [], []
        for k, b in enumerate((0.010, 0.013, 0.008)):
            t, g = fake_imu(hours=0.1, bias=b, seed=10 + k, K=0.0)
            p = os.path.join(d, f"run{k}.csv")
            with open(p, "w") as f:
                f.write("t,gx,gy,gz\n")
                np.savetxt(f, np.column_stack([t * 1e3, np.rad2deg(g)]), delimiter=",", fmt="%.9g")
            paths.append(p); biases.append(b)
        rs = []
        for p in paths:
            tt, gg, _ = imu.load_imu(p, gyro_units="dps", time_units="ms")
            rs.append(imu.analyse_gyro(tt, gg))
        got = np.array([r["axes"][2]["bias"] for r in rs])
        assert np.allclose(got, biases, atol=3e-4), (got, biases)
        txt = imu.turn_on_report(paths, rs)
        assert "TURN-ON" in txt
        assert imu.main(paths + ["--gyro-units", "dps", "--time-units", "ms", "--out", os.path.join(d, "o")]) in (0, 3)
        assert os.path.exists(os.path.join(d, "o", "report.txt"))


def test_accel_gravity_check():
    t, g = fake_imu(hours=0.1, seed=2)
    a = np.zeros((len(t), 3)); a[:, 2] = 9.81 * 1.08                # 8 % too large
    r = imu.analyse_gyro(t, g, a)
    assert any("gravity" in w for w in r["warnings"])


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as ex:
            fails += 1; print(f"FAIL  {name}: {type(ex).__name__}: {ex}")
    print(f"\n{len(tests) - fails} of {len(tests)} passed")
    sys.exit(1 if fails else 0)
