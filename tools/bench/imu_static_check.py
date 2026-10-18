#!/usr/bin/env python3
"""IMU bench check: gyro bias, noise and drift from a recording of an IMU lying STILL.

WHY
  The EKF integrates the gyro to get the heading.  Two kinds of gyro error matter:
    * white noise   - each reading is a little wrong, randomly.  Integrated, it makes the
                      heading wander like a random walk ("angle random walk", ARW).
    * bias drift    - a slowly changing offset that is not zero when the car stands still.
                      The EKF estimates it as a state, but needs to know how fast it can
                      change (its "random walk", RRW) to choose how much to trust new data.
  The numbers below come from the ALLAN DEVIATION, the standard tool for exactly this.

HOW THE ALLAN DEVIATION WORKS (plain words)
  Cut the still recording into blocks of length tau, average the gyro inside every block,
  and look at how much neighbouring block averages differ:
        sigma_A(tau)^2 = 1/2 * mean( (avg_{k+1} - avg_k)^2 )
  * For very short tau the averages are dominated by white noise, which shrinks when you
    average longer: sigma_A falls like 1/sqrt(tau)  (slope -1/2 on a log-log plot).
  * For long tau the slow drift of the bias takes over and sigma_A rises like sqrt(tau)
    (slope +1/2).  The lowest point in between is the "bias instability".
  We fit   sigma_A^2(tau) = N^2/tau + (2 ln2/pi) B^2 + K^2 tau/3   and read:
      N  angle random walk      [rad/sqrt(s)]  -> per-sample noise  sigma = N / sqrt(dt)
      B  bias instability       [rad/s]        -> the floor the bias cannot be known below
      K  rate random walk       [rad/s/sqrt(s)] -> bias process noise for the EKF

USAGE
  python3 imu_static_check.py imu_run1.npz [imu_run2.npz ...] [--out result/]
  Several files = several power-ups: the scatter of their mean biases is the TURN-ON bias
  repeatability.  Accepts .npz (from bench_record.py) or .csv with a header line containing
  t, gx, gy, gz (and optionally ax, ay, az).  Options: --gyro-units rad|dps, --time-units s|ms|us|ns.
  Axis meaning: the axis pointing UP (gz when the board is flat) is the one that matters
  for the car's heading; the report prints all three.
EXIT 0 fine, 3 printed with warnings, 1 refused.
numpy only; matplotlib is optional.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

G = 9.80665
TIME_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


# ----------------------------------------------------------------------------- loading
def load_imu(path, gyro_units="rad", time_units="s"):
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=False)
        cols = {k: np.asarray(d[k], float) for k in d.files}
    else:
        with open(path) as f:
            header = f.readline().strip().lstrip("#").split(",")
        header = [h.strip().lower() for h in header]
        arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        cols = {h: arr[:, i] for i, h in enumerate(header)}
    need = ["t", "gx", "gy", "gz"]
    miss = [k for k in need if k not in cols]
    if miss:
        raise ValueError(f"{path}: missing column(s) {miss} (have {sorted(cols)})")
    t = cols["t"] * TIME_SCALE[time_units]
    g = np.stack([cols["gx"], cols["gy"], cols["gz"]], axis=1)
    if gyro_units == "dps":
        g = np.deg2rad(g)
    a = None
    if all(k in cols for k in ("ax", "ay", "az")):
        a = np.stack([cols["ax"], cols["ay"], cols["az"]], axis=1)
    ok = np.isfinite(t) & np.isfinite(g).all(axis=1)
    t, g = t[ok], g[ok]
    a = a[ok] if a is not None else None
    order = np.argsort(t, kind="stable")
    return t[order], g[order], (a[order] if a is not None else None)


# ----------------------------------------------------------------------------- Allan
def allan_overlapping(x, dt, points_per_octave=4):
    """overlapping Allan deviation of a rate signal x sampled every dt seconds.
    Returns tau [s], sigma_A, and the relative error of every sigma_A (approx)."""
    n = len(x)
    theta = np.concatenate([[0.0], np.cumsum(x) * dt])          # integrated angle
    m_max = n // 9
    if m_max < 2:
        raise ValueError("recording too short for an Allan deviation (need >= ~10 samples per tau)")
    ms = np.unique(np.floor(2 ** (np.arange(0, np.log2(m_max) * points_per_octave + 1) / points_per_octave)).astype(int))
    ms = ms[(ms >= 1) & (ms <= m_max)]
    taus, sig, rel = [], [], []
    for m in ms:
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        var = float((d ** 2).sum() / (2.0 * (m * dt) ** 2 * len(d)))
        taus.append(m * dt); sig.append(np.sqrt(var))
        rel.append(1.0 / np.sqrt(2.0 * max(n / m - 1, 1)))      # rough relative error of sigma
    return np.array(taus), np.array(sig), np.array(rel)


def fit_noise_terms(tau, sig, rel):
    """fit sigma^2 = N^2/tau + c*B^2 + K^2 tau/3 with all terms >= 0 (try every subset).
    Weights = 1/(relative error of the variance), so short and long taus count fairly."""
    var = sig ** 2
    w = 1.0 / (var * 2.0 * rel)
    basis = np.stack([1.0 / tau, np.ones_like(tau), tau / 3.0], axis=1)    # columns: N^2, c B^2, K^2
    best = None
    for k in (1, 2, 3):
        for cols in itertools.combinations(range(3), k):
            A = basis[:, cols] * w[:, None]
            coef, *_ = np.linalg.lstsq(A, var * w, rcond=None)
            if np.all(coef >= 0):
                chi = float(((A @ coef - var * w) ** 2).sum())
                if best is None or chi < best[0] - 1e-12:
                    best = (chi, cols, coef)
    full = np.zeros(3)
    if best is not None:
        for c, v in zip(best[1], best[2]):
            full[c] = v
    N = float(np.sqrt(full[0]))
    B = float(np.sqrt(full[1] / (2 * np.log(2) / np.pi)))
    K = float(np.sqrt(full[2]))
    return N, B, K


# ----------------------------------------------------------------------------- analysis
def timing_report(t):
    dt = np.diff(t)
    med = float(np.median(dt))
    rate = 1.0 / med
    gaps = int((dt > 2.5 * med).sum())
    lost = float(max(dt[dt > 2.5 * med].sum() - gaps * med, 0.0))
    return dict(rate_hz=rate, dt_med=med, dt_std=float(dt.std()), dt_max=float(dt.max()),
                gaps=gaps, lost_s=lost, duration=float(t[-1] - t[0]), n=len(t),
                mean_dt=float((t[-1] - t[0]) / (len(t) - 1)))


def motion_windows(t, g, win_s=1.0):
    """windows where the IMU was NOT still: window mean far from the typical window mean."""
    dtm = (t[-1] - t[0]) / (len(t) - 1)
    k = max(int(round(win_s / dtm)), 1)
    n = len(g) // k
    if n < 10:
        return []
    means = g[: n * k].reshape(n, k, 3).mean(axis=1)
    stds = g[: n * k].reshape(n, k, 3).std(axis=1)
    bad = []
    for ax in range(3):
        center = np.median(means[:, ax]); mad = 1.4826 * np.median(np.abs(means[:, ax] - center))
        s_med = np.median(stds[:, ax])
        thr_m = max(8 * mad, np.deg2rad(0.3))
        thr_s = max(5 * s_med, np.deg2rad(0.3))
        bad.extend(np.where((np.abs(means[:, ax] - center) > thr_m) | (stds[:, ax] > thr_s))[0].tolist())
    bad = sorted(set(bad))
    return [float(t[0] + b * k * dtm) for b in bad]


def analyse_gyro(t, g, a=None):
    warn = []
    tim = timing_report(t)
    if tim["duration"] < 600:
        warn.append(f"recording is only {tim['duration']/60:.1f} min; the Allan floor/drift needs >= 30 min (60 is better)")
    if tim["gaps"] > 0:
        pct = 100 * tim["lost_s"] / tim["duration"]
        warn.append(f"{tim['gaps']} gap(s) in the timestamps ({tim['lost_s']:.2f} s = {pct:.2f} % of the recording lost) - dropped IMU messages? "
                    f"The Allan time axis is off by about that percentage" + (" (too much - fix the logger and re-record)" if pct > 1 else ""))
    if tim["dt_std"] > 0.3 * tim["dt_med"]:
        warn.append(f"sample times jitter a lot (std {1e3*tim['dt_std']:.1f} ms vs {1e3*tim['dt_med']:.1f} ms) - stamp each sample on the microcontroller, not on arrival")
    mv = motion_windows(t, g)
    if mv:
        warn.append(f"the IMU does not look still in {len(mv)} one-second windows (first at t = {mv[0]-t[0]:.0f} s) - touched / vibrating?")
    dt = tim["mean_dt"]
    axes = []
    for i, name in enumerate("xyz"):
        x = g[:, i]
        bias = float(x.mean())
        sample_std = float(x.std())
        tau, sig, rel = allan_overlapping(x, dt)
        N, B, K = fit_noise_terms(tau, sig, rel)
        imin = int(np.argmin(sig))
        # first / last 5 min bias (warm-up drift)
        seg = int(300 / dt)
        warm = None
        if len(x) > 4 * seg:
            warm = float(x[-seg:].mean() - x[:seg].mean())
        axes.append(dict(axis=name, bias=bias, sample_std=sample_std, N=N, B=B, K=K,
                         tau_min=float(tau[imin]), sigma_min=float(sig[imin]),
                         warm_drift=warm, tau=tau.tolist(), sigma=sig.tolist(),
                         white_per_sample=N / np.sqrt(dt)))
    acc = None
    if a is not None:
        mag = np.linalg.norm(a, axis=1)
        acc = dict(mean_mag=float(mag.mean()), scale_err_pct=float(100 * (mag.mean() / G - 1)),
                   noise_std=[float(v) for v in a.std(axis=0)])
        if abs(acc["scale_err_pct"]) > 5:
            warn.append(f"|accel| = {acc['mean_mag']:.2f} m/s^2 differs {acc['scale_err_pct']:+.1f} % from gravity - check units / range setting")
    return dict(timing=tim, axes=axes, accel=acc, motion_windows=len(mv), warnings=warn)


# ----------------------------------------------------------------------------- report
DPS = 180.0 / np.pi


def make_report(name, r):
    t = r["timing"]
    L = [f"IMU STATIC CHECK  {name}", "-" * 72]
    L.append(f"{t['n']} samples, {t['duration']/60:.1f} min, {t['rate_hz']:.1f} Hz (dt jitter std {1e3*t['dt_std']:.2f} ms, max {1e3*t['dt_max']:.1f} ms, {t['gaps']} gaps)")
    L.append("")
    L.append(f"{'axis':>4} {'bias [deg/s]':>13} {'noise/sample':>13} {'ARW N':>14} {'bias inst. B':>13} {'RRW K':>16} {'min at tau':>11}")
    L.append(f"{'':>4} {'':>13} {'[deg/s]':>13} {'[deg/sqrt(h)]':>14} {'[deg/h]':>13} {'[deg/s/sqrt(s)]':>16} {'[s]':>11}")
    for a in r["axes"]:
        L.append(f"{a['axis']:>4} {a['bias']*DPS:13.4f} {a['white_per_sample']*DPS:13.4f} "
                 f"{a['N']*DPS*60:14.3f} {a['B']*DPS*3600:13.1f} {a['K']*DPS:16.2e} {a['tau_min']:11.1f}")
    L.append("")
    az = next(a for a in r["axes"] if a["axis"] == "z")
    L.append("For the car's heading use the axis that points UP (normally z). Numbers for the EKF:")
    L.append(f"  gyro white noise per sample : {az['white_per_sample']:.5f} rad/s at {t['rate_hz']:.0f} Hz   (= {az['white_per_sample']*DPS:.4f} deg/s)")
    L.append(f"  bias random walk (RRW)      : {az['K']:.2e} rad/s/sqrt(s)   <- bias process noise")
    L.append(f"  bias at this power-up       : {az['bias']:.5f} rad/s  (= {az['bias']*DPS:.4f} deg/s)   <- what the start-up calibration should find")
    L.append(f"  heading wander from white noise alone: {np.rad2deg(az['N']*np.sqrt(10)):.3f} deg after 10 s, {np.rad2deg(az['N']*np.sqrt(60)):.3f} deg after 60 s")
    L.append(f"  heading error if the bias were NOT corrected: {np.rad2deg(az['bias']*60):.2f} deg after 60 s  -> why the EKF estimates it")
    if az["warm_drift"] is not None:
        L.append(f"  bias change between first and last 5 min: {az['warm_drift']*DPS:+.4f} deg/s  (warm-up / temperature)")
    if r["accel"]:
        ac = r["accel"]
        L.append(f"  accelerometer |a| = {ac['mean_mag']:.3f} m/s^2 ({ac['scale_err_pct']:+.2f} % from 9.807); noise std per axis "
                 f"{', '.join(f'{v:.4f}' for v in ac['noise_std'])} m/s^2")
    if r["warnings"]:
        L.append(""); L.append("WARNINGS")
        L += ["  ! " + w for w in r["warnings"]]
    return "\n".join(L)


def turn_on_report(names, results):
    """scatter of the mean biases of several power-ups"""
    if len(results) < 2:
        return ""
    L = ["", "TURN-ON BIAS REPEATABILITY  (%d power-ups)" % len(results), "-" * 72]
    for i, ax in enumerate("xyz"):
        b = np.array([r["axes"][i]["bias"] for r in results])
        L.append(f"  {ax}: mean {b.mean()*DPS:+.4f} deg/s, std between power-ups {b.std(ddof=1)*DPS:.4f} deg/s "
                 f"(range {np.ptp(b)*DPS:.4f})")
    L.append("  -> the EKF's INITIAL bias uncertainty should be at least this std (use 2-3x for safety).")
    return "\n".join(L)


def make_figure(path, results, names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for nm, r in zip(names, results):
        for a in r["axes"]:
            ax.loglog(a["tau"], np.array(a["sigma"]) * DPS * 3600, label=f"{os.path.basename(nm)} {a['axis']}")
    ax.set_xlabel("tau [s]"); ax.set_ylabel("Allan deviation [deg/h]"); ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7); ax.set_title("gyro Allan deviation (slope -1/2 = white noise, +1/2 = bias random walk)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return True


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--gyro-units", choices=["rad", "dps"], default="rad")
    ap.add_argument("--time-units", choices=list(TIME_SCALE), default="s")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    results, reports = [], []
    for f in args.files:
        try:
            t, g, a = load_imu(f, args.gyro_units, args.time_units)
            if len(t) < 1000:
                raise ValueError(f"only {len(t)} samples")
            r = analyse_gyro(t, g, a)
        except Exception as ex:
            print(f"REFUSED {f}: {ex}"); return 1
        results.append(r); reports.append(make_report(f, r))
    text = "\n\n".join(reports) + turn_on_report(args.files, results)
    print(text)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "report.txt"), "w") as fh:
            fh.write(text + "\n")
        with open(os.path.join(args.out, "result.json"), "w") as fh:
            json.dump(dict(files=args.files, results=results), fh, indent=1)
        make_figure(os.path.join(args.out, "allan.png"), results, args.files)
    return 3 if any(r["warnings"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
