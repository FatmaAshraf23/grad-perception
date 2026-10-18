#!/usr/bin/env python3
"""LiDAR bench check: range OFFSET, scale error and noise from scans of a flat wall.

WHY
  Every LiDAR reads a little too long or too short by a nearly constant amount (its
  "range offset").  Our obstacle node subtracts one number (`range_offset:=` in
  obstacles.launch.py) from every real return, so we must MEASURE it on the real
  RPLIDAR.  The simulator's value is 0.022 m, which is only an artefact of its ray marching.

HOW IT WORKS (plain words)
  1. You put the LiDAR a known distance D from a flat wall (tape measure, from the
     rotation centre of the LiDAR, along the wall's perpendicular) and record ~60 s.
  2. We turn every beam into a point (x, y) = (r cos a, r sin a) and find the wall as a
     straight line through those points.  A RANSAC fit ignores things that are not the
     wall (a chair, your feet).  The line tells us which way the wall faces (angle phi).
  3. A perfect LiDAR looking at a wall at perpendicular distance D, along a beam that
     makes an angle (a - phi) with the wall normal, would read D / cos(a - phi).
     offset_i = r_i - D / cos(a_i - phi).   The mean of offset_i over beams and scans is
     the range offset.  (Because phi comes from the fit, you do NOT have to point the
     LiDAR exactly at the wall.)
  4. With several distances we also fit  offset(D) = a + e * D :  `a` is the constant
     offset, `e` a scale error (e.g. 0.01 = reads 1 % too long).  Only a constant can
     be given to the obstacle node, so we also print the offset at 2, 4 and 6 m.
  5. Noise: for every beam the standard deviation across scans (repeatability), and the
     spread of all beams together (adds wall roughness and beam-to-beam differences).

INPUT  .npz files made by bench_record.py (keys: ranges[n_scans, n_beams], angle_min,
       angle_increment, range_min, range_max, optional stamps).
USAGE
  python3 lidar_range_check.py --run 1.0:scan_1m.npz --run 2.0:scan_2m.npz --out result/
  (D in metres, measured from the rotation centre of the LiDAR to the wall.)
EXIT   0 = fine, 3 = result printed but with warnings, 1 = refused (bad data).
numpy only; matplotlib is optional (for the picture).
"""
import argparse
import json
import os
import sys

import numpy as np


# ----------------------------------------------------------------------------- loading
class Scans:
    """ranges [n_scans, n_beams] (NaN = no return / padding) + angle of every beam."""

    def __init__(self, ranges, angles, range_min, range_max, stamps=None):
        self.ranges = np.asarray(ranges, float)
        self.angles = np.asarray(angles, float)            # same shape as ranges
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.stamps = None if stamps is None else np.asarray(stamps, float)

    @property
    def n_scans(self):
        return self.ranges.shape[0]


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    rng = np.asarray(d["ranges"], float)
    if rng.ndim != 2:
        raise ValueError(f"{path}: 'ranges' must be 2-D [scans, beams]")
    a0 = np.asarray(d["angle_min"], float)
    inc = np.asarray(d["angle_increment"], float)
    a0 = np.broadcast_to(a0, (rng.shape[0],)) if a0.ndim == 0 else a0
    inc = np.broadcast_to(inc, (rng.shape[0],)) if inc.ndim == 0 else inc
    idx = np.arange(rng.shape[1])[None, :]
    ang = a0[:, None] + inc[:, None] * idx
    return Scans(rng, ang, float(d["range_min"]), float(d["range_max"]),
                 d["stamps"] if "stamps" in d.files else None)


# ----------------------------------------------------------------------------- analysis
def wrap(a):
    """angle wrapped to (-pi, pi]"""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def fit_line_tls(xy):
    """total-least-squares line through points; returns (normal angle phi, distance d >= 0)."""
    c = xy.mean(axis=0)
    u, s, vt = np.linalg.svd(xy - c, full_matrices=False)
    n = vt[1]                      # smallest direction = the normal
    d = float(n @ c)
    if d < 0:
        n, d = -n, -d
    return float(np.arctan2(n[1], n[0])), d


def ransac_wall(xy, true_d, rng, iters=400, thr=0.05):
    """find the dominant straight wall; returns inlier mask (or None)."""
    n = len(xy)
    if n < 50:
        return None
    best, best_cnt = None, 0
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        p, q = xy[i], xy[j]
        t = q - p
        L = np.hypot(*t)
        if L < 0.2 * true_d:           # points too close together -> unstable line
            continue
        nrm = np.array([-t[1], t[0]]) / L
        dist = np.abs((xy - p) @ nrm)
        cnt = int((dist < thr).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, dist < thr
    return best


def analyse_wall(scans, true_d, half_window_deg=40.0, seed=0):
    """Range offset / noise from scans of ONE flat wall at perpendicular distance true_d."""
    warn = []
    rng = np.random.default_rng(seed)
    R, A = scans.ranges, scans.angles
    valid = np.isfinite(R) & (R >= scans.range_min) & (R <= scans.range_max) & (R > 0)
    # a wall at perpendicular distance D is at least D away and (inside +-60 deg) < 2 D
    cand = valid & (R >= 0.8 * true_d) & (R <= 2.0 * true_d)
    if scans.n_scans < 20:
        raise ValueError(f"only {scans.n_scans} scans - record at least ~20 (60 s is better)")
    Rv = np.where(valid, R, np.nan)                          # no-return beams (inf / 0) -> NaN, no warnings
    xy_all = np.stack([Rv * np.cos(A), Rv * np.sin(A)], axis=-1)
    pts = xy_all[cand]
    if len(pts) > 20000:
        pts = pts[rng.choice(len(pts), 20000, replace=False)]
    inl = ransac_wall(pts, true_d, rng)
    if inl is None or inl.sum() < 200:
        raise ValueError("no flat wall found near the given distance - check D, the wall and the setup")
    # refine: total least squares on inliers, then re-select with 3 sigma, 3 rounds
    line_pts = pts[inl]
    phi, d_fit = fit_line_tls(line_pts)
    for _ in range(3):
        nrm = np.array([np.cos(phi), np.sin(phi)])
        res = pts @ nrm - d_fit
        sig = max(1.4826 * np.median(np.abs(res[np.abs(res) < 0.1])), 0.003)
        sel = np.abs(res) < 3 * sig
        phi, d_fit = fit_line_tls(pts[sel])
    # classify every beam of every scan with the final line
    nrm = np.array([np.cos(phi), np.sin(phi)])
    off_ang = wrap(A - phi)                                  # angle to the wall normal
    window = np.abs(off_ang) <= np.deg2rad(half_window_deg)
    perp = xy_all @ nrm - d_fit
    near = valid & window & (np.abs(perp) < 3 * max(sig, 0.01) + 0.01)
    # expected range of a perfect LiDAR
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = true_d / np.cos(off_ang)
        offs = np.where(near, R - expected, np.nan)
    n_used = int(near.sum())
    if n_used < 500:
        raise ValueError(f"only {n_used} wall readings in +-{half_window_deg:.0f} deg - wall too small / too far?")
    offset = float(np.nanmean(offs))
    # honest uncertainty = two parts that do not average away the same way:
    #  (a) scan-to-scan noise: spread of the per-scan means / sqrt(scans)
    #  (b) beam-to-beam differences (each beam has its own small, repeatable error, the same in every
    #      scan): spread of the per-beam means / sqrt(beams)
    scan_means = np.array([np.nanmean(o) for o in offs if np.isfinite(o).sum() >= 10])
    beam_means = np.array([np.nanmean(offs[:, k]) for k in range(offs.shape[1]) if np.isfinite(offs[:, k]).sum() >= 10])
    se_scan = float(np.std(scan_means, ddof=1) / np.sqrt(len(scan_means))) if len(scan_means) > 2 else float("nan")
    se_beam = float(np.std(beam_means, ddof=1) / np.sqrt(len(beam_means))) if len(beam_means) > 2 else float("nan")
    se = float(np.sqrt(se_scan ** 2 + se_beam ** 2))
    sigma_all = float(np.nanstd(offs))
    # repeatability: for every beam index, std across scans (MAD based), then median over beams
    per_beam = []
    for k in range(offs.shape[1]):
        col = offs[:, k]
        col = col[np.isfinite(col)]
        if len(col) >= 10:
            per_beam.append(1.4826 * np.median(np.abs(col - np.median(col))))
    sigma_rep = float(np.median(per_beam)) if per_beam else float("nan")
    # trend of the offset with the angle to the wall normal (reveals tilt / non-flat wall)
    m = np.isfinite(offs)
    slope = float(np.polyfit(off_ang[m], offs[m], 1)[0]) if m.sum() > 100 else float("nan")
    # dropouts inside the window
    in_win = window & np.isfinite(A)
    drop = float(1.0 - (valid & in_win).sum() / max(in_win.sum(), 1))
    # --- checks
    inlier_frac = n_used / max((valid & window).sum(), 1)
    if inlier_frac < 0.7:
        warn.append(f"only {100*inlier_frac:.0f} % of the readings in the window lie on the wall (objects in front of it?)")
    if sigma_all > 0.03:
        warn.append(f"spread of readings {100*sigma_all:.1f} cm is large - is the wall flat, matte and light-coloured?")
    if abs(true_d - d_fit) > 0.15:
        warn.append(f"the fitted wall distance ({d_fit:.3f} m) differs from D ({true_d:.3f} m) by more than 15 cm - wrong D typed?")
    if drop > 0.2:
        warn.append(f"{100*drop:.0f} % of the beams towards the wall return nothing (glass / black / shiny surface?)")
    if np.isfinite(slope) and abs(slope) > 0.02:
        warn.append(f"offset changes with angle (slope {slope:.3f} m/rad): wall not flat or D measured not perpendicular")
    res = dict(true_d=true_d, offset=offset, offset_se=se, sigma_all=sigma_all,
               sigma_repeat=sigma_rep, wall_normal_deg=float(np.rad2deg(phi)),
               d_fit=float(d_fit), line_noise=float(sig), n_scans=int(scans.n_scans),
               n_readings=n_used, dropout=drop, angle_slope=slope, warnings=warn)
    return res, (xy_all, near, phi, d_fit)


def fit_distance_model(results):
    """offset(D) = a + e*D over several distances.  Returns dict (needs >= 2 distances)."""
    D = np.array([r["true_d"] for r in results])
    O = np.array([r["offset"] for r in results])
    out = dict(mean_offset=float(O.mean()))
    if len(D) >= 3 and np.ptp(D) >= 1.0:
        e, a = np.polyfit(D, O, 1)
        resid = O - (a + e * D)
        spread_const = float(O.std(ddof=1))
        spread_line = float(np.sqrt((resid ** 2).sum() / (len(D) - 2)))
        out.update(a=float(a), e=float(e), spread_const=spread_const, spread_line=spread_line,
                   scale_matters=bool(abs(e) * np.ptp(D) > 2 * max(spread_line, 0.002)))
    return out


# ----------------------------------------------------------------------------- report
def make_report(results, model):
    L = []
    L.append("LIDAR RANGE CHECK  (positive offset = the LiDAR reads TOO LONG)")
    L.append("-" * 72)
    L.append(f"{'D [m]':>6} {'offset [cm]':>12} {'+-':>6} {'noise rep.':>11} {'noise all':>10} {'dropout':>8} {'wall dir':>9} {'scans':>6}")
    for r in results:
        L.append(f"{r['true_d']:6.2f} {100*r['offset']:12.2f} {100*r['offset_se']:6.2f} "
                 f"{100*r['sigma_repeat']:10.2f}c {100*r['sigma_all']:9.2f}c {100*r['dropout']:7.1f}% "
                 f"{r['wall_normal_deg']:8.1f}d {r['n_scans']:6d}")
    L.append("")
    if "a" in model:
        a, e = model["a"], model["e"]
        L.append(f"Fit over distance:  offset(D) = {100*a:+.2f} cm {100*e:+.3f} cm per m   (scale error {100*e:+.2f} %)")
        L.append(f"  spread around a constant {100*model['spread_const']:.2f} cm, around the line {100*model['spread_line']:.2f} cm")
        if model["scale_matters"]:
            L.append("  -> the offset CHANGES with distance.  The obstacle node can only take a constant;")
            L.append("     offset at 2 m: %+.1f cm, 4 m: %+.1f cm, 6 m: %+.1f cm.  Use the value near the distance"
                     % tuple(100 * (a + e * x) for x in (2, 4, 6)))
            L.append("     where cones matter most (2-4 m), and raise SIGMA_DET a little for the rest.")
        else:
            L.append("  -> no significant change with distance: a constant offset is a good model.")
    L.append("")
    L.append(f"RANGE OFFSET TO USE:   range_offset:={model['mean_offset']:.3f}   (metres; obstacles.launch.py)")
    noise = np.nanmedian([r["sigma_repeat"] for r in results])
    allv = np.nanmedian([r["sigma_all"] for r in results])
    L.append(f"Noise (1 sigma): repeatability per beam {100*noise:.1f} cm, all beams together {100*allv:.1f} cm")
    L.append("  -> compare with the simulator's 2.0 cm; the obstacle node's SIGMA_DET / SCAN_CORR were tuned to that.")
    allw = [w for r in results for w in r["warnings"]]
    if allw:
        L.append("")
        L.append("WARNINGS")
        for w in allw:
            L.append("  ! " + w)
    return "\n".join(L)


def make_figure(path, results, model, geo):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    xy_all, near, phi, d_fit = geo
    pts = xy_all[0]
    ax[0].scatter(pts[:, 0], pts[:, 1], s=3, c=np.where(near[0], "tab:green", "tab:red"))
    ax[0].plot(0, 0, "k+", ms=12)
    ax[0].set_aspect("equal")
    ax[0].set_title("first scan of the last run (green = on the wall)")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    D = np.array([r["true_d"] for r in results]); O = 100 * np.array([r["offset"] for r in results])
    S = 100 * np.array([r["offset_se"] for r in results])
    ax[1].errorbar(D, O, yerr=S, fmt="o")
    if "a" in model:
        xs = np.linspace(D.min(), D.max(), 20)
        ax[1].plot(xs, 100 * (model["a"] + model["e"] * xs), "--")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("true distance D [m]"); ax[1].set_ylabel("offset [cm]")
    ax[1].set_title("range offset vs distance")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return True


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="D:FILE.npz",
                    help="true wall distance in metres and the recording, e.g. 2.0:scan_2m.npz (repeat per distance)")
    ap.add_argument("--half-window-deg", type=float, default=40.0,
                    help="use beams within this angle of the wall normal (default 40)")
    ap.add_argument("--out", help="folder for report.txt, result.json, range_check.png")
    args = ap.parse_args(argv)
    results, geo = [], None
    for spec in args.run:
        try:
            d_txt, path = spec.split(":", 1)
            D = float(d_txt)
        except ValueError:
            print(f"bad --run '{spec}' (use D:FILE.npz)"); return 1
        try:
            r, geo = analyse_wall(load_npz(path), D, args.half_window_deg)
        except Exception as ex:
            print(f"REFUSED {path}: {ex}"); return 1
        r["file"] = path
        results.append(r)
    results.sort(key=lambda r: r["true_d"])
    model = fit_distance_model(results)
    rep = make_report(results, model)
    print(rep)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "report.txt"), "w") as f:
            f.write(rep + "\n")
        with open(os.path.join(args.out, "result.json"), "w") as f:
            json.dump(dict(runs=results, model=model), f, indent=1)
        make_figure(os.path.join(args.out, "range_check.png"), results, model, geo)
    return 3 if any(r["warnings"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
