"""latency_analysis -- the maths of the camera latency test (latency_probe), without ROS so it can be tested.

The camera films a window that switches BLACK <-> WHITE at known moments (flash_window.py). Every picture is reduced
to a grid of cell brightnesses (GRID). For every switch at time t (in the clock of the stamps):
  pre   = pictures STAMPED in (t - 0.25 s, t]: certainly taken before the switch (a picture is never stamped before it
          was taken);
  post  = pictures stamped in (t, t + max_delay];
  window cells = the cells that move the right way (brighter after a switch to white), judged over ALL switches
          (median), so a cell that changed once for another reason (a hand, a shadow) does not count;
  first picture showing the switch = the first post picture whose window brightness has made half of its move.
  delay = its stamp - t.
The switches fall at random moments between two pictures, so the delays spread over one frame interval T:
  STAMP OFFSET = median(delay) - T/2 = how much later the stamp is than the middle of the picture's exposure,
  PLUS the screen's own delay (drawing + refresh, ~10-30 ms; one screen cannot separate the two).
"""
import math

import numpy as np

GRID = (16, 12)          # cells across, down


def cell_means(gray, grid=GRID):
    """A grey picture (any size) -> the mean brightness of grid[0] x grid[1] cells, flattened (float32)."""
    import cv2
    return cv2.resize(gray, grid, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def _windows(stamps, t, max_delay, pre_s):
    pre = (stamps > t - pre_s) & (stamps <= t)
    post = np.nonzero((stamps > t) & (stamps <= t + max_delay))[0]
    return pre, post


def cell_moves(stamps, cells, t, rising, max_delay=0.8, pre_s=0.25):
    """How far every cell moved the RIGHT way after the switch at t (>= 0 = the right way): the max (switch to
    white) / min (to black) over the post pictures minus the median of the pre pictures. None: too few pictures."""
    pre, post = _windows(stamps, t, max_delay, pre_s)
    if pre.sum() < 2 or len(post) < 2:
        return None
    s = 1.0 if rising else -1.0
    return (s * cells[post]).max(axis=0) - s * np.median(cells[pre], axis=0)


def first_picture(stamps, cells, t, rising, sel, max_delay=0.8, pre_s=0.25, min_jump=12.0):
    """(index of the first picture that shows the switch at t, the move in grey levels) using the window cells
    sel (bool mask); None if the window did not move by min_jump."""
    pre, post = _windows(stamps, t, max_delay, pre_s)
    if pre.sum() < 2 or len(post) < 2 or sel is None or not np.any(sel):
        return None
    s = 1.0 if rising else -1.0
    b = s * (cells[post][:, sel].mean(axis=1) - np.median(cells[pre][:, sel].mean(axis=1)))
    amp = float(b.max())
    if amp < min_jump:
        return None
    return int(post[np.nonzero(b >= 0.5 * amp)[0][0]]), amp


def seen_alone(stamps, cells, t, rising, max_delay=0.8, min_jump=12.0):
    """Quick check for ONE switch (while aiming): did some cells move the right way, and when? -> index or None."""
    m = cell_moves(stamps, cells, t, rising, max_delay)
    if m is None or m.max() < min_jump:
        return None
    r = first_picture(stamps, cells, t, rising, m >= max(min_jump, 0.5 * float(m.max())), max_delay,
                      min_jump=min_jump)
    return None if r is None else r[0]


def window_cells(stamps, cells, flips, max_delay=0.8, min_jump=12.0):
    """The cells that show the blinking window: per switch the move of every cell (cell_moves), the MEDIAN over
    the switches; cells with >= half of the largest median move and >= min_jump. -> (mask or None, median moves)."""
    moves = [m for m in (cell_moves(stamps, cells, t, lev == 1, max_delay) for t, lev in flips) if m is not None]
    if len(moves) < 3:
        return None, None
    med = np.median(np.array(moves), axis=0)
    if med.max() < min_jump:
        return None, med
    return med >= max(min_jump, 0.5 * float(med.max())), med


def clock_map(t_src, t_arr, chunk=8):
    """The flasher's clock -> ours. arrival - source time = offset + drift * (source - source[0]) + transport delay
    (>= 0, usually < 1 ms). The lower envelope (the smallest difference in every chunk of switches) fitted with a
    straight line, shifted so that no switch arrives before it happened. -> (offset, drift, t0); our time of a
    switch = t_src + offset + drift * (t_src - t0)."""
    t_src = np.asarray(t_src, float)
    d = np.asarray(t_arr, float) - t_src
    t0 = float(t_src[0]) if len(t_src) else 0.0
    if len(d) < 2 * chunk:
        return (float(d.min()) if len(d) else 0.0), 0.0, t0
    xs, ys = [], []
    for i in range(0, len(d) - chunk + 1, chunk):
        k = i + int(np.argmin(d[i:i + chunk]))
        xs.append(t_src[k] - t0)
        ys.append(d[k])
    drift, off = np.polyfit(xs, ys, 1)
    resid = d - (off + drift * (t_src - t0))
    return float(off + min(0.0, float(resid.min()))), float(drift), t0


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else math.nan


def analyze(stamps, recv, cells, dup, flips, dets=None, max_delay=0.8, min_jump=12.0, grid=GRID):
    """stamps, recv: (N,) seconds; cells: (N, C); dup: (N,) bool (= the same picture as the message before);
    flips: [(t, level)] (level 1 = white) in the stamps' clock; dets: (M, >= 2) [stamp, receive time,
    (latency_ms)] of the detection messages, or None. -> dict (seconds), incl. 'rows' per switch:
    [t, level, delay stamp, delay receive, move, picture index] (NaN = not seen)."""
    stamps = np.asarray(stamps, float)
    recv = np.asarray(recv, float)
    cells = np.asarray(cells, float).reshape(len(stamps), -1)
    dup = np.asarray(dup, bool)
    o = np.argsort(stamps, kind='stable')
    stamps, recv, cells, dup = stamps[o], recv[o], cells[o], dup[o]
    nan = math.nan
    res = dict(n_pictures=len(stamps), n_flips=len(flips), n_duplicates=int(dup.sum()), rate_hz=nan,
               frame_interval=nan, transport_median=nan, transport_p95=nan, seen=0, window_cells=0,
               n_cells=cells.shape[1] if cells.ndim == 2 else 0, offset=nan, delay_median=nan, delay_min=nan,
               delay_p5=nan, delay_p95=nan, recv_delay_median=nan, rows_delay=(nan, nan), boxes_delay_median=nan,
               boxes_delay_p95=nan, n_detection_msgs=0, yolo_latency_ms_median=nan, sel=None, cell_delay=None,
               order=o)
    if len(stamps) >= 2 and stamps[-1] > stamps[0]:
        res['rate_hz'] = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        u = stamps[~dup]
        if len(u) >= 2:
            res['frame_interval'] = (u[-1] - u[0]) / (len(u) - 1)
        tr = recv - stamps
        res['transport_median'], res['transport_p95'] = float(np.median(tr)), _pct(tr, 95)
    sel, _ = window_cells(stamps, cells, flips, max_delay, min_jump) if len(stamps) else (None, None)
    rows = []
    for t, lev in flips:
        r = first_picture(stamps, cells, t, lev == 1, sel, max_delay, min_jump=min_jump) if sel is not None else None
        if r is None:
            rows.append((t, lev, nan, nan, nan, nan))
        else:
            i, amp = r
            rows.append((t, lev, stamps[i] - t, recv[i] - t, amp, i))
    rows = np.array(rows, float).reshape(-1, 6)
    seen = ~np.isnan(rows[:, 2])
    # far-off delays: something else changed the window's cells (a hand in front, the phone moved) -> not counted.
    # A real delay lies within one frame interval of the median (two when the stamps come from a timer).
    T = res['frame_interval'] if np.isfinite(res['frame_interval']) else 0.04
    if seen.sum() >= 3:
        far = seen & (np.abs(rows[:, 2] - np.median(rows[seen, 2])) > 2.5 * T)
        rows[far, 2:] = np.nan
        res['far_off'] = int(far.sum())
        seen = ~np.isnan(rows[:, 2])
    else:
        res['far_off'] = 0
    res.update(rows=rows, sel=sel, seen=int(seen.sum()), window_cells=int(sel.sum()) if sel is not None else 0)
    if seen.sum() >= 3:
        d = rows[seen, 2]
        res.update(delay_median=float(np.median(d)), delay_min=float(d.min()), delay_p5=_pct(d, 5),
                   delay_p95=_pct(d, 95), recv_delay_median=float(np.median(rows[seen, 3])))
        res['offset'] = res['delay_median'] - res['frame_interval'] / 2
        res['cell_delay'] = _cell_delays(stamps, cells, rows[seen], sel, res['delay_median'],
                                         res['frame_interval'], max_delay)
        gx = grid[0]
        cd = res['cell_delay']
        if cd is not None and len(cd) == gx * grid[1]:
            by_row = [np.nanmean(cd[r * gx:(r + 1) * gx]) for r in range(grid[1])
                      if np.any(~np.isnan(cd[r * gx:(r + 1) * gx]))]
            if len(by_row) >= 2:
                res['rows_delay'] = (float(by_row[0]), float(by_row[-1]))
    if dets is not None and len(dets):
        dd = np.asarray(dets, float).reshape(len(dets), -1)
        x = dd[:, 1] - dd[:, 0]
        res.update(boxes_delay_median=float(np.median(x)), boxes_delay_p95=_pct(x, 95), n_detection_msgs=len(dd))
        if dd.shape[1] >= 3:
            res['yolo_latency_ms_median'] = float(np.median(dd[:, 2]))
    return res


def _cell_delays(stamps, cells, rows, sel, med, T, max_delay, pre_s=0.25):
    """Mean delay of every window cell on its own (its own half-way crossing); far-off values (> 2 frames from the
    overall median) dropped. Shows WHERE in the picture the switch arrives first (rolling shutter: top rows first)."""
    out = np.full(cells.shape[1], np.nan)
    if sel is None:
        return out
    per = {c: [] for c in np.nonzero(sel)[0]}
    for t, lev in rows[:, :2]:
        pre, post = _windows(stamps, t, max_delay, pre_s)
        if pre.sum() < 2 or len(post) < 2:
            continue
        s = 1.0 if lev == 1 else -1.0
        base = np.median(cells[pre], axis=0)
        for c in per:
            b = s * (cells[post, c] - base[c])
            if b.max() <= 0:
                continue
            k = np.nonzero(b >= 0.5 * b.max())[0][0]
            d = stamps[post[k]] - t
            if abs(d - med) <= 2 * T:
                per[c].append(d)
    for c, v in per.items():
        if len(v) >= 3:
            out[c] = float(np.mean(v))
    return out


def report_lines(res, pipe=None):
    """The result in plain words (one string per line)."""
    def ms(x):
        return 'n/a' if x is None or not np.isfinite(x) else f'{x * 1000:.0f} ms'

    L = [f"pictures: {res['n_pictures']} ({res['rate_hz']:.1f} per s), {res['n_duplicates']} of them DUPLICATES "
         f"(the same picture sent again); one NEW picture every {res['frame_interval'] * 1000:.1f} ms",
         f"a picture arrives here {ms(res['transport_median'])} after its stamp (p95 {ms(res['transport_p95'])})"]
    if pipe is not None:
        L.append(f"switch reports: {pipe}")
    L.append(f"switches seen in the camera picture: {res['seen']} of {res['n_flips']} "
             f"(the window covers {res['window_cells']} of {res['n_cells']} picture cells)"
             + (f"; {res['far_off']} more were far off the others and not counted (a hand in front? the camera "
                f"moved?)" if res.get('far_off') else ''))
    if res['seen'] >= 3:
        L += [f"switch -> STAMP of the first picture showing it: median {ms(res['delay_median'])}, min "
              f"{ms(res['delay_min'])}, p5-p95 {ms(res['delay_p5'])} - {ms(res['delay_p95'])} "
              f"(spread of one frame interval = normal)",
              f"switch -> RECEIVED here: median {ms(res['recv_delay_median'])}",
              f"STAMP OFFSET = median - half a frame interval = {ms(res['offset'])}: the stamp is this much LATER "
              f"than the moment the picture shows (incl. the screen's own delay, ~10-30 ms)"]
        top, bottom = res['rows_delay']
        if np.isfinite(top) and np.isfinite(bottom):
            L.append(f"inside the picture: the window's top row shows the switch {ms(top)}, its bottom row "
                     f"{ms(bottom)} after it" + (" -> rolling shutter: the camera reads the rows one after another"
                                                 if abs(bottom - top) > 0.003 else " (no visible rolling shutter)"))
    if res['n_detection_msgs']:
        L.append(f"boxes (/camera/detections, {res['n_detection_msgs']} messages): arrive {ms(res['boxes_delay_median'])}"
                 f" after the picture's stamp (p95 {ms(res['boxes_delay_p95'])}); YOLO's own latency_ms median "
                 f"{res['yolo_latency_ms_median']:.0f} ms")
        if res['seen'] >= 3:
            L.append(f"TOTAL: a cone box reaches the fusion node about {ms(res['offset'] + res['boxes_delay_median'])}"
                     f" after the moment its picture shows")
    return L


def plot(res, stamps, cells, path, title='', grid=GRID):
    """latency.png: delays (histogram), the window brightness around a few switches, where the window is."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stamps = np.asarray(stamps, float)[res['order']]
    cells = np.asarray(cells, float).reshape(len(stamps), -1)[res['order']]
    rows, sel = res['rows'], res['sel']
    seen = ~np.isnan(rows[:, 2])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1.0, 1.4, 0.8]))
    d = rows[seen, 2] * 1000
    if len(d):
        ax[0].hist(d, bins=np.arange(np.floor(d.min()) - 5, np.ceil(d.max()) + 10, 5), color='tab:blue')
        ax[0].axvline(res['delay_median'] * 1000, color='k', label=f"median {res['delay_median'] * 1000:.0f} ms")
        ax[0].axvline(res['offset'] * 1000, color='tab:red', ls='--',
                      label=f"stamp offset {res['offset'] * 1000:.0f} ms (median - T/2)")
        ax[0].legend(fontsize=8)
    ax[0].set_xlabel('switch -> stamp of the first picture showing it [ms]')
    ax[0].set_ylabel('switches')
    ax[0].set_title(f"{res['seen']} of {res['n_flips']} switches seen", fontsize=10)
    if sel is not None and seen.any():
        b = cells[:, sel].mean(axis=1)
        k = np.nonzero(seen)[0][len(np.nonzero(seen)[0]) // 2]
        t0 = rows[k, 0] - 0.6
        m = (stamps >= t0) & (stamps <= t0 + 4.0)
        ax[1].plot(stamps[m] - t0, b[m], '.-', color='0.4', ms=4, lw=0.8, label='pictures (at their stamp)')
        first = True
        for t, lev, dl, _, _, i in rows:
            if t0 <= t <= t0 + 4.0:
                ax[1].axvline(t - t0, color='tab:orange' if lev == 1 else 'k', ls='--', lw=1,
                              label='switch' if first else None)
                first = False
                if np.isfinite(i):
                    ax[1].plot(stamps[int(i)] - t0, b[int(i)], 'o', color='tab:red', ms=7, mfc='none',
                               label='first picture showing it' if t == rows[k, 0] else None)
        ax[1].legend(fontsize=8, loc='upper right')
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel('window brightness (grey level)')
    ax[1].set_title('switches (dashed) and the pictures that show them', fontsize=10)
    cd = res['cell_delay']
    if cd is not None:
        img = (cd * 1000).reshape(grid[1], grid[0])
        im = ax[2].imshow(np.ma.masked_invalid(img), cmap='viridis')
        fig.colorbar(im, ax=ax[2], fraction=0.035, label='mean delay of the cell [ms]')
    ax[2].set_title('where the window is in the picture\n(colour = when each cell shows the switch)', fontsize=9)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_xlabel(f'the camera picture ({grid[0]} x {grid[1]} cells)', fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
