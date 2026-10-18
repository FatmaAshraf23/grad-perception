"""test_latency_probe.py -- the latency test's maths (latency_analysis) on SYNTHETIC camera pictures with a KNOWN delay.
Run: python3 test/test_latency_probe.py   (or pytest). Needs numpy only.

The synthetic camera films the blinking window like the real chain: a picture shows the light at the middle of its
exposure (a switch inside the exposure gives a mix), the screen shows a switch display_s late, the picture arrives
latency_s after it was taken, and it is stamped either on arrival or -- like camera_publisher camera_node today --
by a 30 Hz timer when it is PUBLISHED (late by up to one timer period, sometimes the same picture twice).
"""
import math
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from sensor_fusion import latency_analysis as la      # noqa: E402

GX, GY = la.GRID


def synth(latency=0.110, display=0.020, fps=29.97, timer_hz=30.0, stamp='timer', n_flips=60, window=(3, 10, 2, 9),
          ae=False, noise=1.5, exposure=0.010, hidden=(), seed=0):
    """-> stamps, recv, cells, dup, flips [(t, level)], expected stamp offset."""
    rng = np.random.default_rng(seed)
    t, flips, level = 1.0, [], 0
    for _ in range(n_flips):
        t += rng.uniform(0.9, 1.5)
        level = 1 - level
        flips.append((t, level))
    t_end = t + 1.5
    ft = np.array([f[0] for f in flips])
    fl = np.array([f[1] for f in flips])

    def light(tt):                      # the window's luminance at time tt (the screen is display late)
        k = np.searchsorted(ft + display, tt, side='right') - 1
        lev = np.where(k >= 0, fl[np.clip(k, 0, None)], 0)
        return np.where(lev == 1, 230.0, 20.0)

    c = np.arange(0.2, t_end, 1.0 / fps) + rng.uniform(0, 1.0 / fps)          # middles of the exposures
    win = np.mean([light(c + s) for s in np.linspace(-exposure / 2, exposure / 2, 9)], axis=0)
    for k in hidden:                    # something in front of the window around switch k
        win[(c > ft[k] - 0.3) & (c < ft[k] + 0.6)] = 100.0
    mask = np.zeros((GY, GX), bool)
    x0, x1, y0, y1 = window
    mask[y0:y1, x0:x1] = True
    mask = mask.ravel()
    n_win = mask.sum()
    G, frames = 1.0, []
    for j in range(len(c)):
        if ae:                          # auto exposure: the gain drifts so that the picture's mean becomes 100
            target = 100.0 / ((win[j] * n_win + 100.0 * (mask.size - n_win)) / mask.size)
            G += (np.clip(target, 0.3, 3.0) - G) * (1 - math.exp(-(1.0 / fps) / 0.4))
        f = np.where(mask, win[j], 100.0) * G + rng.normal(0, noise, mask.size)
        frames.append(np.clip(f, 0, 255))
    arrive = c + latency
    stamps, recv, cells, dup = [], [], [], []
    if stamp == 'arrival':
        for j in range(len(c)):
            stamps.append(arrive[j])
            recv.append(arrive[j] + 0.002)
            cells.append(frames[j])
            dup.append(False)
        expected = display + latency
    else:
        tau = arrive[0] + rng.uniform(0, 1.0 / timer_hz)
        last = None
        while tau < arrive[-1]:
            j = int(np.searchsorted(arrive, tau, side='right')) - 1
            if j >= 0:
                stamps.append(tau)
                recv.append(tau + 0.002)
                cells.append(frames[j])
                dup.append(j == last)
                last = j
            tau += 1.0 / timer_hz
        expected = display + latency + 0.5 / timer_hz
    return (np.array(stamps), np.array(recv), np.array(cells), np.array(dup), flips, expected)


def check_offset(res, expected, tol=0.007):
    assert abs(res['offset'] - expected) < tol, f"offset {res['offset'] * 1000:.1f} ms, expected {expected * 1000:.1f} ms"


def test_timer_stamped_chain():
    """camera_node today: stamped when the 30 Hz timer publishes -> offset = display + latency + half a timer period;
    the duplicates are counted."""
    s, r, c, d, f, exp = synth(stamp='timer')
    res = la.analyze(s, r, c, d, f)
    print(f'    timer-stamped: offset {res["offset"] * 1000:.1f} ms (expected {exp * 1000:.1f}), seen {res["seen"]} '
          f'of {res["n_flips"]}, duplicates {res["n_duplicates"]} of {res["n_pictures"]}, window cells '
          f'{res["window_cells"]}')
    check_offset(res, exp)
    assert res['seen'] >= 58
    assert res['n_duplicates'] == int(d.sum()) and res['n_duplicates'] > 0
    assert res['window_cells'] == 49


def test_arrival_stamped_chain():
    """Stamped when the picture arrives (the fix): offset = display + latency, no duplicates."""
    s, r, c, d, f, exp = synth(stamp='arrival', latency=0.080, seed=1)
    res = la.analyze(s, r, c, d, f)
    print(f'    arrival-stamped: offset {res["offset"] * 1000:.1f} ms (expected {exp * 1000:.1f}), seen {res["seen"]}')
    check_offset(res, exp)
    assert res['n_duplicates'] == 0 and res['seen'] >= 58


def test_auto_exposure_small_window():
    """The phone's auto exposure fights the blinking (the levels drift back) and the window covers only 12 cells."""
    s, r, c, d, f, exp = synth(ae=True, window=(10, 14, 6, 9), seed=2)
    res = la.analyze(s, r, c, d, f)
    print(f'    auto exposure, 12-cell window: offset {res["offset"] * 1000:.1f} ms (expected {exp * 1000:.1f}), '
          f'seen {res["seen"]}, window cells {res["window_cells"]}')
    check_offset(res, exp)
    assert res['seen'] >= 57 and res['window_cells'] == 12


def test_hidden_switches():
    """5 switches hidden (a hand in front until 0.6 s after the switch): the window 'changes' only when the hand goes
    away -> far-off delays, not counted; the rest is not disturbed."""
    s, r, c, d, f, exp = synth(hidden=(10, 20, 30, 40, 50), seed=3)
    res = la.analyze(s, r, c, d, f)
    rows = res['rows']
    print(f'    5 hidden: seen {res["seen"]} of {res["n_flips"]}, far off {res["far_off"]}, offset '
          f'{res["offset"] * 1000:.1f} ms')
    assert all(np.isnan(rows[k, 2]) for k in (10, 20, 30, 40, 50))
    assert res['seen'] >= 53 and res['far_off'] >= 4
    assert any('far off' in x for x in la.report_lines(res))
    check_offset(res, exp)


def test_window_not_in_view():
    """The camera does not see the window: nothing seen, no crash, the report says so."""
    s, r, c, d, f, _ = synth(window=(0, 0, 0, 0), seed=4)
    res = la.analyze(s, r, c, d, f)
    lines = la.report_lines(res)
    assert res['seen'] == 0 and math.isnan(res['offset'])
    assert any('0 of 60' in x for x in lines)


def test_clock_map():
    """The flasher's clock: other offset + 40 ppm drift; reports arrive 0.1-3 ms late -> switch times within 0.3 ms."""
    rng = np.random.default_rng(5)
    t_ours = np.cumsum(rng.uniform(0.9, 1.5, 120)) + 1000.0
    t_src = (t_ours - 1000.0) * (1 - 40e-6) + 3.7                      # the flasher's clock
    arr = t_ours + rng.uniform(0.0001, 0.003, len(t_ours))
    off, drift, t0 = la.clock_map(t_src, arr)
    back = t_src + off + drift * (t_src - t0)
    err = np.abs(back - t_ours).max()
    print(f'    clock map: drift {drift * 1e6:.1f} ppm (true 40), worst switch time error {err * 1000:.2f} ms')
    assert err < 0.0003


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    n_fail = 0
    for t in tests:
        try:
            t()
            print(f'PASS {t.__name__}')
        except Exception:
            n_fail += 1
            print(f'FAIL {t.__name__}')
            traceback.print_exc()
    print(f'RESULT {len(tests) - n_fail} passed, {n_fail} failed, 0 skipped')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
