"""flash_window -- a window that switches BLACK <-> WHITE at random moments, for the camera latency test
(sensor_fusion latency_probe starts it). Plain Python + OpenCV, no ROS: it runs on the car's own screen or, for the
development camera, on Windows (the iPhone films the laptop screen).

Right after every switch has been drawn it prints   FLIP <n> <level: 1 white, 0 black> <its clock, seconds>
(time.perf_counter: precise to microseconds; latency_probe converts it to the ROS clock). The window starts black.
It stops after --seconds, on Esc, when the window is closed, or (--stop-on-eof) when its input is closed:
latency_probe starts it that way and closes the input when it has enough.
The screen adds its own delay (drawing + refresh, ~10-30 ms); the probe's result includes it.

2026-10-16: the input is CHECKED (without waiting) instead of read by a waiting thread. On Windows a thread waiting
on the input pipe froze the loading of OpenCV's libraries -> the window never opened (job 144). On Windows the
window is also forced on top of all others and its position is printed (READY line).

  python3 flash_window.py [--seconds 300] [--min-gap 0.9] [--max-gap 1.5] [--size 900x650] [--pos 40,40]
                          [--stop-on-eof] [--virtual]   (no window, only the switch times: self-test)
"""
import argparse
import os
import random
import sys
import time

TITLE = 'latency test - keep the camera still'


def input_closed_check():
    """-> a function that says, WITHOUT waiting, whether our input has been closed (its other end gone)."""
    try:
        fd = sys.stdin.fileno()
    except (AttributeError, ValueError, OSError):
        return lambda: False
    if os.name == 'nt':
        import ctypes
        import msvcrt
        from ctypes import wintypes
        try:
            handle = msvcrt.get_osfhandle(fd)
        except OSError:
            return lambda: False
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        peek = k32.PeekNamedPipe
        peek.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                         ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        peek.restype = wintypes.BOOL
        avail = wintypes.DWORD()

        def closed():
            if peek(handle, None, 0, None, ctypes.byref(avail), None):
                if avail.value:                     # something was sent: read it away (we never wait)
                    os.read(fd, avail.value)
                return False
            return ctypes.get_last_error() == 109   # ERROR_BROKEN_PIPE: the other end closed it
        return closed                               # (not a pipe, e.g. a console: never 'closed')
    import select

    def closed():
        readable, _, _ = select.select([fd], [], [], 0)
        return bool(readable) and os.read(fd, 4096) == b''
    return closed


def windows_on_top(title, x, y):
    """Windows only: put the window above all others at (x, y) -> where it is (for the log)."""
    import ctypes
    from ctypes import wintypes
    u = ctypes.WinDLL('user32', use_last_error=True)
    u.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    u.FindWindowW.restype = wintypes.HWND
    u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND] + [ctypes.c_int] * 4 + [wintypes.UINT]
    u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u.IsWindowVisible.argtypes = [wintypes.HWND]
    hwnd = u.FindWindowW(None, title)
    if not hwnd:
        return 'window handle NOT found'
    u.SetWindowPos(hwnd, wintypes.HWND(-1), x, y, 0, 0, 0x0001 | 0x0040)     # HWND_TOPMOST, NOSIZE | SHOWWINDOW
    r = wintypes.RECT()
    u.GetWindowRect(hwnd, ctypes.byref(r))
    return (f'visible {bool(u.IsWindowVisible(hwnd))}, on screen at ({r.left},{r.top})-({r.right},{r.bottom}), '
            f'screen {u.GetSystemMetrics(0)}x{u.GetSystemMetrics(1)}')


def main(argv=None):
    ap = argparse.ArgumentParser(description='blinking window for the camera latency test')
    ap.add_argument('--seconds', type=float, default=300.0)
    ap.add_argument('--min-gap', type=float, default=0.9, help='shortest time between two switches [s]')
    ap.add_argument('--max-gap', type=float, default=1.5, help='longest time between two switches [s]')
    ap.add_argument('--size', default='900x650', help='window size WIDTHxHEIGHT [px]')
    ap.add_argument('--pos', default='40,40', help='window position X,Y on the screen [px]')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--stop-on-eof', action='store_true', help='stop when the input is closed (latency_probe)')
    ap.add_argument('--virtual', action='store_true', help='no window: only print the switch times')
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    w, h = (int(v) for v in a.size.lower().split('x'))
    x, y = (int(v) for v in a.pos.split(','))
    print(f'STARTING ({"virtual" if a.virtual else "window"}, Python {sys.version.split()[0]} on {sys.platform})',
          flush=True)

    cv2 = None
    where = 'virtual'
    if not a.virtual:
        import cv2
        import numpy as np
        pictures = []
        for level in (0, 1):
            img = np.full((h, w, 3), 255 * level, np.uint8)
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (128, 128, 128), 6)
            cv2.putText(img, 'LATENCY TEST - keep the camera still', (20, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (128, 128, 128), 2, cv2.LINE_AA)
            pictures.append(img)
        cv2.namedWindow(TITLE, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(TITLE, x, y)
        try:
            cv2.setWindowProperty(TITLE, cv2.WND_PROP_TOPMOST, 1)     # stay above the other windows
        except (AttributeError, cv2.error):
            pass
        where = f'window {w}x{h}, OpenCV {cv2.__version__}'

    def show(level):
        if cv2 is not None:
            cv2.imshow(TITLE, pictures[level])
            cv2.waitKey(1)
        return time.perf_counter()

    level = 0
    t = show(level)
    if cv2 is not None and os.name == 'nt':
        try:
            where += ', ' + windows_on_top(TITLE, x, y)
        except (OSError, AttributeError) as e:
            where += f', could not put it on top: {e}'
    closed = input_closed_check() if a.stop_on_eof else (lambda: False)
    print(f'READY {where}, clock {t:.6f}', flush=True)
    end = t + a.seconds
    n = 0
    next_t = t + rng.uniform(a.min_gap, a.max_gap)
    next_check = 0.0
    reason = 'time is up'
    while True:
        now = time.perf_counter()
        if now >= end:
            break
        if now >= next_check:
            next_check = now + 0.05
            if closed():
                reason = 'input closed'
                break
        if now >= next_t:
            level = 1 - level
            n += 1
            t = show(level)
            print(f'FLIP {n} {level} {t:.6f}', flush=True)
            next_t = t + rng.uniform(a.min_gap, a.max_gap)
        elif cv2 is None:
            time.sleep(min(0.002, max(0.0, next_t - now)))
        else:
            if cv2.waitKey(2) == 27:
                reason = 'Esc'
                break
            try:
                if cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    reason = 'window closed'
                    break
            except cv2.error:
                reason = 'window closed'
                break
    print(f'END {n} switches ({reason})', flush=True)
    if cv2 is not None:
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
