#!/usr/bin/env python3
"""Tests for amcl_map.py (AMCL's thin-wall copy of the map), 2026-10-19.
Run: python3 test_amcl_map.py   (or pytest)"""
import os
import sys

import numpy as np

try:
    from state_estimation.amcl_map import surface, thin_walls
except ImportError:                                  # run from the source tree without ROS
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'state_estimation'))
    from amcl_map import surface, thin_walls


def corridor():
    """a 1-cell-free corridor between two 3-cell walls, unknown outside (OccupancyGrid values)"""
    g = np.full((11, 12), -1, dtype=np.int16)
    g[2:5, :] = 100            # wall 1: rows 2-4
    g[5, :] = 0                # free row 5
    g[6:9, :] = 100            # wall 2: rows 6-8
    return g


def test_only_the_wall_surface_stays_occupied():
    out = thin_walls(corridor())
    assert (out[4] == 100).all() and (out[6] == 100).all()          # the rows touching the free row
    assert (out[2:4] == -1).all() and (out[7:9] == -1).all()        # behind the surface: unknown
    assert (out[5] == 0).all()                                      # free stays free
    assert (out[0:2] == -1).all() and (out[9:] == -1).all()         # unknown stays unknown


def test_diagonal_neighbours_count():
    """a wall cell that touches free space only diagonally is a surface cell too (no gaps in corners)"""
    occ = np.ones((3, 3), bool)
    free = np.zeros((3, 3), bool)
    occ[0, 0], free[0, 0] = False, True
    s = surface(occ, free)
    assert s[1, 1] and s[0, 1] and s[1, 0] and not s[2, 2]


def test_values_other_than_0_and_100_are_unknown_like_amcl_reads_them():
    g = np.array([[0, 50, 100, -1],
                  [0, 0, 100, 100],
                  [-1, 99, -1, 100],
                  [-1, -1, -1, 100]], dtype=np.int16)
    out = thin_walls(g)
    assert out[0, 1] == -1 and out[2, 1] == -1                      # 50 / 99: unknown for AMCL anyway
    assert out[0, 2] == 100 and out[1, 2] == 100                    # touch free cells
    assert out[1, 3] == -1 and out[2, 3] == -1 and out[3, 3] == -1  # touch only unknown / walls
    assert out.dtype == np.int8


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS  {name}')
        except Exception as ex:                       # noqa: BLE001
            fails += 1
            print(f'FAIL  {name}: {type(ex).__name__}: {ex}')
    print(f'\n{len(tests) - fails} of {len(tests)} passed')
    sys.exit(1 if fails else 0)
