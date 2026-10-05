"""Reading boundaries.csv + the boundary fingerprint (runtime part, numpy only).
The tool that MAKES the file is boundaries.py (needs Pillow for the map image)."""
import hashlib

import numpy as np

COLUMNS = ['s', 'w_left', 'w_right', 'left_virtual', 'right_virtual']


def read_boundaries_csv(path):
    """-> (meta dict from the '#' lines, columns dict of numpy arrays)."""
    meta, rows = {}, []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                for tok in line[1:].split():
                    if '=' in tok:
                        k, v = tok.split('=', 1)
                        meta[k] = v
                continue
            if line.startswith('s,'):
                continue
            rows.append([float(x) for x in line.split(',')])
    a = np.array(rows)
    cols = {name: a[:, j] for j, name in enumerate(COLUMNS)}
    cols['left_virtual'] = cols['left_virtual'].astype(bool)
    cols['right_virtual'] = cols['right_virtual'].astype(bool)
    return meta, cols


def compute_boundary_hash(track_hash, w_left, w_right, left_virtual, right_virtual):
    """16-hex fingerprint of the boundaries (widths at 1 mm + flags + the track they belong to)."""
    h = hashlib.sha256()
    h.update(f'apex_track boundaries v1 track={track_hash}\n'.encode())
    for a, b, c, d in zip(np.round(np.asarray(w_left) * 1000).astype(np.int64),
                          np.round(np.asarray(w_right) * 1000).astype(np.int64),
                          left_virtual, right_virtual):
        h.update(f'{a},{b},{int(c)},{int(d)}\n'.encode())
    return h.hexdigest()[:16]
