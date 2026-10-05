"""Track identity: ONE track definition file that every team member loads.

A track is described by a small YAML file (example: tracks/levine/track.yaml):

    track_id: levine           # human-readable name
    centerline: centerline.csv # x,y points of the closed centre line, metres,
                               #   map frame, first point = start/finish line
    smooth_m: 0.5              # Gaussian smoothing along the arc [m]
    ds: 0.05                   # resampling step of the smoothed curve [m]
    boundaries: boundaries.csv # OPTIONAL: wall distances left/right (boundaries.py)

load_track(path) reads it and returns a FrenetTrack that also carries
    .track_id    the name from the file
    .track_hash  a 16-hex-character fingerprint of the GEOMETRY
    .boundary_hash  fingerprint of the boundaries ('' if the track has none)

WHY THE BOUNDARIES HAVE THEIR OWN HASH (2026-10-05)
    track_hash = the Frenet geometry (centre line, smoothing, code). It is in
    every StateEstimate and control checks it. The walls do not change s,
    e_y, e_psi or kappa, so adding or re-measuring boundaries must NOT break
    that check. Planning, which uses the walls, compares boundary_hash.
    A boundaries file is tied to one track_hash (its header) -- if the
    centre line or smoothing changes, load_track() refuses the old file.

WHY A HASH (APEX contract, control's response 2026-10-02)
    s, e_y, e_psi and kappa only mean the same thing to perception and control
    if both use exactly the same centre line, smoothing and code. The name
    alone can't guarantee that (someone edits the CSV and keeps the name).
    The hash is computed from the things that change the geometry:
        - the centre-line POINTS (rounded to 1 mm)
        - smooth_m and ds
        - GEOMETRY_VERSION (bumped whenever frenet.py's math changes)
    Perception puts it in every StateEstimate; control compares it with its
    own and rejects the message on a mismatch.

WHY HASH THE NUMBERS, NOT THE FILE BYTES
    The same CSV saved on Windows (CRLF line ends), with a trailing newline,
    or written as 0.50 instead of 0.5 has different bytes but the same track.
    Hashing the parsed points rounded to 1 mm gives the same hash for the
    same geometry, whatever editor or OS touched the file.
"""
import hashlib
import os

import numpy as np
import yaml

from .boundary_io import compute_boundary_hash, read_boundaries_csv
from .frenet import FrenetTrack, load_xy_csv

# Bump this when the math in frenet.py changes in a way that moves s, e_y,
# e_psi or kappa (smoothing, resampling, curvature formula). Old hashes then
# stop matching on purpose.
GEOMETRY_VERSION = 2   # v2 (2026-10-02): projection refined onto the smooth tangent


def compute_track_hash(xy, smooth_m, ds, geometry_version=GEOMETRY_VERSION):
    """16-hex fingerprint of a track's geometry (see module docstring)."""
    pts_mm = np.round(np.asarray(xy, dtype=float) * 1000.0).astype(np.int64)
    h = hashlib.sha256()
    h.update(f'apex_track geometry v{geometry_version}\n'.encode())
    h.update(f'smooth_m={round(float(smooth_m), 4)} ds={round(float(ds), 4)}\n'.encode())
    for x, y in pts_mm:
        h.update(f'{x},{y}\n'.encode())
    return h.hexdigest()[:16]


def load_track(yaml_path, with_boundaries=True, **frenet_kw):
    """Read a track.yaml -> FrenetTrack with .track_id and .track_hash set
    (+ widths and .boundary_hash if the yaml names a boundaries file).

    Extra keyword arguments (window_m, max_e_y) go to FrenetTrack; they only
    change the SEARCH, not the geometry, so they are not part of the hash.
    with_boundaries=False skips the boundaries file (used when remaking it).
    """
    yaml_path = os.path.abspath(os.path.expanduser(yaml_path))
    with open(yaml_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    for key in ('track_id', 'centerline', 'smooth_m', 'ds'):
        if key not in cfg:
            raise ValueError(f'{yaml_path}: missing "{key}"')

    csv_path = os.path.join(os.path.dirname(yaml_path), cfg['centerline'])
    xy = load_xy_csv(csv_path)
    smooth_m, ds = float(cfg['smooth_m']), float(cfg['ds'])

    track = FrenetTrack(xy, smooth_m=smooth_m, ds=ds, **frenet_kw)
    track.track_id = str(cfg['track_id'])
    track.track_hash = compute_track_hash(xy, smooth_m, ds)
    track.source_file = yaml_path
    track.boundary_hash = ''
    if with_boundaries and cfg.get('boundaries'):
        _attach_boundaries(track, os.path.join(os.path.dirname(yaml_path), cfg['boundaries']))
    return track


def _attach_boundaries(track, path):
    meta, b = read_boundaries_csv(path)
    redo = ('-- remake it: python3 -m apex_track.boundaries <track.yaml> --map <map.yaml>')
    if meta.get('track_hash') != track.track_hash:
        raise ValueError(f'{path} was made for track_hash {meta.get("track_hash")}, this track is '
                         f'{track.track_hash} (centre line / smoothing / geometry changed) {redo}')
    if len(b['s']) != track.n or np.max(np.abs(b['s'] - track.s0)) > 1e-3:
        raise ValueError(f'{path}: s grid does not match the track ({len(b["s"])} vs {track.n} '
                         f'points) {redo}')
    track.set_boundaries(b['w_left'], b['w_right'], b['left_virtual'], b['right_virtual'])
    track.boundary_hash = compute_boundary_hash(track.track_hash, b['w_left'], b['w_right'],
                                                b['left_virtual'], b['right_virtual'])
    track.boundary_file = path
