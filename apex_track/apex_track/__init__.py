"""apex_track -- the SHARED track geometry package of the APEX team.

Perception, planning, simulation and control all import this package, so
global (x, y, yaw) <-> Frenet (s, e_y, e_psi, kappa) is computed by ONE
implementation from ONE track definition. Pure Python + numpy + PyYAML,
no ROS needed (works on the Pi 4, in WSL, on Windows, in a notebook).

    from apex_track import load_track
    track = load_track('tracks/levine/track.yaml')
    s_abs, s, e_y, e_psi, kappa = track.project(x, y, yaw)
    print(track.track_id, track.track_hash)
"""
from .frenet import FrenetTrack, load_xy_csv, wrap
from .track import GEOMETRY_VERSION, compute_track_hash, load_track

__all__ = ['FrenetTrack', 'load_xy_csv', 'wrap',
           'GEOMETRY_VERSION', 'compute_track_hash', 'load_track']
__version__ = '0.1.0'
