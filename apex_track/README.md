# apex_track — shared track geometry for the APEX team

One implementation of **global pose ↔ Frenet** for everyone (perception, planning, simulation, control),
as agreed in the Perception → Control state contract (control's response, 2026-10-02).

## Install
- **ROS 2:** copy this folder to `~/ros2_ws/src/apex_track`, then
  `cd ~/ros2_ws && colcon build --packages-select apex_track && source install/setup.bash`
- **Without ROS:** `pip install -e .` inside this folder (needs numpy + pyyaml).

## Use
```python
from apex_track import load_track
track = load_track('tracks/levine/track.yaml')       # in ROS: get_package_share_directory('apex_track') + '/tracks/levine/track.yaml'
s_abs, s, e_y, e_psi, kappa = track.project(x, y, yaw)   # call every update, in time order (it counts laps)
print(track.track_id, track.track_hash, track.L)
kappa = track.kappa_at(s_horizon)        # curvature for a whole MPC horizon (numpy array), any s / s_abs
```
- `s_abs` [m] continuous across laps (never wraps), `s` = s_abs mod L
- `e_y` [m] + = left of the centre line; `e_psi` [rad] = yaw − tangent; `kappa` [1/m] + = left turn
- Map frame, SI units, CG reference point (the caller passes the CG pose).

## Track identity
A track = `tracks/<name>/track.yaml` + `centerline.csv`. `track_hash` is a fingerprint of the
centre-line points (1 mm), `smooth_m`, `ds` and the geometry code version. Every StateEstimate carries it;
receivers reject messages whose hash differs from their own.
**Never edit a track in place without telling the team** — the hash will (correctly) change.

## Tests
`python3 test/test_frenet.py`, `python3 test/test_track_hash.py`, `python3 test/test_vectors.py` — all must print `ALL PASS`.

`test_vectors.py` checks control's tolerance (1 mm / 0.001 rad): pose → Frenet → pose round trip at the
start/finish seam, the 3 sharpest corners and every 2 cm of the track; s_abs continuity driving forward and
backward over the seam; `kappa_at` = `project()` kappa; and the golden file `tracks/<name>/test_vectors.csv`
(21 poses with answers — give it to anyone checking their own implementation).
Regenerate the golden file ONLY after an intended geometry change: `python3 -m apex_track.vectors tracks/<name>/track.yaml`.

## Geometry versions (part of track_hash)
| Version | Date | Change |
|---|---|---|
| 1 | 2026-10-01 | Smoothed, resampled centre line; closest point on the 5 cm polyline |
| 2 | 2026-10-02 | Closest point refined onto the smooth tangent (Newton steps). v1 was off by up to 11 mm in s and 17 mrad in e_psi in Levine's sharpest corners. Levine hash 3bfa397429695ccb → a44e46d238cf50c2 |
