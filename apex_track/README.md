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

## Track boundaries (walls) — added 2026-10-05 for planning
```python
w_left, w_right = track.width_at(s)        # m from the centre line to the left / right wall (any s, s_abs or array)
room_left, room_right = w_left - e_y, w_right + e_y     # with s, e_y from the StateEstimate
left_xy, right_xy = track.boundaries_xy()  # the walls as (n, 2) polylines in the map frame
track.left_virtual, track.right_virtual    # True where there is NO real wall (opening closed by a straight line)
print(track.boundary_hash)                 # planning checks this one (see below)
```
- Drivable corridor at s:  `-w_right(s) <= e_y <= w_left(s)`, measured along the normal like e_y. **Raw walls** — each team
  subtracts its own margin (car half-width + buffer).
- `width_at(s, d_min=0.1)` (default) cuts the INSIDE of tight corners so that D = 1 − κ·e_y ≥ 0.1: beyond the corner's
  centre of curvature Frenet coordinates are not unique. `d_min=None` = raw wall distance. Levine: the raw inner wall
  in Corner 1 is past that point (D = −0.14), 36 of 1255 points are cut.
- Made from the occupancy map (the same map AMCL uses) by ray casting along the normals:
  `python3 -m apex_track.boundaries tracks/<name>/track.yaml --map <map.yaml>` → `tracks/<name>/boundaries.csv`,
  then `boundaries: boundaries.csv` in track.yaml. Needs Pillow (`pip install pillow`) — only this tool, not the runtime.
  Accuracy ≈ half a map pixel diagonal (5 cm map → ~3.5 cm). Openings (doors, junctions) wider than `--max_w` (2 m) are
  closed by a straight virtual wall and flagged.
- **Hashes:** `track_hash` (centre line + smoothing + code) is NOT changed by boundaries — the StateEstimate check stays
  as it is. `boundary_hash` fingerprints the widths. A boundaries file is tied to one track_hash (its header);
  `load_track()` refuses it if the centre line or smoothing changed → remake it.
  Levine: track_hash a44e46d238cf50c2, boundary_hash 4b4ba85fac93f3ba.

## Track identity
A track = `tracks/<name>/track.yaml` + `centerline.csv`. `track_hash` is a fingerprint of the
centre-line points (1 mm), `smooth_m`, `ds` and the geometry code version. Every StateEstimate carries it;
receivers reject messages whose hash differs from their own.
**Never edit a track in place without telling the team** — the hash will (correctly) change.

## Tests
`python3 test/test_frenet.py`, `python3 test/test_track_hash.py`, `python3 test/test_vectors.py`, `python3 test/test_boundaries.py` (needs Pillow) — all must print `ALL PASS`.

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
