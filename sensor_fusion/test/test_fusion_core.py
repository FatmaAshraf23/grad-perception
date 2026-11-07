#!/usr/bin/env python3
"""Tests for fusion_core.py (camera confirmation of LiDAR obstacles, 2026-11-07). No ROS.
Run: python3 test_fusion_core.py   (needs sensor_fusion.projection and apex_description.mounts importable)

Synthetic scenes: the TRUE car pose + true cones make the camera boxes (projection.cone_box, as fake_camera does);
the fusion gets the ESTIMATED pose (StateEstimate) and the LiDAR obstacle list, which may differ from the truth.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from sensor_fusion import fusion_core as fc
    from sensor_fusion import projection as pj
except ImportError:                                   # run from the staging folder
    sys.path.insert(0, os.path.join(HERE, '..'))
    from sensor_fusion import fusion_core as fc
    from sensor_fusion import projection as pj

RESULTS = []


def check(name, cond, info=''):
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'   [{info}]' if info else ''))


def setup_camera():
    """the simulator's camera: fake_camera's default lens (640 x 480) + sensor_mounts_sim.yaml's mount
    (x 0.20, z 0.15, pitch 8 deg down) -- built here without apex_description: base -> camera_link -> optical"""
    try:
        from ament_index_python.packages import get_package_share_directory
        cam = pj.CameraModel.from_yaml(os.path.join(get_package_share_directory('camera_publisher'), 'config',
                                                    'iphone_iriun_640x480.yaml'))
    except Exception:                                 # staging folder: the copy next to the tests
        cam = pj.CameraModel.from_yaml(os.path.join(HERE, 'iphone_iriun_640x480.yaml'))
    T_base_link = pj.T_from_xyz_rpy((0.20, 0.0, 0.15), (0.0, math.radians(8.0), 0.0))
    T_link_opt = pj.T_from_xyz_rpy((0, 0, 0), (-math.pi / 2, 0.0, -math.pi / 2))
    return cam, T_base_link @ T_link_opt


CAM, T_BO = setup_camera()


def boxes_for(true_pose, cones, height=0.25, conf=0.8, noise=0.0, rng=None):
    """what a perfect (or noisy) detector reports: cones = [(x, y, radius)]"""
    x, y, psi = true_pose
    T_opt_map = pj.invert(pj.T_from_xyz_rpy((x, y, 0.0), (0.0, 0.0, psi)) @ T_BO)
    out = []
    for cx, cy, r in cones:
        b = pj.cone_box(CAM, T_opt_map, (cx, cy), r, height)
        if b is None or b['visible_fraction'] < 0.6:
            continue
        u0, v0, u1, v1 = b['box']
        if noise and rng is not None:
            u0, v0, u1, v1 = (v + rng.normal(0, noise) for v in (u0, v0, u1, v1))
        out.append(('cone', conf, u0, v0, u1, v1))
    return out


def poses_const(est_pose, t0=0.0, t1=10.0, std_xy=0.03, std_psi=math.radians(0.5), r=0.0, mode=1, dt=0.01):
    h = fc.PoseHistory(keep_s=20.0)
    for k in range(int((t1 - t0) / dt) + 1):
        h.add(t0 + k * dt, est_pose[0], est_pose[1], est_pose[2], std_xy, std_psi, r, mode)
    return h


def ob(i, x, y, radius=0.1, std=0.03, seen=0.0):
    return dict(id=i, x=x, y=y, radius=radius, std_xy=std, last_seen_s=seen)


def run(fu, poses, true_pose, cones, obstacles, n=12, t0=1.0, dt=1 / 15, **kw):
    res = []
    for k in range(n):
        t = t0 + k * dt
        obs = [dict(o, last_seen_s=t) if o.get('seen', True) else o for o in obstacles]
        res.append(fu.picture(t, CAM.width, CAM.height, boxes_for(true_pose, cones, **kw), obs, poses))
    return res


def main():
    P0 = (0.0, 0.0, 0.0)
    check('bad parameters refused', all(_raises(lambda kw=kw: fc.CameraFusion(CAM, T_BO, **kw))
                                        for kw in ({'gate_k': 0}, {'size_ratio': 1.0}, {'nope': 1})))

    # PoseHistory
    h = fc.PoseHistory()
    h.add(0.0, 0, 0, math.radians(179), 0.03, 0.01, 0.0, 1)
    h.add(1.0, 1, 2, math.radians(-179), 0.05, 0.02, 0.2, 1)
    p = h.at(0.5)
    check('pose history: interpolates position, wraps the heading through +-180 deg, takes the larger std',
          abs(p['x'] - 0.5) < 1e-9 and abs(p['y'] - 1.0) < 1e-9 and abs(abs(p['psi']) - math.pi) < 1e-9
          and p['std_xy'] == 0.05 and p['ok'])
    check('pose history: outside -> None', h.at(-0.1) is None and h.at(1.1) is None)
    h.add(1.5, 1, 2, 0, 0.05, 0.02, 0.0, 2)
    check('pose history: not OK when either side is not mode OK', not h.at(1.2)['ok'])

    # 1. a cone 4 m ahead, perfect pose -> CONFIRMED after 2 pictures
    fu = fc.CameraFusion(CAM, T_BO)
    hp = poses_const(P0)
    r = run(fu, hp, P0, [(4.0, 0.0, 0.1)], [ob(1, 4.0, 0.0)], n=2)
    st = fu.state_of(1)
    check('cone 4 m ahead: CONFIRMED after 2 pictures, class cone, confidence 0.8',
          r == ['used', 'used'] and st['state'] == fc.CONFIRMED and st['class_name'] == 'cone'
          and abs(st['confidence'] - 0.8) < 1e-9, f"{fc.STATE_NAMES[st['state']]}, hits {st['hits']}")
    fu1 = fc.CameraFusion(CAM, T_BO)
    run(fu1, hp, P0, [(4.0, 0.0, 0.1)], [ob(1, 4.0, 0.0)], n=1)
    check('1 picture is not enough (confirm_hits 2): UNKNOWN', fu1.state_of(1)['state'] == fc.UNKNOWN
          and fu1.state_of(1)['hits'] == 1)

    # 2. a LiDAR obstacle with NO cone (cardboard box, map error): marked NO_CONE after miss_limit looks, never dropped
    fu = fc.CameraFusion(CAM, T_BO)
    run(fu, hp, P0, [], [ob(2, 4.0, 0.5)], n=9)
    s9 = fu.state_of(2)['state']
    run(fu, hp, P0, [], [ob(2, 4.0, 0.5)], n=1, t0=2.0)
    st = fu.state_of(2)
    check('obstacle without a cone: UNKNOWN after 9 looks, NO_CONE after 10', s9 == fc.UNKNOWN and st['state'] == fc.NO_CONE
          and st['looks'] == 10 and st['hits'] == 0)
    run(fu, hp, P0, [(4.0, 0.5, 0.1)], [ob(2, 4.0, 0.5)], n=2, t0=3.0)
    check('...and a cone appears there: CONFIRMED again (a hit ends the miss streak)', fu.state_of(2)['state'] == fc.CONFIRMED)

    # 3. a confirmed cone the detector then misses for a while: stays CONFIRMED until miss_limit misses in a row
    fu = fc.CameraFusion(CAM, T_BO)
    run(fu, hp, P0, [(5.0, -0.3, 0.1)], [ob(3, 5.0, -0.3)], n=3)
    run(fu, hp, P0, [], [ob(3, 5.0, -0.3)], n=9, t0=2.0)
    a = fu.state_of(3)['state']
    run(fu, hp, P0, [], [ob(3, 5.0, -0.3)], n=1, t0=4.0)
    check('confirmed cone missed 9 pictures in a row: still CONFIRMED; 10: NO_CONE',
          a == fc.CONFIRMED and fu.state_of(3)['state'] == fc.NO_CONE)

    # 4. two obstacles in the same direction: a near box (no cone) at 2 m and a cone at 6 m -- only the cone's box
    fu = fc.CameraFusion(CAM, T_BO)
    run(fu, hp, P0, [(6.0, 0.05, 0.1)], [ob(41, 2.0, 0.0), ob(42, 6.0, 0.05)], n=3)
    a, b = fu.state_of(41), fu.state_of(42)
    check('same direction, different distance: the far cone gets the box, the near obstacle does not',
          a['hits'] == 0 and b['state'] == fc.CONFIRMED, f"near hits {a['hits']}, far {fc.STATE_NAMES[b['state']]}")
    # the SIZE check is the camera's distance check: alone with the near obstacle (the far cone's own obstacle is not
    # in the list, so uniqueness cannot help), the near obstacle must NOT take the far cone's box
    fu = fc.CameraFusion(CAM, T_BO)
    run(fu, hp, P0, [(6.0, 0.05, 0.1)], [ob(43, 2.0, 0.0)], n=3)
    check('alone: a near obstacle does not take a far cone\'s box (size check)', fu.state_of(43)['hits'] == 0)
    # the vertical gate is wide on purpose (pitch while braking: fy * 1.5 deg = 26 px here), so it is NOT a distance
    # check on its own -- with the size gate opened the near obstacle WOULD take the box (documents the design)
    fu = fc.CameraFusion(CAM, T_BO, size_ratio=10.0)
    run(fu, hp, P0, [(6.0, 0.05, 0.1)], [ob(44, 2.0, 0.0)], n=3)
    check('size gate opened: the vertical gate alone does not separate 2 m from 6 m (why the size check exists)',
          fu.state_of(44)['hits'] > 0)

    # one box, two obstacles side by side (both inside the gate): the box confirms only ONE of them
    fu = fc.CameraFusion(CAM, T_BO)
    run(fu, hp, P0, [(5.0, 0.0, 0.1)], [ob(45, 5.0, 0.0), ob(46, 5.0, 0.12)], n=3)
    h45, h46 = fu.state_of(45)['hits'], fu.state_of(46)['hits']
    check('one box, two obstacles 12 cm apart: only the closer one gets it (one box -> one obstacle)',
          h45 == 3 and h46 == 0, f'hits {h45} / {h46}')

    # a box in the right column and of the right size but far too HIGH in the picture (a cone on a poster / shelf)
    fu = fc.CameraFusion(CAM, T_BO)
    b0 = boxes_for(P0, [(4.0, 0.0, 0.1)])[0]
    up = (b0[0], b0[1], b0[2], b0[3] - 200.0, b0[4], b0[5] - 200.0)
    for k in range(3):
        t = 1 + k / 15
        fu.picture(t, CAM.width, CAM.height, [up], [ob(47, 4.0, 0.0, seen=t)], hp)
    check('a same-size box 200 px too high (not standing where the obstacle is): no match', fu.state_of(47)['hits'] == 0)

    # 5. heading error vs the gate: StateEstimate says std_psi 0.5 deg
    def heading_case(err_deg, std_psi_deg, d=5.0):
        fu = fc.CameraFusion(CAM, T_BO)
        est = (0.0, 0.0, math.radians(err_deg))
        hpe = poses_const(est, std_psi=math.radians(std_psi_deg))
        run(fu, hpe, P0, [(d, 0.0, 0.1)], [ob(5, d, 0.0)], n=3)
        return fu.state_of(5)
    s1, s2, s3 = heading_case(1.0, 0.5), heading_case(5.0, 0.5), heading_case(5.0, 2.5)
    check('heading 1 deg off (std 0.5 deg): still matched', s1['state'] == fc.CONFIRMED)
    check('heading 5 deg off but StateEstimate claims 0.5 deg: NOT matched (the gate trusts the std)', s2['hits'] == 0)
    check('heading 5 deg off and std 2.5 deg: matched (an honest std widens the gate)', s3['state'] == fc.CONFIRMED)

    # 6. obstacles that must NOT count as looks
    fu = fc.CameraFusion(CAM, T_BO)
    obs = [ob(61, -3.0, 0.0), ob(62, 4.0, 6.0), ob(63, 12.0, 0.0), dict(ob(64, 4.0, 0.3), seen=False)]
    run(fu, hp, P0, [], obs, n=12)
    check('behind the car / outside the view / beyond max range: no looks',
          all(fu.state_of(i)['looks'] == 0 for i in (61, 62, 63)))
    check('remembered by the LiDAR but not seen for > 0.3 s (maybe behind a wall): no looks, stays UNKNOWN',
          fu.state_of(64)['looks'] == 0 and fu.state_of(64)['state'] == fc.UNKNOWN)

    # 7. timing: waiting for the pose, StateEstimate not OK
    fu = fc.CameraFusion(CAM, T_BO)
    hshort = poses_const(P0, t0=0.0, t1=1.0)
    w = fu.picture(1.05, CAM.width, CAM.height, [], [ob(7, 4.0, 0.0, seen=1.05)], hshort)
    o = fu.picture(-1.0, CAM.width, CAM.height, [], [ob(7, 4.0, 0.0, seen=1.05)], hshort)
    check("picture newer than the newest StateEstimate: 'wait'; older than the history: 'old'", w == 'wait' and o == 'old')
    hbad = poses_const(P0, mode=2)
    r = run(fu, hbad, P0, [(4.0, 0.0, 0.1)], [ob(71, 4.0, 0.0)], n=3)
    check('StateEstimate not OK: pictures not used, nothing counted', r == ['not_ok'] * 3 and fu.state_of(71)['looks'] == 0
          and fu.state_of(71)['hits'] == 0)

    # 8. camera delay: the car turns at 1 rad/s; the boxes come from a picture taken 0.10 s before its stamp
    def delay_case(delay_param):
        fu = fc.CameraFusion(CAM, T_BO, camera_delay_s=delay_param, sigma_time_s=0.0)
        h = fc.PoseHistory(keep_s=20)
        for k in range(1001):
            t = k * 0.01
            h.add(t, 0.0, 0.0, 1.0 * t, 0.02, math.radians(0.3), 1.0, 1)
        hits = 0
        for k in range(6):
            t_true = 2.0 + k / 15                    # when the picture really was taken
            yaw = 1.0 * t_true
            cone = (4.0 * math.cos(yaw), 4.0 * math.sin(yaw), 0.1)
            bx = boxes_for((0.0, 0.0, yaw), [cone])
            fu.picture(t_true + 0.10, CAM.width, CAM.height, bx, [ob(8, cone[0], cone[1], seen=t_true + 0.1)], h)
            hits = fu.state_of(8)['hits']
        return hits
    h_bad, h_good = delay_case(0.0), delay_case(-0.10)
    check('turning at 1 rad/s, stamps 100 ms late: wrong camera_delay_s -> no match; measured delay -> all matched',
          h_bad == 0 and h_good == 6, f'hits {h_bad} vs {h_good} of 6')

    # 9. small far cone: no look (too few pixels for YOLO), a false box: no effect
    fu = fc.CameraFusion(CAM, T_BO, max_range_m=40.0)
    run(fu, hp, P0, [], [ob(9, 30.0, 0.0)], n=12)
    check('cone so far its box is < min_box_px: no looks', fu.state_of(9)['looks'] == 0)
    fu = fc.CameraFusion(CAM, T_BO)
    fb = [('cone', 0.4, 50.0, 300.0, 70.0, 340.0)]
    for k in range(5):
        t = 1 + k / 15
        fu.picture(t, CAM.width, CAM.height, boxes_for(P0, [(4.0, 0.0, 0.1)]) + fb, [ob(91, 4.0, 0.0, seen=t)], hp)
    check('a false box elsewhere: ignored (counted as unmatched), the real cone still CONFIRMED',
          fu.state_of(91)['state'] == fc.CONFIRMED and fu.stats['unmatched_boxes'] == 5)

    # 10. bookkeeping
    fu.forget_except([])
    check('forget_except: evidence of obstacles that left the list is dropped', fu.ev == {})
    fu = fc.CameraFusion(CAM, T_BO)
    half = CAM.scaled(320, 240)
    T_opt_map = pj.invert(T_BO)
    b = pj.cone_box(half, T_opt_map, (4.0, 0.0), 0.1, 0.25)['box']
    for k in range(2):
        t = 1 + k / 15
        fu.picture(t, 320, 240, [('cone', 0.7) + tuple(b)], [ob(10, 4.0, 0.0, seen=t)], hp)
    check('detections from a 320 x 240 stream: the camera model is scaled, still CONFIRMED',
          fu.state_of(10)['state'] == fc.CONFIRMED)
    check('a low-confidence box (< min_conf) is ignored',
          _conf_ignored())

    # 11. noisy boxes + a realistic pose error, many cones around a curve: no wrong pairings
    rng = np.random.default_rng(1)
    fu = fc.CameraFusion(CAM, T_BO)
    true = (0.0, 0.0, 0.0)
    est = (0.04, -0.03, math.radians(0.6))
    hp2 = poses_const(est, std_xy=0.04, std_psi=math.radians(0.6))
    cones = [(3.0, -0.6, 0.1), (3.5, 0.7, 0.1), (5.0, 0.0, 0.1), (6.5, 1.2, 0.1), (7.5, -1.0, 0.1)]
    obstacles = [ob(100 + k, c[0] + rng.normal(0, 0.03), c[1] + rng.normal(0, 0.03)) for k, c in enumerate(cones)]
    wrong = 0
    for k in range(30):
        t = 1 + k / 15
        bx = boxes_for(true, cones, noise=2.0, rng=rng)
        fu.picture(t, CAM.width, CAM.height, bx, [dict(o, last_seen_s=t) for o in obstacles], hp2)
        T_opt_map = pj.invert(pj.T_from_xyz_rpy((0, 0, 0), (0, 0, 0)) @ T_BO)
        for oid, j, _ in fu.last_matches:            # box j must be the box of cone oid - 100
            cu = 0.5 * (bx[j][2] + bx[j][4])
            want = pj.cone_box(CAM, T_opt_map, cones[oid - 100][:2], 0.1, 0.25)['box']
            if abs(cu - 0.5 * (want[0] + want[2])) > 15:
                wrong += 1
    states = [fc.STATE_NAMES[fu.state_of(100 + k)['state']] for k in range(5)]
    check('5 cones, noisy boxes (2 px), pose 5 cm / 0.6 deg off: all CONFIRMED, no wrong pairing',
          all(s == 'CONFIRMED' for s in states) and wrong == 0, f'{states}, wrong {wrong}')

    print(f'\n{sum(RESULTS)} of {len(RESULTS)} passed')
    return 0 if all(RESULTS) else 1


def _raises(f):
    try:
        f()
    except ValueError:
        return True
    return False


def _conf_ignored():
    fu = fc.CameraFusion(CAM, T_BO)
    hp = poses_const((0.0, 0.0, 0.0))
    b = boxes_for((0.0, 0.0, 0.0), [(4.0, 0.0, 0.1)], conf=0.2)
    for k in range(3):
        t = 1 + k / 15
        fu.picture(t, CAM.width, CAM.height, b, [ob(11, 4.0, 0.0, seen=t)], hp)
    return fu.state_of(11)['hits'] == 0


def test_fusion_core():
    """pytest entry (colcon test): every check above must pass"""
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
