#!/usr/bin/env python3
"""EKF version 3: wheel speed + gyro, corrected by AMCL (LiDAR vs wall map).

The track is bounded by walls (no cones), so the LiDAR correction now comes
from AMCL: a particle filter that matches each scan against the wall map and
reports a pose (x, y, yaw) with a covariance.

TWO INSTANCES OF THIS NODE RUN AT THE SAME TIME (parameter `mode`)

  mode = 'odom'  -> the LOCAL EKF ("ekf_local")
      Wheel + gyro + ZUPT only. Never jumps, drifts slowly.
      Publishes TF  odom -> est/base_link  (smooth odometry for AMCL).
  mode = 'map'   -> the GLOBAL EKF ("ekf_global")
      Same prediction, plus the AMCL pose update. Drift-free.
      Publishes /odom/ekf in the map frame: the output for planning and MPC.

  Why two? AMCL uses odometry (odom -> base_link) to move its particles. If
  that odometry already contained AMCL's own corrections, AMCL would be fed
  its own output (a feedback loop that makes errors look smaller than they
  are). So AMCL gets the pure odometry of the local EKF, and the global EKF
  fuses AMCL on top. This is the standard ROS layout (REP-105):
        map --(AMCL)--> odom --(ekf_local)--> est/base_link --> est/laser

STATE (both modes)
    x = [ px, py, yaw, b, k ]
        px, py, yaw : pose in the node's frame (map or odom)
        b           : gyro bias                 [rad/s]
        k           : wheel-speed scale, v = k * v_meas
                      (only observable in the global EKF, via AMCL)

PREDICT (100 Hz, every IMU message) -- unchanged from v2
    v = k*v_meas,  w = w_meas - b
    px += v cos(yaw+beta) dt,  py += v sin(yaw+beta) dt,  yaw += w dt
    P  = F P F^T + G Qu G^T + Q_b + Q_k

UPDATE 1 -- ZUPT while stopped (both modes)
    While the wheels stand still the true turn rate should be 0, so the gyro reading is bias + noise:
    z = w_meas, h(x) = b, R = sigma_w^2 -> the gyro bias is learned (zero-velocity update).
    STANDSTILL GATE (2026-10-24, parameter zupt_gate, LocalizationEKF.standstill): a reading far from
    the bias we know,  NIS = (w_meas - b)^2 / (P_bb + sigma_w^2) > zupt_gate  (16 = 4 sigma),
    means the car IS turning while its wheels stand still (lifted and turned by hand on the start
    line). Then: no bias update, and the heading follows the gyro (a normal prediction with v = 0).
    Why (job 195, F10 parked): a 15 deg turn of the parked car was learned as bias, +9.6 deg/s
    instead of +0.6, and after the start the heading drifted 6-9 deg/s: lost for the whole run.
    The gate widens by itself while the bias is still uncertain (P_bb large at start-up), so an
    unknown start bias is still learned. zupt_gate 0 = the old rule (always ZUPT).
    REAL CAR: sigma_w from the IMU bench test (imu_static_check) -- a too small sigma_w makes the gate
    reject ordinary noise; sigma_bias0 must cover the bias left after the IMU driver's start-up
    calibration. Known limit: a turn slower than ~4 sigma_w (~1 deg/s in the sim) is still absorbed
    into the bias; the map check flags the heading error that follows.
    LONG "TURN" = BIAS CHANGE (parameter zupt_turn_max_s, StandstillTurns): a hand turn takes a second
    or two, but a gyro bias that CHANGED while driving (warm-up; the local EKF has no AMCL to track it)
    looks like a turn that never ends. A standstill "turn" longer than zupt_turn_max_s (3 s) is taken as
    a bias change: the heading change it made is undone, P_bb is opened again (at least sigma_bias0^2
    and the episode's mean rate^2, so the next reading passes the gate) and the bias is learned again.
    Price: a single continuous hand turn longer than 3 s is undone too (turn the car in short moves).

UPDATE 2 -- AMCL pose (global EKF only)                       NEW in v3
    z = [x, y, yaw] from /amcl_pose,   h(x) = [px, py, yaw],
    H = [ I3 | 0 ]   (AMCL measures the first three states directly)
    R = AMCL's own 3x3 covariance * amcl_cov_scale, with a minimum
        (inflated because AMCL and this EKF share the same odometry, so their
         errors are correlated -- treating them as independent would make the
         EKF overconfident)
        amcl_cov_scale 4 (2026-10-06, was 2): with AMCL's motion noise lowered
        to 0.02 (amcl_sim.yaml) AMCL reports a much smaller covariance, but its
        errors still change slowly (~7 fixes/s that share one error), so the EKF
        averaged them as if independent and became overconfident (run H: only
        80-93 % of position errors inside 2 sigma). Offline replay: scale 4 ->
        98-99 % inside 2 sigma, s error p95 4.8 -> 4.2 cm.
    HALF-CELL CORRECTION (2026-10-06): z uses AMCL's x and y + amcl_offset_m.
        nav2 AMCL puts the CENTRE of map cell i at  origin + i * resolution
        (nav2_amcl map.hpp MAP_GXWX, amcl_node.cpp convertMap), but ROS -- the
        map_server / RViz convention, the simulator, our apex_track track files --
        puts it at  origin + (i + 0.5) * resolution. So AMCL sees every wall half
        a cell towards -x and -y, and its pose comes out shifted by
        (-res/2, -res/2): measured (-2.9, -2.1) cm on closed1 and (-3.1, -2.4) cm
        on Levine (5 cm maps). On the track that is an along-track error whose
        sign depends on the driving direction (behind when driving +x, ahead
        when driving -x), and in a corner it becomes kappa * ds of heading
        error. Offline replay: s error p95 5.9 -> 3.3 cm; run K re-projected:
        corner e_psi p95 2.48 -> 1.03 deg, whole lap 1.88 -> 0.66 deg.
        amcl_offset_m must be HALF THE RESOLUTION OF THE MAP AMCL USES (0.025 for
        a 5 cm map, 0.0125 for 2.5 cm); the node compares it with /map once at
        start-up and warns if it doesn't match. 0 = no correction.
        Not corrected: AMCL's own /amcl_pose and its TF map -> odom.
    Gate: NIS < 11.34 (chi-square, 3 degrees of freedom, 99 %)

    DELAYED MEASUREMENT: the AMCL pose belongs to the moment the scan was
    taken (~0.1-0.2 s ago); the car has moved since. So the innovation is
    computed against the EKF state stored for THAT moment:
        nu = z - x_history(t_scan)
    and the correction K*nu is added to the current state (and to the stored
    history, so the next measurement doesn't correct the same error twice).

    SPEED-SCALE CALIBRATION FILE                                 NEW 2026-10-01
    On a straight corridor AMCL cannot see along-track error, so k is only
    learned at the first corner. Starting from k = 1 with a 3 % wheel error
    gave ~20 cm along-track error after 8 m in every evaluation run. Now:
      - at start-up BOTH EKFs read k from `scale_file` (if it exists) and
        start with sigma_k = max(saved sigma, scale_file_min_sigma).
        `scale_file` is the ANCHOR: written only by `wheel_calib` (tape-
        measured straight drive) or by hand -- NEVER by this node;
      - the GLOBAL EKF writes the k it learned to `scale_learned_file`
        (every 10 s after `scale_save_min_distance` m, and at shutdown) and
        warns if it differs from the anchor by more than `scale_warn` (1 %)
        -> on the real car: tyres changed/worn, redo wheel_calib.
    Why not write the learned k back into scale_file? Tried 2026-10-01: it is
    a feedback loop (file -> local EKF -> AMCL odometry -> AMCL pose on
    straights -> global EKF -> file). In 3 runs k slid 0.9706 -> 0.9689 ->
    0.9683 -> 0.9679 (true 0.9709) while every uncalibrated run ended within
    0.05 % of the truth.

    STEERING CALIBRATION (2026-10-03): beta uses the steering corrected with the
    file of steer_calib: delta = gain * (delta_measured - offset) (calibration.py,
    parameter steer_calib_file; no file = no correction). A +0.01 rad steering
    offset made beta 0.30 deg too large and the heading 0.30 deg too low.

    SIDESLIP MODEL (2026-10-18, vehicle_model.py, parameter beta_model):
      'steer' = the kinematic beta above (the default until 2026-10-18).
      'slip'  = DEFAULT since 2026-10-18 (job 152 live A/B, Fatma: "commit it"):
                beta from the GYRO and the speed with the rear tyres' slip:
                vy = r * (lr - K * vx^2), beta = atan(vy / vx), r = gyro - bias, vx = k * v_meas.
      Why: at racing speed the tyres slip, the kinematic beta is too large in fast corners
      (closed1 sweep job 148: vy error = 0.05 s * v * r, EKF position p95 4 -> 24 cm from 1.5 to
      6 m/s, inside 2 sigma 54 %). Offline replay (job 151 recordings): 16-22 -> 4-9 cm at
      3-6 m/s, as good as the TRUE sideslip, 100 % inside 2 sigma. K: slip_k >= 0, or (slip_k < 0)
      from slip_file (written by slip_calib); no file -> K = 0 + warning. Both EKFs use it (the
      local one feeds AMCL's odometry). Live A/B closed1 (job 152, K 0.017): EKF position p95
      14.7 / 20.9 / 23.6 -> 5.1 / 6.8 / 7.6 cm at 3 / 4.5 / 6 m/s, inside 2 sigma 54-66 -> 100 %;
      1.5 m/s 4.0 -> 5.1 cm (AMCL's own along-track bias no longer cancelled by the old beta error).

    WHEEL-SPEED DELAY COMPENSATION (2026-10-19, wheel_delay.py, parameter wheel_delay_s):
      The wheel speed is late: its own delay (an encoder speed = counts over a window -> it describes
      the window's middle; + the link; simulator 10-12 ms) + the EKF holds each value until the next
      one (0-20 ms at 50 Hz). Accelerating, the integrated distance falls behind by (speed gained) x
      (delay), ~6 cm from 2 to 6 m/s, invisible on a featureless straight -- and ekf_local is AMCL's
      odometry, so AMCL's particles fell behind too (it was the speed-dependent part of AMCL's
      along-track slide). Now: v = v_wheel + integral of the IMU's forward acceleration (minus its
      bias, learned while standing) since the moment the wheel value was true, i.e. over
      [t_wheel - wheel_delay_s, now] -- only 10-30 ms, restarted at every wheel message.
      wheel_delay_s < 0 = off (the node default); >= 0 = on (the simulator launch: 0.010; real car:
      measure it). imu_yaw_deg = how the IMU's x axis is turned against the car's forward axis (0 =
      x forward). The compensated speed is also the published twist (-> StateEstimate vx).
      Offline (job-154 recordings, the same scans): AMCL along-track at 6 m/s mean -8.2 -> -4.8 cm,
      p95 17.8 -> 12.8 cm; the error no longer grows with speed.

    RECOVERY = JUMP TO AMCL, ONLY WHEN AMCL IS CLEARLY RIGHT (2026-10-06).
    Before: after `reset_after_rejects` (10) AMCL poses rejected in a row the EKF
    assumed IT was lost and jumped to AMCL. When AMCL was the one that was lost
    (closed1 with a solid-black map, jobs 068-070) the EKF followed it into the
    walls, 10-20 m off, and reported OK. Now:
      1. AMCL poses with sqrt(cov_xx + cov_yy) > amcl_max_sigma (0.35 m) are
         ignored -- AMCL itself says it is unsure (lost AMCL: > 0.2 m 99.9 % of
         the time; normal laps: max 0.29 m) -- UNLESS it is a CORRIDOR POSE
         (2026-10-19, amcl_corridor_sigma, function amcl_pose_class): the
         ellipse's NARROW axis sigma <= amcl_corridor_sigma (0.10 m). On a long
         featureless straight the two walls fix the car's position ACROSS the
         track (and its heading), not ALONG it: AMCL's particle cloud becomes a
         long thin ellipse (closed1 at 6 m/s: along ~40 cm, across ~5.5 cm), and
         the one-number size test threw away the good across-track part too.
         Up to 1.6 s without an accepted fix -> StateEstimate DEGRADED / LOST
         at 6 m/s while the EKF was right (sigma 3.8 cm, error p95 4.4 cm).
         A corridor pose is used as a normal fix: R is AMCL's own covariance,
         so the Kalman gain is ~0.1 across and ~0.002 along (it corrects e_y and
         heading, hardly s); the NIS gate still applies; it NEVER counts towards
         a jump (a jump replaces the whole pose, so AMCL must be sure in every
         direction). Evidence: the 12 solid-map lost-AMCL sequences (4566 lost
         poses): narrow sigma 12-150 cm (5th percentile); "<= 0.35 OR narrow
         <= 0.10" lets through EXACTLY the same 76 lost poses as "<= 0.35"
         alone (narrow <= 0.25 would add 885 -> 0.10 on purpose); normal laps:
         narrow sigma max 8.1 cm. Offline replay of jobs 158/159 (6 m/s):
         OK while driving 94.6 -> 100 % / 97.2 -> 99.7 %, accuracy unchanged.
         amcl_corridor_sigma 0 = off (the old rule). REAL CAR: check the narrow
         sigma of AMCL on a real recording (like amcl_max_sigma).
      2. After `reset_after_rejects` confident AMCL poses rejected in a row the
         EKF jumps only if BOTH checks pass, otherwise it keeps going on odometry
         (StateEstimate reports LOST after 1 s of driving without a fix):
         a. AMCL MOVED LIKE THE ODOMETRY over those poses (motion_consistent:
            first pose to every later one, each in its own frame, within
            reset_tol_pos + reset_tol_frac * distance and reset_tol_yaw_deg);
         b. THE LIDAR SCAN FITS THE MAP CLEARLY BETTER AT AMCL'S POSE than at
            the EKF's (scan_fit: share of scan_beams beams whose measured range
            matches the map's within scan_tol; AMCL >= reset_scan_min and
            >= EKF + reset_scan_margin). (a) alone is not enough: a lost AMCL
            moves with the odometry too.
         Otherwise the oldest of the poses is dropped and the next one re-checks.
    Offline (robust.py, recorded laps): AMCL lost (12 solid-map runs): old rule
    52 jumps onto the wrong pose, EKF up to 23 m off -> new rule 0 jumps, EKF
    error p95 0.24 m / max 0.34 m; EKF lost (40 injected faults, AMCL fine):
    recovered in 1.4 s (up to ~6 s when the error lies along a featureless
    corridor, where the scan looks the same); normal laps unchanged.
    The scan check needs the map (map_topic) and the scan AMCL uses (scan_topic),
    and where the LiDAR sits on the car (laser_x, laser_y, laser_yaw).
    test_fault (TEST ONLY, default off): 't_s,dx,dy,dyaw_deg' moves ekf_global's
    pose once, t_s seconds after its first IMU message -- a "kidnapped" EKF while
    AMCL is fine, to test the jump live.
    test_kick (TEST ONLY, 2026-10-20, default off): the same at RUNTIME --
    `ros2 param set /ekf_global test_kick 'dx,dy,dyaw_deg'` moves the pose at the next IMU
    message (fault scenarios for control, task 3(c)); setting it again with a new value kicks again.
    LOCKED (2026-10-24): both work only when allow_test_faults is true -- a READ-ONLY parameter set
    once at start-up, by the SIMULATOR launch files (sim_localization / sim_eval: default true there).
    Node default false = the real car: test_kick is then not even declared (`ros2 param set` is
    refused) and a test_fault value is ignored with an error in the log. state_pipeline's
    test_lag_ms follows ekf_global's switch. Never set allow_test_faults in the real car's launch.

MAP CHECK -- along-track position vs the LiDAR (global EKF only, 2026-10-21)  NEW
    Why: on a straight AMCL only knows the ACROSS-track position (corridor fixes), so an
    along-track error (wheel over-read / spin, a pose kick) stayed invisible until the next
    corner -- fault scenarios F05 / F06 / F08: mode OK for 1.0-2.3 s with 0.5-0.9 m error.
    What: for every scan, slide the EKF pose along its heading (+-map_check_max_shift in
    map_check_step) and find where the scan fits the map best (map_check.py: distance field
    built once from the map -> one table lookup per beam and shift, ~0.1 ms per scan on the
    laptop, est. 0.3-0.5 ms on a Pi 5). Alarm when map_check_k scans in a row say the pose is
    map_check_shift_min .. (max_shift - step) off with a clear (gain >= map_check_gain_min),
    sharp (width <= map_check_width_max) optimum.
    Offline (jobs 178-181, 22 recorded runs): measures the along error to ~5.5 cm (median),
    0 false alarms on 5603 clean scans, F08 detected after 0.19 s (mode today: 2.4 s), F05 0.59 s
    (2.3 s), F06 0.88 s (1.5 s).
    ALARM WITH HYSTERESIS (contract v0.3, 2026-10-24; map_check.MismatchAlarm). Every checked scan is
      BAD    the along rule above, OR fit0 < map_check_fit_bad: the scan hardly fits the map at the EKF
             pose at all (sideways / heading errors, poses beyond the search window -- F04, F09, F10),
      CLEAN  no shift fits more than map_check_gain_clear better AND fit0 >= map_check_fit_clear,
      or neither (keeps the state). ON after map_check_k BAD scans in a row; OFF after
    map_check_k_clear CLEAN scans in a row and >= map_check_min_on_s on. The alarm goes to state_estimate
    (map_check_hooks in state_pipeline, or the topic) -> FLAG_MAP_MISMATCH (1024) + mode at least
    DEGRADED. The EKF itself does not use it (its state and covariance are unchanged).
    Offline with the same class (2026-10-24; 12 clean + 10 fault runs): 0 false alarms, 0 premature
    clears; time with the EKF > 30 cm off while the mode said OK 10.3 s -> 1.6 s over the 10 faults.
    The distance field is built in a BACKGROUND THREAD: numpy without scipy needs 0.1-0.23 s on the
    laptop, and inside the map callback that stalled the whole estimator (jobs 182-183: a 0.13-0.19 s
    lag spike at start-up; a Pi would take longer; a re-published map would do it mid-race).
    CPU guard (map_check_budget_ms): every run is timed; if the mean cost exceeds the budget the
    check runs only on every n-th scan (n <= map_check_max_every) instead of slowing the EKF.
    REAL CAR: the thresholds come from the simulator, where the LiDAR sees exactly the map;
    refit on real recordings (people near the walls, an imperfect map lower the fit).

UPDATE 3 -- WHEEL-SLIP CHECK (both modes, 2026-10-26, Fix 3; slip_check.py)
    Why: the prediction moves the car with the wheel speed. Fault F05 (wheel x 1.15 for 1 s = wheelspin) gave
    90 cm of error, F06 (x 0.85 for 0.6 s = skid) 52 cm, and AMCL cannot see it on a straight (it moves its
    particles with the same wheel odometry, from ekf_local).
    What: slip_check.SlipCheck integrates the IMU's forward acceleration (a_x - accelerometer bias) to a second
    speed, pulled gently towards the wheel while they agree; ON when the wheel (k * v_meas) differs from it by
    > slip_r0 + slip_c_a * |a| for slip_k_on samples in a row (each wheel sample compared slip_lag_s later:
    the simulated IMU lags ~30 ms more than the wheel). Bias learned at standstill AND while driving
    (slip_ki_bias); OFF threshold grows with the slip's duration (slip_r_off_rate). Offline (36 recorded runs,
    also with a 0.1 m/s^2 bias change while driving): 0 false alarms, F05/F06 detected after 40-53 ms, off
    82-139 ms after the fault (slip_check docstring).
    While ON (slip_use_observer true): predict(observer speed, use_k=False) -- the speed in TRUE m/s, the wheel
    scale k has no part in it (otherwise AMCL fixes during a slip would "learn" a wrong wheel scale), speed
    noise slip_sigma_v instead of sigma_v; standstill (ZUPT) is NOT decided while ON (a locked wheel during
    hard braking reads 0 while the car still slides); the odometry twist carries the observer speed.
    slip_use_observer false = flag only (shadow mode for the first real-car runs).
    Out: <diag_prefix>/slip (one per decided wheel sample) and slip_hooks (state_pipeline -> state_estimate
    FLAG_WHEEL_SLIP (8) -> DEGRADED); log lines 'WHEEL SLIP: ...' / 'wheel slip over after ...'.
    SIMULATOR ARTIFACT: with fake_vehicle_sensors v3 the simulated wheel dips when the simulator runs slower
    than real time (looks exactly like a skid; 2-14 % of clean driving). sim_wheel_ideal_topic (sim launch:
    /sim/wheel_ideal) feeds slip_check.SimGate, which suppresses decisions there. '' on the real car.
    REAL CAR: measure slip_lag_s (wheel vs IMU step test), mount the IMU near the centre (or correct the lever
    arm), refit slip_r0 above the normal tyre slip of hard acceleration; start with slip_use_observer false.

UPDATE 4 -- IMU DROPOUT: predict from wheel + steering (both modes, 2026-10-27, Fix 2; imu_gap.py)
    Why: the EKF predicted only when an IMU sample arrived -> during a dropout the estimate stood still, and a gap
    > 0.5 s was skipped completely when the IMU came back (fault F04, 1 s at 6 m/s: ~6 m lost in both EKFs, AMCL lost
    too, never recovered). Control chose Option A (26 Oct): keep publishing during a dropout, DEGRADED + FLAG_IMU_STALE,
    LOST after 50 ms (provisional), no automatic recovery -- approved for isolated development and synthetic
    evaluation; the wheel-based stamp needs a contract amendment before operational use.
    imu_gap_predict false (THE NODE DEFAULT) = the old behaviour exactly. The simulator launch files set it true.
    What (imu_gap.ImuGap): at every wheel sample, no IMU sample for > imu_gap_s (25 ms) = DROPOUT -- by the stamps AND
    by the receive times (this node's clock when the callbacks run): the wheel message must be fresh and no IMU message
    handled for > imu_gap_s, else it is a processing backlog / stall, not a dropout (job 207: the stamps alone gave
    false dropouts at start-up, 2 of 10 runs). Then each wheel
    sample predicts to its own stamp: speed k*v_meas, yaw rate r = v tan(steer)/(L + imu_gap_kus v^2) + (last gyro -
    model at the gap start) * exp(-t/imu_gap_tau_b), yaw-rate noise imu_gap_sigma_w per step (0.4 rad/s: 97-100 % of the
    offline errors inside 2 sigma); /odom/ekf is published at the WHEEL rate with the wheel sample's stamp.
    The time step is now measured from the state's own time (t_state), so the first IMU sample after a gap continues
    from the last wheel prediction -- nothing is skipped.
    Recovery: the returning gyro is NOT trusted until imu_gap_rec_n samples in a row agree with the wheel model within
    imu_gap_rec_tol (control: no automatic clear); until then the model's yaw rate is used, no ZUPT, the slip check and
    the wheel-delay compensation get no IMU data. <prefix>/imu_gap + gap_hooks -> state_estimate (FLAG_IMU_STALE,
    LOST after imu_gap_lost_s without a trusted gyro, DEGRADED until a fresh AMCL fix after the recovery).
    Offline (gap_eval.py, validation, p95 heading / lateral at the end of the gap): 0.2 s 1.7 deg / 1.1 cm,
    0.5 s 4.0 deg / 4.8 cm, 1 s 6.5 deg / 17 cm (the old code: 4.5 deg / 6.5 cm, 23 deg / 44 cm, 68 deg / 1.9 m + 4.5 m).
    REAL CAR: imu_gap_s from the IMU's real sample spacing (~2.5 periods), imu_gap_kus fitted on real laps, steering =
    the commanded angle (servo lag); keep imu_gap_predict false until control has accepted the stamp amendment.

Topics
  in : /imu/data (gyro z; + accel x/y with wheel_delay_s >= 0 or slip_check), /vehicle/measured,
       /amcl_pose (global), /initialpose (global)
       <map_topic> (global, read once: checks amcl_offset_m against the resolution;
                    the walls for the scan check), <scan_topic> (global, scan check)
       <truth_topic> (global, simulation truth for the 2-s report only;
                      default 'none' = off -- the real car has no truth)
       <sim_wheel_ideal_topic> (SIMULATOR ONLY, geometry_msgs/TwistStamped from fake_vehicle_sensors; '' = off)
  out: <odom_topic>, <path_topic> (only if path_period_s > 0; for viewing),
       <diag_prefix>/bias, /bias_sigma, /speed_scale, /nis ;
       <diag_prefix>/map_check (global; std_msgs/Float64MultiArray, one per checked scan:
           [scan stamp s, shift m, gain, width m, fit0, beams, alarm 0/1, cost ms, every_n,
            state age ms, scan class (-1 no result, 0 neither, 1 clean, 2 bad along, 3 bad fit,
            4 both; v0.3)]; shift > 0 = the EKF is BEHIND, nan = too few returns) ;
       <diag_prefix>/slip (slip_check; std_msgs/Float64MultiArray, one per decided wheel sample:
           [wheel stamp s, slip 0/1, wheel speed k*v m/s, observer speed m/s, residual m/s, threshold m/s,
            accelerometer bias m/s^2, gated samples so far]) ;
       <diag_prefix>/imu_gap (imu_gap_predict; std_msgs/Float64MultiArray, with every odometry while the gyro is not
           trusted and once when it is again: [stamp s, state 0 OK / 1 GAP / 2 RECOVERING, s since the last trusted
           gyro, yaw rate used rad/s]) ;
       TF odom -> est/base_link (local only)

CPU (2026-10-03): in Python ROS 2 the cost is mostly MESSAGES, not math. The sim
truth arrives at 250 Hz, and the path message carries up to path_max_len poses;
both are off by default so nothing sim-only or view-only runs on the Pi.
state_pipeline.py runs both EKFs + state_estimate in ONE process: one IMU and one
wheel subscription for all three; odom_hooks / fix_hooks hand ekf_global's output
to state_estimate without a message.
"""
import math
import os
import threading
import time
from collections import deque

import numpy as np

CHI2_3DOF_99 = 11.34


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# ----------------------------------------------------------------------------
# The filter itself: plain numpy, no ROS -> can be tested offline
# ----------------------------------------------------------------------------

class LocalizationEKF:
    PX, PY, YAW, B, K = range(5)
    N = 5

    def __init__(self, x0, P0_diag, sigma_v, sigma_w, sigma_bw, sigma_kw):
        self.x = np.array(x0, dtype=float)
        self.P = np.diag(np.array(P0_diag, dtype=float) ** 2)
        self.sigma_v = sigma_v      # wheel speed noise              [m/s]
        self.sigma_w = sigma_w      # gyro white noise per sample    [rad/s]
        self.sigma_bw = sigma_bw    # gyro bias random walk          [rad/s/sqrt(s)]
        self.sigma_kw = sigma_kw    # speed-scale random walk        [1/sqrt(s)]

    # --- prediction ----------------------------------------------------------

    def predict(self, v_meas, w_meas, beta, dt, use_k=True, sigma_v=None, sigma_w=None):
        """use_k False (wheel slip, docstring UPDATE 3): v_meas is already a TRUE speed (the IMU observer), the
        wheel scale k takes no part (no k in the motion, no k column in F). sigma_v: speed noise override."""
        px, py, yaw, b, k = self.x
        if not use_k:
            k = 1.0
        v = k * v_meas
        heading = yaw + beta
        c, s = math.cos(heading), math.sin(heading)

        self.x[self.PX] = px + v * c * dt
        self.x[self.PY] = py + v * s * dt
        self.x[self.YAW] = wrap(yaw + (w_meas - b) * dt)

        F = np.eye(self.N)
        F[0, 2] = -v * s * dt
        F[1, 2] = v * c * dt
        if use_k:
            F[0, 4] = v_meas * c * dt
            F[1, 4] = v_meas * s * dt
        F[2, 3] = -dt
        G = np.zeros((self.N, 2))
        G[0, 0] = k * c * dt
        G[1, 0] = k * s * dt
        G[2, 1] = dt
        sv = self.sigma_v if sigma_v is None else sigma_v
        sw = self.sigma_w if sigma_w is None else sigma_w     # yaw-rate noise override (IMU dropout, UPDATE 4)
        Qu = np.diag([sv ** 2, sw ** 2])
        self.P = F @ self.P @ F.T + G @ Qu @ G.T
        self.P[self.B, self.B] += self.sigma_bw ** 2 * dt
        self.P[self.K, self.K] += self.sigma_kw ** 2 * dt

    def stationary_predict(self, dt):
        self.P[self.B, self.B] += self.sigma_bw ** 2 * dt
        self.P[self.K, self.K] += self.sigma_kw ** 2 * dt

    # --- generic EKF update --------------------------------------------------

    def _update(self, nu, H, R):
        """Standard EKF update. Returns (NIS, correction dx)."""
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        K = self.P @ H.T @ S_inv
        dx = K @ nu
        self.x = self.x + dx
        self.x[self.YAW] = wrap(self.x[self.YAW])
        I_KH = np.eye(self.N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        return float(nu @ S_inv @ nu), dx

    # --- update 1: zero velocity ---------------------------------------------

    def zupt(self, w_meas):
        H = np.zeros((1, self.N))
        H[0, self.B] = 1.0
        nu = np.array([w_meas - self.x[self.B]])
        R = np.array([[self.sigma_w ** 2]])
        self._update(nu, H, R)

    def standstill(self, w_meas, dt, gate):
        """One IMU step with the wheels standing still (2026-10-24, docstring UPDATE 1).
        Normally the true turn rate is 0, so the reading is bias + noise -> ZUPT learns b.
        NIS = (w_meas - b)^2 / (P_bb + sigma_w^2) > gate -> the car is being turned: no bias
        update, the heading follows the gyro (prediction with v = 0). gate <= 0: always ZUPT.
        Returns (turning, nis)."""
        nu = float(w_meas - self.x[self.B])
        nis = nu * nu / (float(self.P[self.B, self.B]) + self.sigma_w ** 2)
        if gate > 0.0 and nis > gate:
            self.predict(0.0, w_meas, 0.0, dt)
            return True, nis
        self.stationary_predict(dt)
        self.zupt(w_meas)
        return False, nis

    # --- update 2: pose (from AMCL) ------------------------------------------

    H_POSE = np.hstack([np.eye(3), np.zeros((3, 2))])

    def pose_nis(self, nu, R):
        S = self.H_POSE @ self.P @ self.H_POSE.T + R
        return float(nu @ np.linalg.solve(S, nu))

    def pose_update(self, nu, R):
        return self._update(nu, self.H_POSE, R)

    # --- helpers -------------------------------------------------------------

    def reset_pose(self, px, py, yaw, P3=None):
        """New pose; bias and speed scale are kept (they belong to the sensors)."""
        self.x[:3] = [px, py, yaw]
        self.P[:3, :] = 0.0
        self.P[:, :3] = 0.0
        if P3 is None:
            P3 = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(2.0) ** 2])
        self.P[:3, :3] = P3

    def sigma(self, i):
        return math.sqrt(max(self.P[i, i], 0.0))

    def sigma_pos(self):
        return math.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0))


class StandstillTurns:
    """The standstill gate's bookkeeping (docstring UPDATE 1), no ROS -> testable offline.
    step(ekf, w_meas, dt): one gyro sample with the wheels standing still; moving(): the car set off.
    Both return an event for the log, or None:
      ('start', rate_deg_s, nis)                a turn while standing began
      ('end', deg, why)                         it ended (standing still again / the car set off)
      ('bias_change', deg, secs, bias_deg_s)    it went on longer than max_turn_s -> it was the gyro bias
                                                changing, not a hand turn: heading change undone, bias re-learned
    """

    def __init__(self, gate=16.0, max_turn_s=3.0, still_after=20, sigma_bias0=0.05):
        self.gate, self.max_turn_s = float(gate), float(max_turn_s)
        self.still_after, self.sigma_bias0 = int(still_after), float(sigma_bias0)
        self.zupts = 0                      # samples used to learn the bias
        self.turn_samples = 0               # samples taken as a turn
        self.bias_changes = 0               # "turns" that were really a bias change (max_turn_s)
        self.deg = None                     # degrees turned in the episode going on (None = no episode)
        self.secs = 0.0                     # how long the episode has been going on
        self.calm = 0                       # bias-learning samples in a row since the last turn sample

    def step(self, ekf, w_meas, dt):
        turning, nis = ekf.standstill(w_meas, dt, self.gate)
        ev = None
        if turning:
            self.turn_samples += 1
            self.calm = 0
            rate = w_meas - float(ekf.x[ekf.B])           # what predict() just turned the heading by, per s
            if self.deg is None:
                self.deg, self.secs = 0.0, 0.0
                ev = ('start', math.degrees(rate), nis)
            self.deg += math.degrees(rate * dt)
        else:
            self.zupts += 1
            self.calm += 1
        if self.deg is None:
            return ev
        self.secs += dt
        if not turning and self.calm >= self.still_after:
            return self._end('standing still again')
        if self.max_turn_s > 0.0 and self.secs > self.max_turn_s:
            mean_rate = math.radians(self.deg) / self.secs
            ekf.x[ekf.YAW] = wrap(float(ekf.x[ekf.YAW]) - math.radians(self.deg))
            ekf.P[ekf.B, ekf.B] = max(float(ekf.P[ekf.B, ekf.B]), self.sigma_bias0 ** 2, mean_rate ** 2)
            self.bias_changes += 1
            ev = ('bias_change', self.deg, self.secs, math.degrees(float(ekf.x[ekf.B])))
            self.deg = None
        return ev

    def moving(self):
        return self._end('the car set off') if self.deg is not None else None

    def _end(self, why):
        ev = ('end', self.deg, why)
        self.deg = None
        return ev


class StateHistory:
    """Past states, to compare a delayed measurement with the right moment."""

    def __init__(self, max_age_s):
        self.max_age_ns = int(max_age_s * 1e9)
        self.buf = deque()                     # (t_ns, state copy)

    def add(self, t_ns, x):
        self.buf.append((t_ns, x.copy()))
        while self.buf and t_ns - self.buf[0][0] > self.max_age_ns:
            self.buf.popleft()

    def at(self, t_ns):
        """State closest in time to t_ns, or None if t_ns is outside the buffer."""
        if not self.buf or t_ns < self.buf[0][0] - 20_000_000:
            return None
        return min(self.buf, key=lambda e: abs(e[0] - t_ns))[1]

    def nearest(self, t_ns):
        """(t_ns, state) closest in time to t_ns, or None (same rule as at()). Searches from the
        newest entry backwards and stops once it moves away -- a scan is always recent, so this
        is a few steps instead of the whole buffer (per-scan cost on the Pi)."""
        if not self.buf or t_ns < self.buf[0][0] - 20_000_000:
            return None
        best = None
        for t, x in reversed(self.buf):
            dt = abs(t - t_ns)
            if best is None or dt < best[0]:
                best = (dt, t, x)
            elif t < t_ns:
                break
        return best[1], best[2]

    def shift(self, dx):
        """Apply a correction to the stored past too (it was wrong by the same amount)."""
        for _, x in self.buf:
            x += dx
            x[2] = wrap(x[2])


def amcl_measurement_noise(cov6x6, scale, min_sigma_pos, min_sigma_yaw):
    """3x3 R (x, y, yaw) from AMCL's 6x6 covariance, inflated and floored.
    The position floor acts on the ellipse's AXES (eigenvalues of the x-y block), so "never trust
    AMCL better than min_sigma_pos" holds in EVERY direction (2026-10-19; before: on the map's x / y
    diagonal only, which a long thin corridor ellipse at an angle to the map axes slips through --
    on the real car the map's axes are wherever SLAM started). Axis-aligned: identical to before."""
    c = np.asarray(cov6x6, dtype=float).reshape(6, 6)
    R = c[np.ix_([0, 1, 5], [0, 1, 5])] * scale
    R = 0.5 * (R + R.T)
    lam, V = np.linalg.eigh(R[:2, :2])
    R[:2, :2] = (V * np.maximum(lam, min_sigma_pos ** 2)) @ V.T
    R[2, 2] = max(R[2, 2], min_sigma_yaw ** 2)
    return R


def amcl_pose_class(cxx, cxy, cyy, max_sigma, corridor_sigma):
    """How the global EKF treats an AMCL pose, from its x-y covariance (docstring RECOVERY 1):
      'confident' sqrt(cxx + cyy) <= max_sigma: a normal fix; may count towards a jump
      'corridor'  wider, but the ellipse's NARROW axis sigma <= corridor_sigma (> 0): AMCL is sure
                  across a corridor, unsure along it -> a normal fix (its own covariance keeps the
                  long axis out of the update), never a jump candidate
      'unsure'    everything else: ignored"""
    if math.sqrt(max(cxx + cyy, 0.0)) <= max_sigma:
        return 'confident'
    if corridor_sigma > 0.0:
        half = 0.5 * (cxx + cyy)                          # smallest eigenvalue of [[cxx, cxy], [cxy, cyy]]
        rad = math.sqrt(0.25 * (cxx - cyy) ** 2 + cxy ** 2)
        if math.sqrt(max(half - rad, 0.0)) <= corridor_sigma:
            return 'corridor'
    return 'unsure'


# ----------------------------------------------------------------------------
# Jump-to-AMCL check: plain numpy, no ROS -> tested offline (see RECOVERY)
# ----------------------------------------------------------------------------

class WallMap:
    """The walls of the map AMCL uses, for ray casting. ROS cell convention (map_server,
    RViz, the simulator): cell (row, col) spans origin + [col, col + 1) * res in x and
    origin + [row, row + 1) * res in y; row 0 = the BOTTOM row (OccupancyGrid order).
    Unknown cells do not stop a ray (like the simulator)."""

    def __init__(self, occupied, res, ox, oy):
        self.occ = np.asarray(occupied, dtype=bool)
        self.h, self.w = self.occ.shape
        self.res, self.ox, self.oy = float(res), float(ox), float(oy)

    @classmethod
    def from_grid(cls, data, width, height, res, ox, oy, occupied_from=65):
        """OccupancyGrid data: 0 free, 100 wall, -1 unknown."""
        grid = np.asarray(data, dtype=np.int16).reshape(height, width)
        return cls(grid >= occupied_from, res, ox, oy)

    def ranges(self, lx, ly, beam_yaws, max_range):
        """Distance from (lx, ly) to the first wall cell along each beam (half-cell
        steps); inf = no wall within max_range."""
        ts = np.arange(0.5 * self.res, max_range + 1e-9, 0.5 * self.res)
        xs = lx + np.outer(np.cos(beam_yaws), ts)
        ys = ly + np.outer(np.sin(beam_yaws), ts)
        col = np.floor((xs - self.ox) / self.res).astype(np.int64)
        row = np.floor((ys - self.oy) / self.res).astype(np.int64)
        inside = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        hit = np.zeros(xs.shape, dtype=bool)
        hit[inside] = self.occ[row[inside], col[inside]]
        return np.where(hit.any(axis=1), ts[hit.argmax(axis=1)], np.inf)


def scan_fit(wall_map, pose, laser, beam_angles, r_meas, max_range, tol):
    """Share of the beams the map explains if the car is at `pose` (x, y, yaw of
    base_link): measured and map range within `tol`, or both 'no return' (inf).
    laser = (x, y, yaw) of the LiDAR on the car. 1.0 = the scan fits perfectly."""
    x, y, yaw = pose[0], pose[1], pose[2]
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = x + c * laser[0] - s * laser[1], y + s * laser[0] + c * laser[1]
    r_exp = wall_map.ranges(lx, ly, yaw + laser[2] + np.asarray(beam_angles), max_range)
    meas_far, exp_far = ~np.isfinite(r_meas), ~np.isfinite(r_exp)
    both = ~meas_far & ~exp_far
    close = np.zeros(len(r_exp), dtype=bool)
    close[both] = np.abs(r_meas[both] - r_exp[both]) < tol
    return float(np.mean(close | (meas_far & exp_far)))


def scan_beams(ranges, angle_min, angle_increment, range_min, range_max, n_beams):
    """n_beams evenly spread beams of a LaserScan: (measured ranges, inf = no valid
    return; their angles in the LiDAR frame)."""
    r_all = np.asarray(ranges, dtype=float)
    idx = np.unique(np.linspace(0, len(r_all) - 1, n_beams).round().astype(np.int64))
    r = r_all[idx]
    r = np.where(np.isfinite(r) & (r >= range_min) & (r <= range_max), r, np.inf)
    return r, angle_min + idx * angle_increment


def body_motion(a, b):
    """Motion from pose a to pose b in a's frame: (forward, left, turn)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])


def motion_consistent(streak, tol_pos, tol_frac, tol_yaw):
    """streak = [(t_ns, AMCL pose, EKF pose at that time), ...] of rejected AMCL poses
    (the EKF accepted nothing meanwhile, so its motion = odometry). Did AMCL move the way
    the odometry says the car moved? From the first pose to every later one, each in its
    own frame, so a constant offset between the two does not matter.
    -> (ok, worst position excess over the tolerance [m], worst turn difference [rad])"""
    z0, x0 = streak[0][1], streak[0][2]
    ok, worst_pos, worst_yaw = True, -math.inf, 0.0
    for _, z, x in streak[1:]:
        a, e = body_motion(z0, z), body_motion(x0, x)
        excess = math.hypot(a[0] - e[0], a[1] - e[1]) - (tol_pos + tol_frac * math.hypot(e[0], e[1]))
        dyaw = abs(wrap(a[2] - e[2]))
        worst_pos, worst_yaw = max(worst_pos, excess), max(worst_yaw, dyaw)
        ok = ok and excess <= 0.0 and dyaw <= tol_yaw
    return ok, worst_pos, worst_yaw


# ----------------------------------------------------------------------------
# ROS 2 node
# ----------------------------------------------------------------------------

try:
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped, TwistStamped
    from nav_msgs.msg import OccupancyGrid, Odometry, Path
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import Imu, LaserScan
    from std_msgs.msg import Float64, Float64MultiArray
    from tf2_ros import TransformBroadcaster

    from state_estimation.calibration import load_steering
    from state_estimation.map_check import (BAD_ALONG, BAD_FIT, CpuBudget, DistanceField, MismatchAlarm, make_shifts,
                                            shift_search)
    from state_estimation.slip_check import SimGate, SlipCheck
    from state_estimation.imu_gap import GAP as IMU_GAP, ImuGap, STATE_NAMES as GAP_NAMES
    from state_estimation.vehicle_model import sideslip_from_params
    from state_estimation.wheel_delay import WheelSpeedCompensator
except ImportError:
    Node = object


def quat_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class EkfNode(Node):

    def __init__(self, node_name='ekf_node', own_inputs=True):
        # node_name / own_inputs=False: used by state_pipeline.py (both EKFs + state_estimate
        # in ONE process, the IMU and wheel messages arrive through ONE shared subscription)
        super().__init__(node_name)
        d = self.declare_parameter
        d('mode', 'map')                    # 'map' = global EKF, 'odom' = local EKF
        d('frame_id', 'map')                # frame of the estimated pose
        d('child_frame_id', 'est/base_link')
        d('publish_tf', False)              # True for the local EKF: odom -> est/base_link
        d('odom_topic', '/odom/ekf')
        d('path_topic', '/odom/ekf_path')
        d('diag_prefix', '/ekf')
        d('path_max_len', 1500)             # poses kept in the path (5 per s -> 5 min)
        d('path_period_s', 0.0)             # s between path messages; 0 = no path (default)
        d('truth_topic', 'none')            # sim truth for the report; 'none' = off (real car)
        d('diag_every', 10)                 # publish bias/scale every Nth IMU step (10 Hz)
        # Start pose and its uncertainty
        d('x0', 0.0); d('y0', 0.0); d('yaw0', 0.0)
        d('sigma_pos0', 0.1)                # m
        d('sigma_yaw0_deg', 2.0)            # deg
        d('sigma_bias0', 0.05)              # rad/s
        d('sigma_scale0', 0.05)             # 5 %
        # Process noise
        d('sigma_v', 0.05)
        d('sigma_w', 0.005)
        d('sigma_bw', 0.0001)
        d('sigma_kw', 0.0005)
        # AMCL update (global EKF)
        d('use_amcl', True)
        d('amcl_topic', '/amcl_pose')
        d('amcl_cov_scale', 4.0)            # inflate AMCL covariance (shared odometry; 2 -> 4 2026-10-06)
        d('amcl_min_sigma_pos', 0.10)       # m (0.10 chosen 2026-10-01, was 0.05)
        d('amcl_min_sigma_yaw_deg', 1.0)    # deg
        d('amcl_offset_m', 0.025)           # added to AMCL x and y: nav2 half-cell offset = map resolution / 2
        d('map_topic', '/map')              # read once, only to check amcl_offset_m ('' = no check)
        d('gate', CHI2_3DOF_99)
        d('reset_after_rejects', 10)        # confident AMCL poses rejected in a row before a jump is considered
        # Robust jump to AMCL (2026-10-06, docstring RECOVERY)
        d('amcl_max_sigma', 0.35)           # m: AMCL poses with sqrt(cov_xx + cov_yy) above this are ignored ...
        d('amcl_corridor_sigma', 0.10)      # m ... unless the ellipse's NARROW axis sigma <= this (a corridor
                                            # pose: a fix, never a jump; 2026-10-19, docstring RECOVERY 1); 0 = off
        d('reset_tol_pos', 0.15)            # m   AMCL motion vs odometry over the rejected poses ...
        d('reset_tol_frac', 0.05)           #     ... + 5 % of the distance
        d('reset_tol_yaw_deg', 3.0)         # deg
        d('reset_scan_check', True)         # also require the scan to fit the map clearly better at AMCL's pose
        d('scan_topic', '/scan_amcl')       # the scan AMCL uses
        d('scan_beams', 60)                 # beams used by the scan check
        d('scan_tol', 0.10)                 # m: a beam fits if measured and map range differ less
        d('reset_scan_min', 0.6)            # AMCL's pose must explain >= 60 % of the beams ...
        d('reset_scan_margin', 0.1)         # ... and 10 points more than the EKF's pose
        d('laser_x', 0.275)                 # m   where the LiDAR sits on the car (base_link -> laser;
        d('laser_y', 0.0)                   # m   sim: scan_distance_to_base_link)
        d('laser_yaw', 0.0)                 # rad
        # Map check: along-track position vs the LiDAR (2026-10-21, docstring MAP CHECK; its alarm ->
        # state_estimate FLAG_MAP_MISMATCH + DEGRADED since contract v0.3, 2026-10-24)
        d('map_check', True)                # global EKF: run it on every scan (needs map_topic + scan_topic)
        d('map_check_beams', 180)           # beams per scan (evenly spread)
        d('map_check_max_shift', 1.0)       # m: search +- this along the heading ...
        d('map_check_step', 0.05)           # m  ... in these steps (41 shifts)
        d('map_check_tol', 0.10)            # m: a beam fits if its end point lies this close to a wall
        d('map_check_shift_min', 0.30)      # m: alarm only for errors at least this big ...
        d('map_check_gain_min', 0.05)       #    ... if the best shift fits >= 5 points better than 0 ...
        d('map_check_width_max', 0.30)      # m  ... is one sharp optimum (not a flat corridor) ...
        d('map_check_k', 2)                 #    ... in this many checked scans in a row
        d('map_check_budget_ms', 2.0)       # ms per scan on average; above -> check only every n-th scan
        d('map_check_max_every', 7)         #    n <= this (7 = once per second at 7 Hz)
        d('map_check_fit_bad', 0.5)         # BAD also when the scan fits the map worse than this at the EKF pose
        d('map_check_gain_clear', 0.02)     # CLEAN: no shift fits more than this better than the pose ...
        d('map_check_fit_clear', 0.90)      #        ... and the scan fits at least this well at the pose
        d('map_check_k_clear', 4)           # alarm OFF after this many CLEAN scans in a row ...
        d('map_check_min_on_s', 0.5)        #           ... and at least this long on (s, scan time)
        # TEST ONLY switches (simulator fault scenarios) work only with allow_test_faults true: read-only, set
        # at start-up by the SIMULATOR launch files; false (the default) on the real car (2026-10-24, docstring)
        from rcl_interfaces.msg import ParameterDescriptor
        d('allow_test_faults', False, ParameterDescriptor(
            read_only=True, description='simulator only: unlock test_fault / test_kick / test_lag_ms'))
        self.allow_test_faults = bool(self.get_parameter('allow_test_faults').value)
        d('test_fault', '')                 # TEST ONLY: 't_s,dx,dy,dyaw_deg' moves ekf_global's pose once
        if self.allow_test_faults:
            d('test_kick', '')              # TEST ONLY, runtime: 'dx,dy,dyaw_deg' moves the pose at the next IMU
        d('history_s', 2.0)
        # Vehicle geometry for the slip angle
        d('lf', 0.15875); d('lr', 0.17145)
        # Sideslip model (2026-10-18, docstring SIDESLIP MODEL, vehicle_model.py)
        d('beta_model', 'slip')             # 'slip' (gyro + rear-tyre slip, default) | 'steer' (kinematic, steering)
        d('slip_k', -1.0)                   # s^2/m; < 0 = read slip_file (no file -> 0)
        d('slip_file', '~/.ros/vehicle_slip.yaml')  # written by slip_calib (READ only here)
        # Wheel-speed delay compensation (2026-10-19, docstring WHEEL-SPEED DELAY COMPENSATION)
        d('wheel_delay_s', -1.0)            # s: the wheel speed's own delay; < 0 = compensation OFF
        d('imu_yaw_deg', 0.0)               # deg: IMU x axis vs the car's forward axis (CCW); 0 = x forward
        # Zero-velocity detection
        d('stopped_speed', 0.01)
        d('stopped_time', 0.3)
        d('zupt_gate', 16.0)                # standstill: NIS of gyro vs bias above this = turned by hand (docstring
                                            # UPDATE 1); 0 = old rule (always learn the bias)
        d('zupt_turn_max_s', 3.0)           # a standstill "turn" longer than this = the gyro bias changed (docstring
                                            # UPDATE 1); 0 = no limit
        d('initialpose_topic', '/initialpose')  # global EKF: where a start pose comes from (sim tests: /initialpose_est)
        # Wheel-slip check (2026-10-26, Fix 3, docstring UPDATE 3, slip_check.py)
        d('slip_check', True)               # detect wheel slip (wheel vs IMU forward acceleration)
        d('slip_use_observer', True)        # while slipping predict with the IMU speed (false = flag only)
        d('slip_sigma_v', 0.15)             # m/s: speed noise while predicting with the IMU speed
        d('slip_lag_s', 0.03)               # s: each wheel sample is compared with the IMU speed this much later
        d('slip_tau_s', 0.5)                # s: pull of the IMU speed towards the trusted wheel
        d('slip_r0', 0.20)                  # m/s: ON threshold ...
        d('slip_c_a', 0.03)                 # s    ... + this * |acceleration|
        d('slip_k_on', 2)                   # ... in this many wheel samples in a row
        d('slip_r_off', 0.10)               # m/s: OFF threshold ...
        d('slip_r_off_rate', 0.05)          # m/s per s of slip ... growing (max 0.9 * slip_r0) ...
        d('slip_k_off', 5)                  # ... in this many samples in a row ...
        d('slip_min_on_s', 0.2)             # ... and on at least this long
        d('slip_max_on_s', 3.0)             # s: longer -> give up, re-sync to the wheel
        d('slip_imu_gap_s', 0.05)           # s: IMU gap -> re-sync to the wheel
        d('slip_ki_bias', 0.3)              # 1/s^2: accelerometer-bias learning while driving (0 = standstill only)
        d('sim_wheel_ideal_topic', '')      # SIMULATOR ONLY: /sim/wheel_ideal (artifact gate); '' on the real car
        # IMU dropout (2026-10-27, Fix 2, docstring UPDATE 4, imu_gap.py). Default OFF = the old behaviour; the simulator
        # launch files switch it on (control: Option A for development + synthetic evaluation; stamp amendment pending)
        d('imu_gap_predict', False)         # predict from wheel + steering while the IMU is silent
        d('imu_gap_s', 0.025)               # s without an IMU sample = dropout (2.5 periods at 100 Hz)
        d('imu_gap_kus', 0.0)               # s^2/m understeer of the yaw-rate model (simulator fit ~0)
        d('imu_gap_tau_b', 0.2)             # s: the last gyro's offset to the model fades with this
        d('imu_gap_sigma_w', 0.4)           # rad/s: yaw-rate noise per prediction step during the dropout
        d('imu_gap_rec_n', 10)              # returning gyro trusted after this many samples in a row ...
        d('imu_gap_rec_tol', 0.5)           # rad/s ... that agree with the wheel model within this
        # Steering calibration (steer_calib); 'none' = use the measured steering as it is
        d('steer_calib_file', '~/.ros/steering_calibration.yaml')
        # Speed-scale calibration file (see docstring)
        d('scale_file', 'none')             # anchor (READ only); 'none' = start with k = 1
        d('scale_learned_file', '~/.ros/ekf_speed_scale_learned.yaml')  # global EKF writes
        d('scale_warn', 0.01)               # warn if learned k differs > 1 % from the anchor
        d('scale_file_min_sigma', 0.01)     # never trust the file more than 1 %
        d('scale_save_min_distance', 20.0)  # m driven before the global EKF saves


        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mode = p('mode')
        self.global_mode = self.mode == 'map'
        sf = str(p('scale_file')).strip()
        self.scale_file = os.path.expanduser(sf) if sf and sf.lower() != 'none' else ''
        lf_ = str(p('scale_learned_file')).strip()
        self.learned_file = os.path.expanduser(lf_) if lf_ and lf_.lower() != 'none' else ''
        k0, sigma_k0 = self.load_scale(1.0, p('sigma_scale0'))
        self.k_anchor = k0 if self.scale_file and k0 != 1.0 else None
        self.warned_scale = False
        self.steer_cal, problem = load_steering(p('steer_calib_file'))
        if problem:
            self.get_logger().warn(problem)
        else:
            self.get_logger().info(self.steer_cal.describe())
        self.sideslip, slip_msg, slip_warn = sideslip_from_params(
            str(p('beta_model')).strip().lower(), p('lf'), p('lr'), p('slip_k'), p('slip_file'))
        (self.get_logger().warn if slip_warn else self.get_logger().info)(slip_msg)
        tau_w = float(p('wheel_delay_s'))
        self.wheel_comp = WheelSpeedCompensator(tau_w) if tau_w >= 0.0 else None
        self.imu_c, self.imu_s = math.cos(math.radians(p('imu_yaw_deg'))), math.sin(math.radians(p('imu_yaw_deg')))
        self.get_logger().info(self.wheel_comp.describe() if self.wheel_comp is not None
                               else 'wheel-speed delay compensation OFF (wheel_delay_s < 0)')
        self.ekf = LocalizationEKF(
            x0=[p('x0'), p('y0'), p('yaw0'), 0.0, k0],
            P0_diag=[p('sigma_pos0'), p('sigma_pos0'), math.radians(p('sigma_yaw0_deg')),
                     p('sigma_bias0'), sigma_k0],
            sigma_v=p('sigma_v'), sigma_w=p('sigma_w'),
            sigma_bw=p('sigma_bw'), sigma_kw=p('sigma_kw'))
        self.history = StateHistory(p('history_s'))

        self.v = 0.0
        self.v_used = 0.0                   # the speed the EKF predicts with (v + delay compensation)
        self.speed_used = 0.0               # ... in TRUE m/s (k * v_used, or the IMU speed while slipping)
        self.t_state = None                 # time the state is valid at (IMU samples; wheel samples in a dropout)
        self.steer = 0.0
        self.stopped_since = None
        self.last_imu_t = None
        self.still = StandstillTurns(gate=p('zupt_gate'), max_turn_s=p('zupt_turn_max_s'),
                                     sigma_bias0=p('sigma_bias0'))      # standstill gate (docstring UPDATE 1)
        # wheel-slip check (docstring UPDATE 3)
        self.slip = None
        self.slip_gate = None
        self.slip_use = bool(p('slip_use_observer'))
        self.slip_sigma_v = float(p('slip_sigma_v'))
        self.slip_hooks = []                # state_pipeline: ekf_global -> state_estimate.slip_cb
        ideal_topic = str(p('sim_wheel_ideal_topic')).strip()
        if p('slip_check'):
            self.slip_gate = SimGate() if ideal_topic else None
            self.slip = SlipCheck(lag=p('slip_lag_s'), tau=p('slip_tau_s'), r0=p('slip_r0'), c_a=p('slip_c_a'),
                                  r_off=p('slip_r_off'), k_on=p('slip_k_on'), k_off=p('slip_k_off'),
                                  min_on=p('slip_min_on_s'), max_on=p('slip_max_on_s'), gap=p('slip_imu_gap_s'),
                                  sim_gate=self.slip_gate, ki_bias=p('slip_ki_bias'),
                                  r_off_rate=p('slip_r_off_rate'))
        # IMU dropout (docstring UPDATE 4)
        self.gap = None
        self.gap_hooks = []                 # state_pipeline: ekf_global -> state_estimate.imu_gap_cb
        self.gap_sigma_w = float(p('imu_gap_sigma_w'))
        self.gap_predictions = 0
        self.gap_last_pub = 0
        if p('imu_gap_predict'):
            self.gap = ImuGap(gap_s=p('imu_gap_s'), wheelbase=p('lf') + p('lr'), kus=p('imu_gap_kus'),
                              tau_b=p('imu_gap_tau_b'), rec_n=p('imu_gap_rec_n'), rec_tol=p('imu_gap_rec_tol'))
        self.distance = 0.0
        self.truth = None
        self.truth_hist = deque(maxlen=400)
        self.stats = self.new_stats()
        self.streak = []                    # rejected confident AMCL poses in a row: (t_ns, z, x_then)
        self.imu_count = 0
        self.map_checked = False
        self.wall_map = None                # walls of the map, for the scan check
        self.dfield = None                  # distance field of the same map, for the map check
        self.map_check_on = bool(self.global_mode and p('use_amcl') and p('map_check'))
        self.mc_shifts = make_shifts(p('map_check_max_shift'), p('map_check_step'))
        self.mc_monitor = MismatchAlarm(
            p('map_check_shift_min'), p('map_check_max_shift') - p('map_check_step') + 1e-9, p('map_check_gain_min'),
            p('map_check_width_max'), p('map_check_k'), p('map_check_fit_bad'), p('map_check_gain_clear'),
            p('map_check_fit_clear'), p('map_check_k_clear'), p('map_check_min_on_s'))
        self.mc_budget = CpuBudget(p('map_check_budget_ms'), p('map_check_max_every'))
        self.mc_stats = self.new_mc_stats()
        self.scans = deque(maxlen=40)       # recent scans: (t_ns, msg)
        self.last_refuse_log = -1e9
        self.fault = None                   # test_fault (global EKF only)
        self.fault_t0 = None
        tf_ = str(p('test_fault')).strip()
        if tf_ and self.global_mode and not self.allow_test_faults:
            self.get_logger().error(f"test_fault '{tf_}' IGNORED: the test switches are locked "
                                    '(allow_test_faults false -- only the simulator launch files set it)')
        elif tf_ and self.global_mode:
            vals = [float(v) for v in tf_.split(',')]
            if len(vals) != 4:
                raise ValueError("test_fault must be 't_s,dx,dy,dyaw_deg'")
            self.fault = vals

        self.path = Path()
        self.path.header.frame_id = p('frame_id')
        self.tf_pub = TransformBroadcaster(self) if p('publish_tf') else None

        # In-process consumers (state_pipeline): called with every published odometry, with the NIS
        # message of every ACCEPTED AMCL fix and (v0.3) with every map-check result -- the same data
        # the topics carry.
        self.odom_hooks = []
        self.fix_hooks = []
        self.map_check_hooks = []
        if own_inputs:
            self.create_subscription(Imu, '/imu/data', self.imu_cb, 100)
            self.create_subscription(AckermannDriveStamped, '/vehicle/measured', self.meas_cb, 50)
        if self.global_mode:
            self.create_subscription(PoseWithCovarianceStamped, p('amcl_topic'), self.amcl_cb, 10)
            self.create_subscription(PoseWithCovarianceStamped, p('initialpose_topic'), self.reset_cb, 10)
            if p('use_amcl') and str(p('map_topic')).strip():
                # map_server publishes the map once, "latched" (transient local) -> late joiners get it
                self.create_subscription(OccupancyGrid, p('map_topic'), self.map_cb, QoSProfile(
                    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    reliability=ReliabilityPolicy.RELIABLE))
            if p('use_amcl') and (p('reset_scan_check') or p('map_check')):
                self.create_subscription(LaserScan, p('scan_topic'), self.scan_cb, qos_profile_sensor_data)
            tt = str(p('truth_topic')).strip()
            if tt and tt.lower() != 'none':
                self.create_subscription(Odometry, tt, self.truth_cb, 10)
        self.odom_pub = self.create_publisher(Odometry, p('odom_topic'), 10)
        self.path_pub = self.create_publisher(Path, p('path_topic'), 10)
        pre = p('diag_prefix')
        self.bias_pub = self.create_publisher(Float64, pre + '/bias', 10)
        self.bias_sigma_pub = self.create_publisher(Float64, pre + '/bias_sigma', 10)
        self.scale_pub = self.create_publisher(Float64, pre + '/speed_scale', 10)
        self.nis_pub = self.create_publisher(Float64, pre + '/nis', 10)
        self.mc_pub = (self.create_publisher(Float64MultiArray, pre + '/map_check', 10)
                       if self.map_check_on else None)
        self.slip_pub = (self.create_publisher(Float64MultiArray, pre + '/slip', 10)
                         if self.slip is not None else None)
        self.gap_pub = (self.create_publisher(Float64MultiArray, pre + '/imu_gap', 10)
                        if self.gap is not None else None)
        if self.gap is not None:
            self.get_logger().info(
                f"IMU dropout prediction ON -> {pre}/imu_gap: no IMU for > {1000 * p('imu_gap_s'):.0f} ms -> predict at "
                f"every wheel sample (yaw rate v tan(steer) / (L + {p('imu_gap_kus'):g} v^2) + fading last-gyro offset, "
                f"tau {p('imu_gap_tau_b'):g} s, sigma {self.gap_sigma_w:g} rad/s); returning gyro trusted after "
                f"{p('imu_gap_rec_n')} samples within {p('imu_gap_rec_tol'):g} rad/s of the model")
        else:
            self.get_logger().info('IMU dropout prediction OFF (imu_gap_predict false): the estimate stands still '
                                   'while the IMU is silent (the old behaviour)')
        if self.slip_gate is not None:      # SIMULATOR ONLY (docstring UPDATE 3)
            self.create_subscription(TwistStamped, ideal_topic, self.wheel_ideal_cb, 50)
        if self.slip is not None:
            self.get_logger().info(
                f"wheel-slip check -> {pre}/slip" + (' -> state_estimate FLAG_WHEEL_SLIP' if self.global_mode else '')
                + f": ON when |k*wheel - IMU speed| > {p('slip_r0'):.2f} m/s + {p('slip_c_a'):.2f} s * |a| for "
                f"{p('slip_k_on')} samples (compared {1000 * p('slip_lag_s'):.0f} ms later); while ON "
                + (f"predict with the IMU speed (sigma_v {self.slip_sigma_v:.2f} m/s)" if self.slip_use
                   else 'FLAG ONLY (slip_use_observer false)')
                + (f"; SIMULATOR artifact gate on {ideal_topic}" if self.slip_gate is not None else
                   '; no simulator gate'))
        else:
            self.get_logger().warn('wheel-slip check OFF (slip_check false)')
        if p('path_period_s') > 0.0:
            self.create_timer(p('path_period_s'), self.publish_path)
        self.create_timer(2.0, self.report)
        if self.global_mode and self.learned_file:
            self.create_timer(10.0, self.save_scale)
        what = ('GLOBAL (map frame, wheel + gyro + AMCL)' if self.global_mode
                else f'LOCAL (odom frame, wheel + gyro), TF {p("frame_id")} -> {p("child_frame_id")}')
        self.get_logger().info(f'EKF v3 {what}')
        if self.global_mode and p('use_amcl'):
            self.get_logger().info(
                f"AMCL half-cell correction: +{100.0 * p('amcl_offset_m'):.2f} cm in x and y "
                '(amcl_offset_m = map resolution / 2; 0 = off)')
            self.get_logger().info(
                f"jump to AMCL only if it is clearly right: ignore AMCL sigma > {p('amcl_max_sigma'):.2f} m"
                + (f" unless a corridor pose (narrow axis sigma <= {p('amcl_corridor_sigma'):.2f} m: a fix, never a jump); "
                   if p('amcl_corridor_sigma') > 0.0 else ' (corridor poses OFF); ')
                + f"after {p('reset_after_rejects')} rejected poses AMCL must move like the odometry"
                + (f" AND the scan ({p('scan_topic')}) must fit AMCL's pose >= {p('reset_scan_min'):.2f} and "
                   f"{p('reset_scan_margin'):.2f} better than the EKF's" if p('reset_scan_check') else
                   ' (scan check OFF)'))
        if self.map_check_on:
            self.get_logger().info(
                f"map check -> {pre}/map_check -> state_estimate FLAG_MAP_MISMATCH (contract v0.3): "
                f"+-{p('map_check_max_shift'):.2f} m in {100 * p('map_check_step'):.0f} cm steps, {p('map_check_beams')} "
                f"beams; BAD scan = |shift| >= {p('map_check_shift_min'):.2f} m with gain >= {p('map_check_gain_min'):.2f} "
                f"and width <= {p('map_check_width_max'):.2f} m, or fit < {p('map_check_fit_bad'):.2f}; ON after "
                f"{p('map_check_k')} BAD scans in a row; OFF after {p('map_check_k_clear')} CLEAN scans in a row (gain <= "
                f"{p('map_check_gain_clear'):.2f}, fit >= {p('map_check_fit_clear'):.2f}) and >= "
                f"{p('map_check_min_on_s'):.1f} s on; CPU budget {p('map_check_budget_ms'):.1f} ms per scan")
        if self.fault is not None:
            self.get_logger().warn(f'TEST FAULT armed: ekf_global pose will move by ({self.fault[1]}, '
                                   f'{self.fault[2]}) m, {self.fault[3]} deg, {self.fault[0]} s after the first IMU')
        if self.allow_test_faults:
            self.add_on_set_parameters_callback(self._on_set_test_kick)
            self.get_logger().warn('TEST SWITCHES UNLOCKED (allow_test_faults true): test_fault / test_kick'
                                   + (' / test_lag_ms' if self.global_mode else '') + ' -- simulator only')

    def _on_set_test_kick(self, params):
        """TEST ONLY (test_kick): arm a pose kick at runtime; it is applied at the next IMU message."""
        from rcl_interfaces.msg import SetParametersResult
        for prm in params:
            if prm.name != 'test_kick' or not str(prm.value).strip():
                continue
            try:
                dx, dy, dyaw = [float(v) for v in str(prm.value).split(',')]
            except ValueError:
                return SetParametersResult(successful=False, reason="test_kick must be 'dx,dy,dyaw_deg'")
            self.fault, self.fault_t0 = [0.0, dx, dy, dyaw], None
            self.get_logger().warn(f'TEST KICK armed: pose moves by ({dx}, {dy}) m, {dyaw} deg at the next IMU message')
        return SetParametersResult(successful=True)

    # --- speed-scale calibration file ----------------------------------------

    def load_scale(self, k_default, sigma_default):
        """(k0, sigma_k0) from the calibration file, or the defaults."""
        if not self.scale_file:
            return k_default, sigma_default
        try:
            vals = {}
            with open(self.scale_file, encoding='utf-8') as f:
                for line in f:
                    if ':' in line and not line.lstrip().startswith('#'):
                        key, val = line.split(':', 1)
                        vals[key.strip()] = val.strip()
            k = float(vals['speed_scale'])
            sig = max(float(vals.get('speed_scale_sigma', sigma_default)),
                      self.get_parameter('scale_file_min_sigma').value)
            if not 0.8 < k < 1.2:
                raise ValueError(f'speed_scale {k} outside 0.8..1.2')
        except FileNotFoundError:
            self.get_logger().warn(f'No calibration file {self.scale_file} -> starting with '
                                   f'k = {k_default}. Create it with wheel_calib.')
            return k_default, sigma_default
        except (KeyError, ValueError) as e:
            self.get_logger().error(f'Bad calibration file {self.scale_file} ({e}) -> k = {k_default}')
            return k_default, sigma_default
        self.get_logger().info(f'Speed scale from {self.scale_file}: k0 = {k:.4f} (+-{sig:.3f})')
        return k, sig

    def save_scale(self):
        """Global EKF only: write the LEARNED k to its own file (never the anchor)."""
        if not (self.global_mode and self.learned_file):
            return
        if self.distance < self.get_parameter('scale_save_min_distance').value:
            return
        k, sig = float(self.ekf.x[4]), self.ekf.sigma(4)
        if self.k_anchor is not None and not self.warned_scale:
            diff = (k - self.k_anchor) / self.k_anchor
            if abs(diff) > self.get_parameter('scale_warn').value:
                self.get_logger().warn(
                    f'Learned speed scale {k:.4f} differs {100 * diff:+.1f} % from the '
                    f'calibration {self.k_anchor:.4f} -> tyres changed/worn? Redo wheel_calib.')
                self.warned_scale = True
        tmp = self.learned_file + '.tmp'
        try:
            os.makedirs(os.path.dirname(self.learned_file) or '.', exist_ok=True)
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write('# Wheel-speed scale learned by ekf_global (v_true = k * v_measured)\n')
                f.write(f'speed_scale: {k:.6f}\n')
                f.write(f'speed_scale_sigma: {sig:.6f}\n')
                f.write(f'distance_m: {self.distance:.1f}\n')
                if self.k_anchor is not None:
                    f.write(f'anchor_speed_scale: {self.k_anchor:.6f}\n')
                f.write(f"saved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            os.replace(tmp, self.learned_file)
        except OSError as e:
            self.get_logger().error(f'Could not save {self.learned_file}: {e}')

    # --- inputs --------------------------------------------------------------

    def meas_cb(self, msg):
        self.v = msg.drive.speed
        self.steer = self.steer_cal.correct(msg.drive.steering_angle)
        now = Time.from_msg(msg.header.stamp)
        if self.slip is not None:           # wheel in TRUE m/s (docstring UPDATE 3)
            for row in self.slip.wheel(now.nanoseconds * 1e-9, float(self.ekf.x[4]) * self.v):
                self.publish_slip(row)
            self.log_slip()
        if self.wheel_comp is not None:
            self.wheel_comp.on_wheel(now.nanoseconds * 1e-9)
        if abs(self.v) < self.get_parameter('stopped_speed').value:
            if self.stopped_since is None:
                self.stopped_since = now
        else:
            self.stopped_since = None
        if self.gap is not None:            # IMU dropout: predict at the wheel rate (docstring UPDATE 4)
            st = self.gap.wheel(now.nanoseconds * 1e-9, float(self.ekf.x[4]) * self.v, self.steer,
                                rx=self.get_clock().now().nanoseconds * 1e-9)
            self.log_gap()
            if st == IMU_GAP and self.t_state is not None:
                self.gap_predict(now, msg.header.stamp)

    def gap_predict(self, now, stamp):
        """One prediction to the wheel sample's stamp from wheel speed + steering (IMU silent, docstring UPDATE 4)."""
        dt = (now - self.t_state).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self.t_state = now
        if dt > 0.5:                        # wheel AND IMU silent for > 0.5 s: nothing to predict with
            return
        b, k = float(self.ekf.x[3]), float(self.ekf.x[4])
        r = self.gap.yaw_rate(now.nanoseconds * 1e-9)
        self.v_used = self.v
        self.speed_used = k * self.v
        if self.is_stopped(now):
            self.ekf.stationary_predict(dt)
        else:
            beta = self.sideslip.beta(self.speed_used, r, self.steer)
            self.ekf.predict(self.v, r + b, beta, dt, sigma_w=self.gap_sigma_w)
            self.distance += abs(self.speed_used) * dt
        self.gap_predictions += 1
        self.history.add(now.nanoseconds, self.ekf.x)
        self.publish_odom(stamp, r + b)

    def log_gap(self):
        """One log line per dropout event (docstring UPDATE 4)."""
        lg = self.get_logger()
        for kind, t_s, a, g_s in self.gap.pop_events():
            if kind == 'gap':
                lg.warn(f'IMU DROPOUT: no trusted IMU sample for {1000 * (t_s - a):.0f} ms -> predicting from wheel + '
                        f'steering (yaw rate {self.gap.yaw_rate(t_s):+.2f} rad/s, last gyro {self.gap.r_trusted:+.2f})')
            elif kind == 'back':
                lg.info(f'IMU back after {a:.3f} s -> checking the gyro against the wheel model before trusting it')
            elif kind == 'valid':
                lg.info(f'IMU trusted again: {1000 * a:.0f} ms of consistent gyro ({g_s:.3f} s without a trusted gyro)')
            else:
                lg.warn(f'returning gyro disagrees with the wheel model by {a:+.2f} rad/s -> not trusted yet')

    def is_stopped(self, now):
        if self.stopped_since is None:
            return False
        waited = (now - self.stopped_since).nanoseconds * 1e-9
        return waited >= self.get_parameter('stopped_time').value

    def imu_cb(self, msg):
        t = Time.from_msg(msg.header.stamp)
        if self.last_imu_t is None:
            self.last_imu_t = t
            self.t_state = t
            return
        # the step starts at the state's time: after an IMU dropout that is the last WHEEL prediction (UPDATE 4)
        dt = (t - (self.t_state if self.gap is not None else self.last_imu_t)).nanoseconds * 1e-9
        self.last_imu_t = t
        w_m = msg.angular_velocity.z
        trusted = True
        if self.gap is not None:            # a returning gyro is trusted only after the check (docstring UPDATE 4)
            trusted = self.gap.imu(t.nanoseconds * 1e-9, w_m - float(self.ekf.x[3]),
                                   rx=self.get_clock().now().nanoseconds * 1e-9)
            self.log_gap()
        if dt <= 0.0 or dt > 0.5:
            if dt > 0.5:
                self.t_state = t
            return
        self.t_state = t
        if not trusted:                     # RECOVERING: predict with the wheel model, not the returning gyro
            w_m = self.gap.yaw_rate(t.nanoseconds * 1e-9) + float(self.ekf.x[3])
        a = msg.linear_acceleration
        slipping = False
        if self.slip is not None and trusted:   # docstring UPDATE 3 (no IMU data for it while untrusted)
            self.slip.imu(t.nanoseconds * 1e-9, self.imu_c * a.x - self.imu_s * a.y)
            self.log_slip()
            slipping = self.slip.on
        stopped = self.is_stopped(t) and not slipping     # locked wheels while slipping: NOT parked
        self.v_used = self.v
        if self.wheel_comp is not None and trusted:     # speed gained since the wheel value was true (true m/s)
            a = msg.linear_acceleration
            dv = self.wheel_comp.on_imu(t.nanoseconds * 1e-9, self.imu_c * a.x - self.imu_s * a.y, dt, stopped)
            self.v_used = self.v + dv / max(float(self.ekf.x[4]), 0.5)      # raw units, as v
        use_obs = slipping and self.slip_use
        self.speed_used = self.slip.observer_speed if use_obs else float(self.ekf.x[4]) * self.v_used

        sw = None if trusted else self.gap_sigma_w
        if stopped and not trusted:         # no ZUPT from an untrusted gyro (docstring UPDATE 4)
            self.ekf.stationary_predict(dt)
        elif stopped:                       # ZUPT, or a turn while standing (docstring UPDATE 1)
            self.log_standstill(self.still.step(self.ekf, w_m, dt))
        else:
            self.log_standstill(self.still.moving())
            # sideslip (vehicle_model): 'steer' = kinematic from the steering, 'slip' = gyro + tyre slip
            beta = self.sideslip.beta(self.speed_used, w_m - self.ekf.x[3], self.steer)
            if use_obs:                     # wheel slip: the IMU speed, true m/s, no wheel scale (UPDATE 3)
                self.ekf.predict(self.speed_used, w_m, beta, dt, use_k=False, sigma_v=self.slip_sigma_v, sigma_w=sw)
            else:
                self.ekf.predict(self.v_used, w_m, beta, dt, sigma_w=sw)
            self.distance += abs(self.speed_used) * dt

        if self.fault is not None:                  # TEST ONLY (test_fault)
            if self.fault_t0 is None:
                self.fault_t0 = t
            elif (t - self.fault_t0).nanoseconds * 1e-9 >= self.fault[0]:
                _, dx_, dy_, dyaw_ = self.fault
                self.ekf.x[0] += dx_
                self.ekf.x[1] += dy_
                self.ekf.x[2] = wrap(self.ekf.x[2] + math.radians(dyaw_))
                self.history.buf.clear()
                self.get_logger().warn(f'TEST FAULT: ekf_global pose moved by ({dx_}, {dy_}) m, {dyaw_} deg')
                self.fault = None
        self.history.add(t.nanoseconds, self.ekf.x)
        self.publish_odom(msg.header.stamp, w_m)

    def log_standstill(self, ev):
        """One log line per standstill event (StandstillTurns, docstring UPDATE 1)."""
        if ev is None:
            return
        lg = self.get_logger()
        b = math.degrees(float(self.ekf.x[self.ekf.B]))
        if ev[0] == 'start':
            lg.warn(f'TURNED WHILE STANDING: the gyro reads {ev[1]:+.1f} deg/s with the wheels still (NIS {ev[2]:.0f} > '
                    f'{self.still.gate:g}) -> the heading follows the gyro, the bias is not updated')
        elif ev[0] == 'end':
            lg.info(f'{ev[2]}: turned {ev[1]:+.1f} deg while the wheels stood still (gyro bias kept at {b:+.2f} deg/s)')
        else:
            lg.warn(f'STANDSTILL "TURN" LASTED {ev[2]:.1f} s (> zupt_turn_max_s {self.still.max_turn_s:g}): taken as a '
                    f'GYRO-BIAS CHANGE, not a hand turn -> its heading change of {ev[1]:+.1f} deg undone, the bias '
                    f'({ev[3]:+.2f} deg/s) is learned again')

    def wheel_ideal_cb(self, msg):
        """SIMULATOR ONLY: /sim/wheel_ideal -> slip_check.SimGate (docstring UPDATE 3)."""
        self.slip_gate.update(Time.from_msg(msg.header.stamp).nanoseconds * 1e-9,
                              msg.twist.linear.x, msg.twist.linear.y)

    def publish_slip(self, row):
        """<diag_prefix>/slip + slip_hooks (state_pipeline -> state_estimate.slip_cb), one per decided wheel sample."""
        tw, z, vr, r, thr, on = row
        m = Float64MultiArray()
        m.data = [float(tw), 1.0 if on else 0.0, float(z), float(vr), float(r), float(thr),
                  float(self.slip.bias), float(self.slip.gated_n)]
        if self.slip_pub is not None:
            self.slip_pub.publish(m)
        for hook in self.slip_hooks:
            hook(m)

    def log_slip(self):
        """One log line per slip ON / OFF (docstring UPDATE 3)."""
        for t_s, on, why, r in self.slip.pop_events():
            if on:
                self.slip_t_on = t_s
                what = (f'the EKF predicts with the IMU speed {self.slip.observer_speed:.2f} m/s'
                        if self.slip_use else 'FLAG ONLY')
                self.get_logger().warn(f'WHEEL SLIP: {why} (wheel {float(self.ekf.x[4]) * self.v:.2f} m/s) -> {what}')
            else:
                dur = t_s - getattr(self, 'slip_t_on', t_s)
                self.get_logger().info(f'wheel slip over after {dur:.2f} s: {why} (residual {r:+.2f} m/s, '
                                       f'accelerometer bias {self.slip.bias:+.3f} m/s^2)')

    def amcl_cb(self, msg):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        if not p('use_amcl'):
            return
        t_ns = Time.from_msg(msg.header.stamp).nanoseconds
        x_then = self.history.at(t_ns)
        if x_then is None:
            return                                  # too old, or no IMU yet
        pose = msg.pose.pose
        off = p('amcl_offset_m')                    # nav2 AMCL half-cell offset (see docstring)
        z = np.array([pose.position.x + off, pose.position.y + off, yaw_from_quat(pose.orientation)])
        nu = z - x_then[:3]
        nu[2] = wrap(nu[2])
        R = amcl_measurement_noise(msg.pose.covariance, p('amcl_cov_scale'),
                                   p('amcl_min_sigma_pos'),
                                   math.radians(p('amcl_min_sigma_yaw_deg')))
        self.record_amcl_error(t_ns, z)
        cov = msg.pose.covariance
        kind = amcl_pose_class(cov[0], cov[1], cov[7], p('amcl_max_sigma'), p('amcl_corridor_sigma'))
        if kind == 'unsure':
            self.stats['unsure'] += 1               # AMCL itself is unsure (lost?) -> ignore this pose
            return

        nis = self.ekf.pose_nis(nu, R)
        if nis > p('gate'):
            self.stats['rej'] += 1
            if kind != 'confident':
                return                              # a corridor pose never counts towards a jump
            self.streak.append((t_ns, z.copy(), x_then[:3].copy()))
            if len(self.streak) >= p('reset_after_rejects'):
                ok, why = self.jump_check(t_ns, z, x_then)
                if ok:
                    self.get_logger().warn(f'{len(self.streak)} AMCL poses rejected in a row; {why} '
                                           '-> EKF jumps to AMCL')
                    self.ekf.reset_pose(z[0], z[1], z[2], R)
                    self.history.buf.clear()
                    self.streak = []
                    self.stats['jumps'] += 1
                else:
                    self.streak.pop(0)              # slide: the next rejected pose re-checks
                    self.stats['refused'] += 1
                    now = time.monotonic()
                    if now - self.last_refuse_log > 2.0:
                        self.last_refuse_log = now
                        self.get_logger().warn(f'AMCL rejected {p("reset_after_rejects")}+ times in a row, '
                                               f'NOT jumping: {why} -> EKF continues on odometry')
            return

        self.streak = []
        nis, dx = self.ekf.pose_update(nu, R)
        self.history.shift(dx)
        self.stats['acc'] += 1
        if kind == 'corridor':
            self.stats['corridor'] += 1
        self.stats['nis'] += nis
        nis_msg = Float64(data=nis)
        self.nis_pub.publish(nis_msg)
        for hook in self.fix_hooks:
            hook(nis_msg)

    def reset_cb(self, msg):
        pose = msg.pose.pose
        self.ekf.reset_pose(pose.position.x, pose.position.y, yaw_from_quat(pose.orientation),
                            np.diag([0.3 ** 2, 0.3 ** 2, math.radians(10.0) ** 2]))
        self.history.buf.clear()
        self.distance = 0.0
        self.path.poses.clear()
        self.get_logger().info(f"EKF pose reset from {self.get_parameter('initialpose_topic').value} "
                               '(bias and speed scale kept)')

    def jump_check(self, t_ns, z, x_then):
        """-> (ok, reason): may the EKF jump to AMCL? (docstring RECOVERY)"""
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        ok, worst_pos, worst_yaw = motion_consistent(self.streak, p('reset_tol_pos'), p('reset_tol_frac'),
                                                     math.radians(p('reset_tol_yaw_deg')))
        if not ok:
            return False, (f'AMCL did not move like the odometry ({worst_pos:+.2f} m over the tolerance, '
                           f'turn {math.degrees(worst_yaw):.1f} deg)')
        if not p('reset_scan_check'):
            return True, 'AMCL moved like the odometry (scan check off)'
        scan = self.scan_at(t_ns)
        if self.wall_map is None or scan is None:
            return False, 'AMCL moved like the odometry, but no map or no scan for the scan check'
        fit_amcl, fit_ekf = self.scan_fits(scan, (z, x_then[:3]))
        if fit_amcl < p('reset_scan_min') or fit_amcl - fit_ekf < p('reset_scan_margin'):
            return False, (f'AMCL moved like the odometry, but the scan does not fit it clearly better '
                           f'(scan fit AMCL {fit_amcl:.2f}, EKF {fit_ekf:.2f})')
        return True, (f'AMCL moved like the odometry and the scan fits it better '
                      f'(scan fit AMCL {fit_amcl:.2f}, EKF {fit_ekf:.2f})')

    def scan_fits(self, scan, poses):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        r, ang = scan_beams(scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min,
                            scan.range_max, int(p('scan_beams')))
        laser = (p('laser_x'), p('laser_y'), p('laser_yaw'))
        return [scan_fit(self.wall_map, q, laser, ang, r, scan.range_max, p('scan_tol')) for q in poses]

    def scan_cb(self, msg):
        t_ns = Time.from_msg(msg.header.stamp).nanoseconds
        self.scans.append((t_ns, msg))
        if self.dfield is None:
            return
        if not self.get_parameter('map_check').value:   # switched off at runtime (ros2 param set)
            if self.mc_monitor.alarm:                    # nothing could clear it any more -> clear it now
                self.mc_monitor.reset()
                self.get_logger().warn('MAP CHECK switched off while its alarm was on -> alarm cleared')
                self.publish_map_check(t_ns, None, 0.0, float('nan'))
            return
        if self.mc_budget.due():
            self.map_check(t_ns, msg)

    def map_check(self, t_ns, scan):
        """Map check of this scan (docstring MAP CHECK) -> MismatchAlarm -> topic + state_estimate."""
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        t0 = self.mc_budget.start()
        st = self.history.nearest(t_ns)
        if st is None:
            return
        t_state, x = st
        r, ang = scan_beams(scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min,
                            scan.range_max, int(p('map_check_beams')))
        res = shift_search(self.dfield, x[:3], (p('laser_x'), p('laser_y'), p('laser_yaw')), ang, r,
                           self.mc_shifts, tol=p('map_check_tol'))
        mon = self.mc_monitor
        was = mon.alarm
        alarm = mon.update(res, t_ns * 1e-9)
        ms = self.mc_budget.stop(t0)
        s = self.mc_stats
        s['runs'] += 1
        s['ms'].append(ms)
        if mon.last_class >= BAD_ALONG:
            s['bad'] += 1
        if res is not None:
            s['last'] = res.shift
        if alarm and not was:                       # BAD scans: res is not None
            s['alarms'] += 1
            if mon.cause == BAD_FIT:
                why = (f'the scan fits the map only {res.fit0:.2f} at the EKF pose (sideways / heading error, or '
                       f'more than {p("map_check_max_shift"):.1f} m off)')
            else:
                ahead = 'AHEAD of' if res.shift < 0 else 'BEHIND'
                why = (f'the scan fits the map best {abs(res.shift):.2f} m along the heading from the EKF pose -> the '
                       f'EKF is ~{abs(res.shift):.2f} m {ahead} the car (fit {res.fit0:.2f} -> {res.fit_best:.2f}, '
                       f'optimum {res.width:.2f} m wide)')
            self.get_logger().warn(f'MAP CHECK ALARM ({mon.n_bad} bad scans in a row): {why} '
                                   '-> FLAG_MAP_MISMATCH, mode DEGRADED')
        elif was and not alarm:                     # CLEAN scans: res is not None
            s['cleared'] += 1
            self.get_logger().info(f'MAP CHECK cleared: {mon.n_clean} clean scans in a row (fit {res.fit0:.2f}, '
                                   f'gain {res.gain:.2f}) -> FLAG_MAP_MISMATCH off')
        self.publish_map_check(t_ns, res, ms, (t_ns - t_state) * 1e-6)

    def publish_map_check(self, t_ns, res, ms, state_age_ms):
        """<diag_prefix>/map_check and the in-process hooks (state_pipeline -> state_estimate.map_check_cb)."""
        nan = float('nan')
        m = Float64MultiArray()
        m.data = [t_ns * 1e-9,
                  res.shift if res else nan, res.gain if res else nan, res.width if res else nan,
                  res.fit0 if res else nan, float(res.n_beams) if res else 0.0,
                  1.0 if self.mc_monitor.alarm else 0.0, float(ms), float(self.mc_budget.every_n),
                  float(state_age_ms), float(self.mc_monitor.last_class)]
        if self.mc_pub is not None:
            self.mc_pub.publish(m)
        for hook in self.map_check_hooks:
            hook(m)

    def scan_at(self, t_ns, max_dt_ns=50_000_000):
        """The scan taken at t_ns (AMCL's pose carries the stamp of the scan it used)."""
        if not self.scans:
            return None
        ts, msg = min(self.scans, key=lambda e: abs(e[0] - t_ns))
        return msg if abs(ts - t_ns) <= max_dt_ns else None

    def map_cb(self, msg):
        """Once: is amcl_offset_m half the resolution of the map AMCL uses? + keep its walls."""
        if self.map_checked:
            return
        self.map_checked = True
        info = msg.info
        o = info.origin
        if abs(o.orientation.z) > 1e-6 or abs(o.orientation.x) > 1e-6 or abs(o.orientation.y) > 1e-6:
            self.get_logger().warn('map origin is rotated -> no scan check (jumps to AMCL refused)')
        else:
            self.wall_map = WallMap.from_grid(msg.data, info.width, info.height, info.resolution,
                                              o.position.x, o.position.y)
            self.get_logger().info(f'scan check: map {info.width} x {info.height} cells @ {info.resolution:.3f} m, '
                                   f'{int(self.wall_map.occ.sum())} wall cells')
            if self.map_check_on:
                # in a BACKGROUND THREAD: numpy without scipy needs 0.1-0.23 s on the laptop (more on a Pi), and
                # inside this callback that stalled the whole estimator (single-threaded executor) that long
                data = np.asarray(msg.data, dtype=np.int16).copy()
                geo = (info.width, info.height, info.resolution, o.position.x, o.position.y)
                threading.Thread(target=self._build_dfield, args=(data, geo), daemon=True,
                                 name='map_check_field').start()
        off, res = self.get_parameter('amcl_offset_m').value, msg.info.resolution
        if abs(off - 0.5 * res) > 1e-4:
            self.get_logger().warn(
                f'amcl_offset_m is {off:.4f} m but the map resolution is {res:.4f} m: it should be '
                f'{0.5 * res:.4f} m (nav2 AMCL half-cell offset, see the ekf_node docstring)')
        else:
            self.get_logger().info(f'amcl_offset_m {off:.4f} m = half the map resolution ({res:.3f} m): OK')

    def _build_dfield(self, data, geo):
        """Background thread (map_cb): the map check starts with the first scan after this finishes."""
        t0 = time.perf_counter()
        try:
            field = DistanceField.from_grid(data, *geo)
        except Exception as ex:                     # noqa: BLE001 -- the check must never stop the estimator
            self.get_logger().error(f'map check: distance field FAILED ({ex}) -> no map check, no FLAG_MAP_MISMATCH')
            return
        self.dfield = field                         # one reference assignment: atomic for scan_cb
        self.get_logger().info(f'map check: distance field built in {1000 * (time.perf_counter() - t0):.0f} ms '
                               f'(background thread, {field.n_walls} wall cells)')

    def truth_cb(self, msg):
        self.truth = msg
        tp = msg.pose.pose
        self.truth_hist.append((Time.from_msg(msg.header.stamp).nanoseconds,
                                tp.position.x, tp.position.y))

    def record_amcl_error(self, t_ns, z):
        """AMCL's own error vs. truth at the scan time (simulation only)."""
        if not self.truth_hist:
            return
        _, tx, ty = min(self.truth_hist, key=lambda e: abs(e[0] - t_ns))
        self.stats['amcl_err'].append(math.hypot(z[0] - tx, z[1] - ty))

    # --- outputs -------------------------------------------------------------

    def publish_odom(self, stamp, w_m):
        px, py, yaw, b, k = (float(v) for v in self.ekf.x)
        P = self.ekf.P
        q = quat_from_yaw(yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.path.header.frame_id
        odom.child_frame_id = self.get_parameter('child_frame_id').value
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) = q
        cov = [0.0] * 36
        cov[0], cov[1], cov[5] = P[0, 0], P[0, 1], P[0, 2]
        cov[6], cov[7], cov[11] = P[1, 0], P[1, 1], P[1, 2]
        cov[30], cov[31], cov[35] = P[2, 0], P[2, 1], P[2, 2]
        odom.pose.covariance = [float(c) for c in cov]
        odom.twist.twist.linear.x = self.speed_used          # k * v_used (delay comp.), or the IMU speed while slipping
        odom.twist.twist.angular.z = float(w_m) - b
        if self.gap is not None and (self.gap.state != 0 or self.gap_last_pub != 0):   # UPDATE 4, BEFORE the odometry
            t_s = Time.from_msg(stamp).nanoseconds * 1e-9
            gm = Float64MultiArray()
            gm.data = [t_s, float(self.gap.state), self.gap.gap_len(t_s), float(w_m) - b]
            self.gap_pub.publish(gm)
            for hook in self.gap_hooks:
                hook(gm)
            self.gap_last_pub = self.gap.state
        self.odom_pub.publish(odom)
        for hook in self.odom_hooks:
            hook(odom)
        self.imu_count += 1
        if self.imu_count % self.get_parameter('diag_every').value == 0:
            self.bias_pub.publish(Float64(data=b))
            self.bias_sigma_pub.publish(Float64(data=self.ekf.sigma(3)))
            self.scale_pub.publish(Float64(data=k))

        if self.tf_pub is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.path.header.frame_id
            tf.child_frame_id = odom.child_frame_id
            tf.transform.translation.x = px
            tf.transform.translation.y = py
            (tf.transform.rotation.x, tf.transform.rotation.y,
             tf.transform.rotation.z, tf.transform.rotation.w) = q
            self.tf_pub.sendTransform(tf)

    def publish_path(self):
        px, py, yaw = (float(v) for v in self.ekf.x[:3])
        ps = PoseStamped()
        ps.header.frame_id = self.path.header.frame_id
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = px, py
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = quat_from_yaw(yaw)
        self.path.poses.append(ps)
        del self.path.poses[:-self.get_parameter('path_max_len').value]
        self.path.header.stamp = ps.header.stamp
        self.path_pub.publish(self.path)

    def report(self):
        px, py, yaw, b, k = self.ekf.x
        line = f'[{self.mode}] driven {self.distance:5.1f} m'
        if self.global_mode and self.truth is not None:
            tp = self.truth.pose.pose
            pos_err = math.hypot(px - tp.position.x, py - tp.position.y)
            yaw_err = math.degrees(wrap(yaw - yaw_from_quat(tp.orientation)))
            line += (f' | pos err {pos_err:5.2f} m (+-{2 * self.ekf.sigma_pos():4.2f})'
                     f' | heading err {yaw_err:+6.2f} deg'
                     f' (+-{2 * math.degrees(self.ekf.sigma(2)):4.2f})')
        line += (f' | bias {math.degrees(b):+5.2f} deg/s'
                 f' (+-{2 * math.degrees(self.ekf.sigma(3)):4.2f})')
        if self.global_mode:
            s = self.stats
            line += f' | speed scale {k:5.3f} (+-{2 * self.ekf.sigma(4):5.3f})'
            mean_nis = s['nis'] / s['acc'] if s['acc'] else float('nan')
            amcl = (f"{sum(s['amcl_err']) / len(s['amcl_err']):4.2f} m"
                    if s['amcl_err'] else ' n/a')
            line += (f" | AMCL used {s['acc']:2d} rej {s['rej']:2d} unsure {s['unsure']:2d}"
                     f" corridor {s['corridor']:2d} NIS {mean_nis:4.1f} AMCL err {amcl}")
            if s['refused'] or s['jumps']:
                line += f" | jump refused {s['refused']} JUMPED {s['jumps']}"
            fit = self.scan_fit_now()
            if fit is not None:
                line += f' | scan fit {fit:.2f}'
            mc = self.mc_stats
            if self.map_check_on and self.dfield is None:
                line += ' | map check: NO MAP / distance field yet'
            elif self.map_check_on and mc['runs']:
                line += (f" | map check {mc['runs']} scans, bad {mc['bad']}, ALARMS {mc['alarms']}, cleared "
                         f"{mc['cleared']}{' (ALARM ON)' if self.mc_monitor.alarm else ''}, last shift "
                         f"{mc['last']:+.2f} m, {np.mean(mc['ms']):.2f} ms (max {max(mc['ms']):.2f}), "
                         f"every {self.mc_budget.every_n}")
            self.mc_stats = self.new_mc_stats()
        if self.slip is not None:
            sl = self.slip
            line += (f' | slip episodes {sl.episodes}, on {sl.on_s:.1f} s, accel bias {sl.bias:+.3f}'
                     + (f', sim-gated {sl.gated_n}' if self.slip_gate is not None else '')
                     + (' (SLIP ON)' if sl.on else ''))
        if self.gap is not None:
            g = self.gap
            line += (f' | IMU dropouts {g.episodes}' + (f', longest {g.longest:.2f} s, wheel predictions '
                     f'{self.gap_predictions}, gyro rejects {g.rejects}' if g.episodes else '')
                     + (f' ({GAP_NAMES[g.state]})' if g.state else ''))
        st = self.still
        line += f' | ZUPTs {st.zupts}' + (f' | turned while standing: {st.turn_samples} samples' if st.turn_samples
                                          else '') + (f', bias changes {st.bias_changes}' if st.bias_changes else '')
        self.stats = self.new_stats()
        self.get_logger().info(line)

    @staticmethod
    def new_stats():
        return {'acc': 0, 'rej': 0, 'nis': 0.0, 'amcl_err': [], 'unsure': 0, 'corridor': 0, 'refused': 0,
                'jumps': 0}

    @staticmethod
    def new_mc_stats():
        return {'runs': 0, 'bad': 0, 'alarms': 0, 'cleared': 0, 'last': float('nan'), 'ms': []}

    def scan_fit_now(self):
        """Health check every 2 s: how well does the latest scan fit the map at the EKF's
        pose? (~0.9-1.0 when localized; proves the scan check sees what it should.)"""
        if self.wall_map is None or not self.scans:
            return None
        t_ns, scan = self.scans[-1]
        x = self.history.at(t_ns)
        return None if x is None else self.scan_fits(scan, (x[:3],))[0]


def main(args=None):
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_scale()                       # last learned k, at Ctrl+C
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
