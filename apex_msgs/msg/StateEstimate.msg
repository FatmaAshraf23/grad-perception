# APEX StateEstimate -- schema "apex-state-estimate/1.0"
# Perception -> Control state contract v0.2, section 2. SI units, map (track) frame,
# all values refer to the vehicle CG. NaN / Inf are never sent.
#
# header.stamp = time at which the state is VALID (newest IMU sample fused), same
# instant as stamp_us; header.frame_id = "map". Perception never forward-predicts;
# control propagates from this time to its command time.

std_msgs/Header header

string schema_version          # "apex-state-estimate/1.0"; reject unknown major versions
uint32 sequence_id             # +1 per message; gaps reveal drops
uint64 stamp_us                # time the state is valid [us] (same clock as header.stamp)
uint64 publish_us              # send time [us]; publish_us - stamp_us = estimator latency
string track_id                # from apex_track; must match control's loaded track
string track_hash              # 16 hex chars; mismatch -> control rejects the message

# --- pose (CG, map frame) ---
float64 x_m
float64 y_m
float64 psi_rad                # yaw of body x from map +x, CCW +, wrapped to (-pi, pi]

# --- control state x = [vx, vy, r, e_psi, s_abs, e_y] ---
float64 vx_mps                 # forward velocity of the CG, body frame
float64 vy_mps                 # lateral velocity of the CG, body frame, left +
float64 r_radps                # yaw rate, CCW +, gyro bias removed
float64 s_abs_m                # continuous progress along the centre line; never wraps
float64 e_y_m                  # signed distance centre line -> CG, left +
float64 e_psi_rad              # psi - centre-line tangent heading, wrapped to (-pi, pi]

# --- track progress helpers ---
float64 s_track_m              # s_abs_m modulo track length L, in [0, L)
int32 lap                      # s_abs_m = lap * L + s_track_m

# --- recommended ---
float64 ax_mps2                # body-frame CG acceleration, gravity and bias removed
float64 ay_mps2

# --- 1-sigma uncertainty of each estimate ---
float64 std_xy_m               # major axis of the position ellipse
float64 std_psi_rad
float64 std_vx_mps
float64 std_vy_mps
float64 std_r_radps
float64 std_e_y_m
float64 std_e_psi_rad

# --- health (contract section 6; thresholds provisional) ---
uint8 MODE_INIT = 0
uint8 MODE_OK = 1
uint8 MODE_DEGRADED = 2
uint8 MODE_LOST = 3
uint8 mode

uint16 FLAG_LIDAR_STALE = 1          # bit 0: no accepted LiDAR correction within 0.3 s
uint16 FLAG_IMU_STALE = 2            # bit 1: IMU gap > 10 ms
uint16 FLAG_WHEEL_STALE = 4          # bit 2: wheel-speed gap > 30 ms
uint16 FLAG_WHEEL_SLIP = 8           # bit 3: wheel speed disagrees with IMU; wheel de-weighted
uint16 FLAG_VY_KINEMATIC = 16        # bit 4: vy from the no-slip kinematic model
uint16 FLAG_CORRECTION_STEP = 32     # bit 5: last LiDAR correction moved x/y by > 5 cm
uint16 FLAG_OFF_TRACK = 64           # bit 6: |e_y| > half the local track width
uint16 FLAG_FILTER_RESET = 128       # bit 7: estimator re-initialized since the last message
uint16 FLAG_PROJECTION_RESEEDED = 256  # bit 8: Frenet projection used a global search
uint16 FLAG_NEAR_SINGULAR = 512      # bit 9: 1 - kappa * e_y < 0.3
uint16 flags

uint16 valid_for_ms            # control may use this state until stamp + valid_for_ms (30 ms, provisional)
