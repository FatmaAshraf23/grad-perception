# APEX Obstacle -- one entry of apex_msgs/ObstacleArray (schema "apex-obstacles/0.1", DRAFT: to be agreed with
# planning and control). Map frame, SI units. Produced by lidar_perception obstacle_node (LiDAR map difference).

uint32 id                      # stays the same while the obstacle is in the list; never reused
float64 x_m                    # estimated centre, map frame
float64 y_m
float64 radius_m               # half its size as the LiDAR saw it (at least the cone radius the detector assumes)
float64 std_xy_m               # 1-sigma position uncertainty per axis (includes a floor for errors that don't average out)
float64 s_track_m              # Frenet (apex_track): distance along the centre line, [0, L)
float64 e_y_m                  # Frenet: signed distance from the centre line, left +
uint32 detections              # number of scans it was found in
uint64 first_seen_us           # stamp of the scan it was first found in [us]
uint64 last_seen_us            # stamp of the latest scan it was found in [us]
bool seen_now                  # found in THIS message's scan (false = remembered while out of sight)

# --- camera confirmation (schema "apex-obstacles/0.2", 2026-11-07) -------------------------------------------------
# Filled ONLY on /obstacles/fused (sensor_fusion fusion_node); on the plain LiDAR list /obstacles they stay 0 / "".
# The camera never removes an obstacle: NO_CONE only says "the camera looked at it several times and saw no cone".
uint8 CAMERA_UNKNOWN = 0       # not looked at enough yet (out of view, too far, too small, hidden, or < 2 matches)
uint8 CAMERA_CONFIRMED = 1     # a cone box matched it at least twice (and not 10 looks in a row without one since)
uint8 CAMERA_NO_CONE = 2       # the camera looked at it 10 times in a row and saw no cone there (still an obstacle!)
uint8 camera_state
string class_name              # "cone" while CONFIRMED, else ""
float32 camera_confidence      # mean detector confidence of the matched boxes (0 = never matched)
uint32 camera_looks            # pictures in which the camera SHOULD have seen it (in view, big enough, LiDAR-visible)
uint32 camera_hits             # pictures in which a box matched it
uint64 camera_last_hit_us      # time of the newest picture with a match [us] (0 = never)
