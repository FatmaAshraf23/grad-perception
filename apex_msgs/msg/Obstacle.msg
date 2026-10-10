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
