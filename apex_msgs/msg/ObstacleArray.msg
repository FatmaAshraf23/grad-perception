# APEX ObstacleArray -- the obstacle list (schema "apex-obstacles/0.1", DRAFT: to be agreed with planning and
# control). One message per LiDAR scan. header.stamp = when the scan was taken (observation time, NOT send
# time); header.frame_id = "map". Only CONFIRMED obstacles are listed (found in >= 2 scans).

std_msgs/Header header

string schema_version          # "apex-obstacles/0.1"
uint32 sequence_id             # +1 per message; gaps reveal drops
uint64 stamp_us                # observation time [us] (same instant as header.stamp)
uint64 publish_us              # send time [us]; publish_us - stamp_us = detection latency
string track_id                # from apex_track, as in StateEstimate
string track_hash

uint8 STATUS_OK = 0                    # list updated with this scan
uint8 STATUS_FROZEN_LOCALIZATION = 1   # StateEstimate not OK: list NOT updated (positions may be outdated)
uint8 STATUS_FROZEN_SCAN = 2           # too much of this scan is unexplained by the map (pose probably wrong): NOT updated
uint8 STATUS_RESET = 3                 # list emptied in this message after a jump of the car's pose
uint8 status
uint32 list_epoch              # +1 every time the list is emptied; obstacles of an older epoch are gone

float32 max_range_m            # obstacles farther than this from the LiDAR are not detected
float32 blind_strip_m          # obstacles closer than this to a wall of the map can't be detected

Obstacle[] obstacles
