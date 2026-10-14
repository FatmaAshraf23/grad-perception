# APEX CameraDetectionArray -- everything the camera's detector found in ONE picture (schema
# "apex-camera-detections/0.1", DRAFT). Published by camera_publisher yolo_node (the car) and sensor_fusion fake_camera
# (the simulator), so the fusion works the same with both.
# header.stamp    = when the PICTURE was taken (the image message's stamp) -- NOT when the detector finished: at 1.5 m/s
#                   the car moves 12 cm in 80 ms, so the fusion must use the car's pose at THIS instant.
# header.frame_id = the camera's optical frame (z forward, x right, y down): with /camera/camera_info + TF a box becomes
#                   a direction in the world (sensor_fusion projection.pixel_to_ray).

std_msgs/Header header
string schema_version       # "apex-camera-detections/0.1"
uint32 sequence_id          # +1 per message; gaps reveal drops
string source               # "yolo" (the real detector) or "fake_camera" (the simulator)
uint32 image_width          # the picture the boxes refer to
uint32 image_height
float32 latency_ms          # publish time - header.stamp: how old the picture is when its boxes arrive
CameraDetection[] detections
