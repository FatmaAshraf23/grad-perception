# APEX CameraDetection -- one object the camera's detector (YOLO) found, as a box in the picture (schema
# "apex-camera-detections/0.1", DRAFT). Pixels of the picture described by the array's image_width x image_height
# (the stream the detector saw); origin = top-left corner, u = column (right), v = row (down).

string class_name          # "cone"
int32 class_id             # the detector's class index (-1 = unknown)
float32 confidence         # 0 .. 1
float32 u_min              # box left [px]
float32 v_min              # box top [px]
float32 u_max              # box right [px]
float32 v_max              # box bottom [px]
