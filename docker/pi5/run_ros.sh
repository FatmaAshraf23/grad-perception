#!/usr/bin/env bash
# Start the Humble container. The LiDAR is passed in only if it is plugged in.
DEVICES=()
[ -e /dev/ttyUSB0 ] && DEVICES+=(--device /dev/ttyUSB0)
exec docker run -it --rm --net=host --name apex-ros \
  -v "$HOME/ros2_ws:/home/dev/ros2_ws" \
  "${DEVICES[@]}" --group-add dialout apex-humble "$@"
