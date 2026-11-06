# ROS 2 Humble (Ubuntu 22.04) for the Raspberry Pi 5 (Raspberry Pi OS, Debian 13).
# Build on the Pi:  docker build -t apex-humble --build-arg UID=$(id -u) .
FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential python3-colcon-common-extensions python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Dependencies of our workspace (found with: rosdep install --from-paths src --ignore-src -r -s)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-nav2-amcl ros-humble-nav2-map-server ros-humble-nav2-lifecycle-manager \
    ros-humble-topic-tools ros-humble-cv-bridge ros-humble-ackermann-msgs \
    python3-matplotlib python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with the Pi user's UID, so files written to the mounted workspace belong to the Pi user
ARG UID=1000
RUN useradd -m -u ${UID} -s /bin/bash dev
USER dev
WORKDIR /home/dev/ros2_ws

# Every interactive shell loads ROS, and the workspace once it has been built
RUN echo 'source /opt/ros/humble/setup.bash' >> /home/dev/.bashrc \
 && echo '[ -f /home/dev/ros2_ws/install/setup.bash ] && source /home/dev/ros2_ws/install/setup.bash' >> /home/dev/.bashrc
