from state_estimation.imu_to_cg import imu_to_cg

# Constant 2 m/s on a 1 m radius left circle, sensor 10 cm ahead of the CG.
reading = (-0.4, 4.0)
ax, ay = imu_to_cg(*reading, r=2.0, r_dot=0.0, mount_yaw=0.0, px=0.10, py=0.0)
print(f'IMU reads : ax = {reading[0]:+.2f}  ay = {reading[1]:+.2f}  m/s^2')
print(f'At the CG : ax = {ax:+.2f}  ay = {ay:+.2f}  m/s^2')
