# Bench tools — measure the sensors BEFORE the car exists

Two sensors have properties that belong to the sensor itself, not to how it is mounted. We can
measure them on a table the day the parts arrive, and every number goes straight into our software.

| Tool | Measures | The number goes into |
|---|---|---|
| `lidar_range_check.py` | range offset, scale error, noise of the RPLIDAR | `range_offset:=` in `obstacles.launch.py`; the noise tells us if `SIGMA_DET` / `SCAN_CORR` (tuned to the simulator's 2 cm) are honest |
| `imu_static_check.py` | gyro bias, white noise, bias drift (Allan deviation), turn-on bias scatter, accelerometer sanity | EKF gyro noise, bias process noise, initial bias uncertainty |
| `bench_record.py` | records `/scan` or `/imu/data` to `.npz` for the two tools above | — |
| `test_bench.py` | proves the tools recover numbers we put into synthetic data (12 tests) | — |

Do **not** do the sensor-mount calibrations here (wheel speed, steering, camera–LiDAR extrinsics):
they depend on the real mounting and must be redone when a bracket moves.

## 0. Check the tools (any computer with numpy)

    python3 test_bench.py        # expect: 12 of 12 passed

## 1. LiDAR range offset (about 20 minutes)

**Why:** a LiDAR reads a little long or short by a nearly constant amount. Our obstacle node subtracts
one number from every return, so a wrong number shifts every cone by that much.

1. Find a **flat, matte, light-coloured wall** (painted wall; not glass, mirror, dark or shiny) at least
   3 m wide and clear of furniture. Why: glass and black surfaces change how the laser returns.
2. Put the RPLIDAR on a stable table. Power it and start the driver (`ros2 topic hz /scan` should show ~5–10 Hz).
3. Measure the **perpendicular distance from the centre of the rotating head to the wall** with a tape
   measure and a set square (or a laser distance meter). Use D = 1, 2, 3, 4 and 5 m (as far as the wall allows).
   Why this matters: an error of 1 cm in D is an error of 1 cm in the offset. Measure twice.
4. At each distance record one minute (the LiDAR does not have to face the wall exactly):

        python3 bench_record.py scan --topic /scan --seconds 60 --out scan_1m.npz

   Keep still, keep people out of the view.
5. Analyse all distances together:

        python3 lidar_range_check.py --run 1.0:scan_1m.npz --run 2.0:scan_2m.npz --run 3.0:scan_3m.npz \
            --run 4.0:scan_4m.npz --run 5.0:scan_5m.npz --out result_lidar/

6. Read the report:
   * **RANGE OFFSET TO USE** — put it in `range_offset:=` (metres). Positive = the LiDAR reads too long.
   * **scale error** — if the offset grows with distance, the report says so and gives 2 / 4 / 6 m values.
   * **noise** — compare with the simulator's 2 cm. If the real noise is much bigger, the obstacle tracker needs
     bigger tolerances (we re-tune on a real recording).
   * Warnings tell you if the wall was not flat, objects stood in front, or the beams returned nothing.
7. Optional but useful: repeat at 2 m with a darker surface (cardboard). If the offset differs by > 1 cm,
   cone colour matters on the real track.

## 2. IMU gyro bias, noise and drift (60+ minutes of waiting, no work)

**Why:** the EKF integrates the gyro. It needs to know (a) how noisy each reading is, (b) how fast the
bias wanders. Both come out of one recording of an IMU lying still.

1. Fix the IMU board firmly on a heavy surface (a floor tile or a brick on foam is better than a desk).
   No fans, no motors running, nobody walking heavily nearby. Lay it flat (z axis up).
2. Power it **and start recording immediately**, so the warm-up is in the data:

        python3 bench_record.py imu --topic /imu/data --seconds 3600 --out imu_run1.npz

   (If your IMU logger writes CSV instead: columns `t,gx,gy,gz[,ax,ay,az]`; the tool reads that too.)
3. For the **turn-on bias** repeat 4 more times: power off, power on, record 3 minutes, still, each to its own file
   (`imu_p2.npz` …). Why: the bias is slightly different at every power-up; the EKF's initial uncertainty
   should cover that scatter.
4. Analyse:

        python3 imu_static_check.py imu_run1.npz imu_p2.npz imu_p3.npz imu_p4.npz imu_p5.npz --out result_imu/

   Long and short files can be mixed; the Allan numbers come from the long one.
5. Read the report — the block "Numbers for the EKF":
   * gyro white noise per sample (rad/s) → the EKF's gyro measurement noise,
   * bias random walk K (rad/s/√s) → the bias process noise,
   * turn-on bias std → initial bias uncertainty (use 2–3× it),
   * bias at this power-up → what the car's start-up calibration (car standing still) should find. If it
     finds something very different, the car was moving or the IMU is vibrating.
6. Look at `allan.png`: a V-shape. Falling line (slope −1/2) = white noise; the bottom = bias instability;
   rising line (slope +1/2) = bias random walk. A curve that only falls means the recording was too short.

## 3. How the numbers are obtained (so you can explain them)

* **LiDAR:** for a wall at perpendicular distance D, a beam at angle Δ from the wall's normal would read
  D / cos Δ. `offset = reading − D / cos Δ`, averaged over beams and scans. The wall direction comes from a
  straight-line fit, so the LiDAR need not point exactly at the wall. Noise: per-beam spread across scans.
  The "±" combines scan-to-scan noise and beam-to-beam differences (each beam has its own small repeatable error).
* **IMU:** Allan variance σ²(τ) = ½·mean((avg_{k+1} − avg_k)²) over blocks of length τ, fitted with
  N²/τ + (2 ln2/π)B² + K²τ/3 (white noise + bias instability + rate random walk).
  Per-sample noise = N/√dt.

## Self-tests: what is proven

Synthetic wall with a known offset (also negative), a tilted wall, a box standing in front of it, 3 % lost
beams, per-beam errors: offset within 3 mm, noise within 15 %, and the "±" covers the truth in ≥ 75 % of
random recordings. Scale error 1 % recovered to ±0.15 %. A wrongly typed D is caught. Synthetic gyro (1 h):
white noise within 10 %, rate random walk within 30 %, dropped samples and a bumped IMU flagged.
These tests prove the *maths* works; the real sensors may add effects the synthetic data does not
(temperature drift, reflectivity, vibration), which is why the report prints warnings.
