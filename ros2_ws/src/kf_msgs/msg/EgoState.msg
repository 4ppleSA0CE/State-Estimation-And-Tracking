# Full ego vehicle state published by the ESKF/UKF nodes.
std_msgs/Header header              # stamp + frame_id (map)
geometry_msgs/Pose pose             # position + orientation in the map frame
geometry_msgs/Twist twist           # linear + angular velocity
geometry_msgs/Vector3 accel_bias    # accelerometer bias estimate, m/s^2
geometry_msgs/Vector3 gyro_bias     # gyroscope bias estimate, rad/s
float64[225] covariance             # 15x15 error-state covariance, row-major
uint8 filter_type                   # which filter produced this estimate
uint8 FILTER_ESKF=0
uint8 FILTER_UKF=1
