# One confirmed/tentative track maintained by the IMM tracker.
int32 id                            # unique track id
geometry_msgs/Pose pose             # track box center + orientation (yaw about the vertical axis)
geometry_msgs/Vector3 dimensions    # bounding-box extents (length, width, height), m
float64[] state                     # BEV motion state [x, z, vx, vz]
float64[] covariance                # state covariance, row-major, length = state^2
float64[] mode_probabilities        # IMM mode probabilities; length = bank size (CV, CA, one per CT rate)
uint32 age                          # number of updates since the track was created
uint32 missed_count                 # consecutive frames with no associated detection
