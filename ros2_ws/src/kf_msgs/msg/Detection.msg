# A single object detection fed to the tracker.
geometry_msgs/Pose pose             # detected object pose
geometry_msgs/Vector3 dimensions    # bounding-box extents (length, width, height), m
float64[36] covariance              # 6x6 pose covariance, row-major
uint8 classification                # object class id ('class' is a C++ keyword)
int32 object_id                     # source-assigned id, -1 if unknown
