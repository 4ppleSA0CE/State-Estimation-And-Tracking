# One confirmed/tentative track maintained by the IMM tracker.
int32 id                            # unique track id
float64[] state                     # motion state vector (length depends on model)
float64[] covariance                # state covariance, row-major, length = state^2
float64[3] mode_probabilities       # IMM mode probabilities (CV / CA / CT)
uint32 age                          # number of updates since the track was created
uint32 missed_count                 # consecutive frames with no associated detection
