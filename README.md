# quaternion_eskf
Orientation estimation using quaternions and the Error State Kalman Filter (ESKF)

## Run the ESKF

    python3 main.py <dataset> [--update]

where \<dataset\> is (1-6, or custom). Adding the --update option will enable the update step. 

- Sensor data will be plotted, and sensor mean and average will be printed. Close window to continue.
- Quaternion components (true and estimated) will be plotted, where the estimated quaternion components are either with or without the update step (if the --update option was given or not). Angular errors will be plotted and printed (if the --update option is given, angular errors with and without update step will be plotted in the same graph). The diagonal components of the error covariance (local and global) will be plotted. Close windows to continue.
- A 3D visualization will be displayed, with the estimated and true orientation.