# quaternion_eskf
Orientation estimation using quaternions and the Error State Kalman Filter (ESKF). For detailed explanation of the implementation, see quaternion_eskf.pdf.

## Run the ESKF

    python3 main.py <dataset> [--update]

where \<dataset\> is (1-6, or custom). Adding the --update option will enable the update step. 

- Sensor data will be plotted, and sensor mean and average will be printed. Close window to continue.

    <img src="plots/example_sensor.png" title="Sensor data" alt="sensor" width="250"/>

- Quaternion components (true, estimated, error), angular errors and diagonal components of (local/global) covariance will be plotted. The quaternion components will be shown with the update step if  the --update option is given. Angular errors is shown with and without the update step if the --update option is given. Close windows to continue.

    <img src="plots/example_quat.png" title="Quaternions components" alt="quat" width="200"/>
    <img src="plots/example_error.png" title="Angular error" alt="error" width="218.5"/> 
    <img src="plots/example_covariance.png" title="Covariance" alt="covariance" width="218.5"/>

- A 3D visualization will be displayed, with the estimated and true orientation. 

    <img src="plots/example_3D.png" alt="3D" width="300"/>