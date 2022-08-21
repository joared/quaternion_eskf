import numpy as np
import matplotlib.pyplot as plt
from loaddata import loadDataset

# Config. params
DATA_FILE = 'dataset/TStick_Test01_Static.csv'  # CSV data file "gx,gy,gz"
fs = 100  # Sample rate [Hz]

def allan_deviation(dataArr: np.ndarray, fs: float, maxNumM: int=100):
    """Compute the Allan deviation (sigma) of time-series data.

    Algorithm obtained from Mathworks:
    https://www.mathworks.com/help/fusion/ug/inertial-sensor-noise-analysis-using-allan-variance.html

    Args
    ----
        dataArr: 1D data array
        fs: Data sample frequency in Hz
        maxNumM: Number of output points
    
    Returns
    -------
        (taus, allanDev): Tuple of results
        taus (numpy.ndarray): Array of tau values
        allanDev (numpy.ndarray): Array of computed Allan deviations
    """
    ts = 1.0 / fs
    N = len(dataArr)
    Mmax = 2**np.floor(np.log2(N / 2))
    M = np.logspace(np.log10(1), np.log10(Mmax), num=maxNumM)
    M = np.ceil(M)  # Round up to integer
    M = np.unique(M)  # Remove duplicates
    taus = M * ts  # Compute 'cluster durations' tau

    # Compute Allan variance
    allanVar = np.zeros(len(M))
    for i, mi in enumerate(M):
        twoMi = int(2 * mi)
        mi = int(mi)
        allanVar[i] = np.sum(
            (dataArr[twoMi:N] - (2.0 * dataArr[mi:N-mi]) + dataArr[0:N-twoMi])**2
        )
    
    allanVar /= (2.0 * taus**2) * (N - (2.0 * M))
    return (taus, np.sqrt(allanVar))

# Load CSV into np array
data = loadDataset(DATA_FILE)

ts = 1.0 / fs

# Separate into arrays
gx = np.array(data["gyro"])[:, 0] * (180.0 / np.pi)  # [deg/s]
gy = np.array(data["gyro"])[:, 1] * (180.0 / np.pi)
gz = np.array(data["gyro"])[:, 2] * (180.0 / np.pi)

# Calculate gyro angles
thetax = np.cumsum(gx) * ts  # [deg]
thetay = np.cumsum(gy) * ts
thetaz = np.cumsum(gz) * ts

# Compute Allan deviations
(taux, adx) = allan_deviation(thetax, fs, maxNumM=200)
(tauy, ady) = allan_deviation(thetay, fs, maxNumM=200)
(tauz, adz) = allan_deviation(thetaz, fs, maxNumM=200)

# Plot data on log-scale
plt.figure()
plt.title('Gyro Allan Deviations')
plt.plot(taux, adx, label='gx')
plt.plot(tauy, ady, label='gy')
plt.plot(tauz, adz, label='gz')
plt.xlabel(r'$\tau$ [sec]')
plt.ylabel('Deviation [deg/sec]')
plt.grid(True, which="both", ls="-", color='0.65')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()