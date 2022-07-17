import csv

def loadAndroid(filename):
    data = {"time": [],
            "orientation": [],
            "accel": [],
            "gyro": [],
            "magnetometer": []}

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(reader):
            if i > 1:
                data["time"].append(row[0])
                data["accel"].append(list(map(float, row[1:4])))
                data["gyro"].append(list(map(float, row[4:7])))
                #data["magnetometer"].append(row[11:14])
    
    return data

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R
    import numpy as np

    rotmat = R.from_euler("xyz", (np.pi/2, 0, 0))
    print(np.matmul(rotmat, [0, 0, 1]))

    data = loadAndroid("android_1")
    plt.plot([a[0] for a in data["accel"]], c="r")
    plt.plot([a[1] for a in data["accel"]], c="g")
    plt.plot([a[2] for a in data["accel"]], c="b")
    plt.show()
