import csv
import numpy as np
    

def load_custom(groundtruthfile, sensorfile):
    data = {"time": [],
            "orientation": [],
            "accel": [],
            "gyro": [],
            "magnetometer": []}

    with open(groundtruthfile, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i > 6:
                data["time"].append(float(row[1]))
                if row[2]:
                    
                    q = list(map(float, row[2:6]))

                    # re-arange quaternion components
                    q = [q[3], q[0], q[1], q[2]]
                    q = [q[1], -q[3], -q[0], q[2]]

                    data["orientation"].append(q)
                else:
                    data["orientation"].append(data["orientation"][-1])

    with open(sensorfile, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if row and row[0] == "1":
                accel = np.array(row[1:4], dtype=np.float32)
                accel *= 10./8192
                data["accel"].append(list(accel))

                gyro = np.array(row[4:7], dtype=np.float32)
                gyro /= 16.4
                gyro *= np.pi/180
                data["gyro"].append(list(gyro))

    # Align/synchronize time
    temp = 0
    start = 3158 +temp
    end = 6688 + temp
    data["accel"] = data["accel"][start:end]
    data["gyro"] = data["gyro"][start:end]
    l_s = len(data["accel"])

    pad = 200
    data["time"] = data["time"][1999:5629+pad]
    data["orientation"] = data["orientation"][1999:5629+pad]
    l_q = len(data["time"])

    c = l_q/l_s

    time_arr = []
    orientation_arr = []

    for index in range(l_s):
        time_arr.append(data["time"][int(index*c)])
        orientation_arr.append(data["orientation"][int(index*c)])

    data["time"] = list(np.array(time_arr)-time_arr[0])
    data["orientation"] = orientation_arr

    return data


def load_repoimu(filename):
    data = {"time": [],
            "orientation": [],
            "accel": [],
            "gyro": [],
            "magnetometer": []}

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(reader):
            if i > 1:
                data["time"].append(float(row[0]))
                orientation = list(map(float, row[2:5])) + [float(row[1])]
                data["orientation"].append(orientation)
                data["accel"].append(list(map(float, row[5:8])))
                data["gyro"].append(list(map(float, row[8:11])))
                data["magnetometer"].append(list(map(float, row[11:14])))

    return data
