import csv
from scipy.spatial.transform import Rotation as R
import numpy as np
    

def load_custom(groundtruthfile, sensorfile):
    data = {"time": [],
            "orientation": [],
            "accel": [],
            "gyro": [],
            "magnetometer": []}

    init_q = None
    with open(groundtruthfile, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i > 6:
                data["time"].append(float(row[1]))
                if row[2]:
                    
                    q = list(map(float, row[2:6]))
                    #q = [-q[0], q[2], q[3], -q[1]] # re-arange quaternion
                    q = [q[3], q[0], q[1], q[2]]
                    q = [q[1], -q[3], -q[0], q[2]]
                    #if init_q is None:
                    #    init_q = q

                    #rot = R.from_quat(init_q).inv()*R.from_quat(q)
                    #q = rot.as_quat()
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

    
    temp = 0
    start = 3158 +temp
    end = 6688 + temp
    data["accel"] = data["accel"][start:end]
    data["gyro"] = data["gyro"][start:end]
    l_s = len(data["accel"])

    pad = 200 # 200

    data["time"] = data["time"][1999:5629+pad]
    data["orientation"] = data["orientation"][1999:5629+pad]
    l_q = len(data["time"])

    #print("Time elapsed:")
    #print(data["time"][-1] - data["time"][0])

    c = l_q/l_s

    time_arr = []
    orientation_arr = []

    for index in range(l_s):
        time_arr.append(data["time"][int(index*c)])
        orientation_arr.append(data["orientation"][int(index*c)])
    
    #print(len(time_arr))
    #print(len(orientation_arr))

    data["time"] = list(np.array(time_arr)-time_arr[0])
    data["orientation"] = orientation_arr
    
    """
    #l_s = len(data["orientation"]) 
    #data["accel"] = [[0., 0., 0.] for _ in range(l_s)]#data["accel"][5000:]
    #data["gyro"] = [[0., 0., 0.] for _ in range(l_s)]#data["gyro"][5000:]
    
    start = 3159
    end = 6689
    print(len(data["accel"]))
    data["accel"] = data["accel"][start:end]
    data["gyro"] = data["gyro"][start:end]
    l_s = len(data["accel"])

    data["time"] = list(np.array(range(l_s))*0.01)
    q_init = data["orientation"][0]
    data["orientation"] = [q_init for _ in range(l_s)]
    """

    #print(len(data["time"]))
    #print(len(data["orientation"]))
    #input()

    return data

def load_sensorstream(filename):
    data = {"time": [],
            "orientation": [],
            "accel": [],
            "gyro": [],
            "magnetometer": []}

    sensors = {
               3:"accel",
               4:"gyro",
               5:"magnetometer",
               84:"orientation"
              }

    # Add data with None values when no data exist
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            data["time"].append(float(row.pop(0)))
            
            
            for sensor_idx in (3,4,5,84):
                sensor = sensors[sensor_idx]
                try:
                    idx = int(row.pop(0))
                except:
                    data[sensor].append(None)
                    continue

                if idx == sensor_idx:
                    values = list(float(row.pop(0)) for _ in range(3))
                    if sensor == "orientation":
                        q = R.from_rotvec(np.array(values)*np.pi).as_quat()
                        data["orientation"].append(q)
                    else:
                        data[sensor].append(values)
                else:
                    data[sensor].append(None)

    print(len(data["time"]))
    print(len(data["gyro"]))
    print(len(data["orientation"]))

    # interpolate
    for sensor in data.keys():
        new_data = []

        for values in data[sensor]:
            if values is None and not new_data:
                if sensor == "orientation":
                    # orientation handled below
                    new_data.append(None)
                else:
                    new_data.append([0, 0, 0.])
            elif values is None:
                new_data.append(new_data[-1])
            else:
                new_data.append(values)
        
        data[sensor] = new_data

    
    init_q = next(item for item in data["orientation"] if item is not None)
    new_orientation = []
    for q in data["orientation"]:
        if q is None:
            new_orientation.append([0, 0, 0, 1.])
        else:
            rot = R.from_quat(init_q).inv()*R.from_quat(q)
            q = rot.as_quat()
            #if q[3] < 0:
            q = list(-np.array(q))
            new_orientation.append(q)
    data["orientation"] = new_orientation

    # adjust time to start at 0
    data["time"] = list(np.array(data["time"])-data["time"][0])

    

    return data


def loadDataset(filename):
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

def loadAndroid(filename):
    data = {"time": [],
            "orientation": [],
            "accel": [],
            "gyro": [],
            "magnetometer": []}

    initial_q = None

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(reader):
            if i > 1:
                data["time"].append(row[0])
                data["accel"].append(list(map(float, row[1:4])))
                data["gyro"].append(list(map(float, row[4:7])))
                data["magnetometer"].append(list(map(float, row[7:10])))
                print(len(row))
                #if row[10:3]:
                #    print(row[10:3])

                q = [0, 0, 0, 1]
                data["orientation"].append(q)


                
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
