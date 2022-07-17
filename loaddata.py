import csv

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
                data["time"].append(row[0])
                orientation = row[2:5] + [row[1]]
                data["orientation"].append(orientation)
                data["accel"].append(row[5:8])
                data["gyro"].append(row[8:11])
                data["magnetometer"].append(row[11:14])
    
    return data