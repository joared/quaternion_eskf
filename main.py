import matplotlib.pyplot as plt
from cube import Cube
from kalman import Kalman
from coordinate_system import CoordinateSystemArtist, CoordinateSystem

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import time
from loaddata import loadAndroid

def loadDataset(filename):
    import csv
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

class CSAnimation:
    def __init__(self):
        pass

    def animateDataset(self, data, reference_data):
        self.data = data
        self.ref_data = reference_data
        self.fig = plt.figure()

        self.ax = self.fig.gca(projection='3d')
        ax = self.ax
        ax.clear()
        self.cs = CoordinateSystemArtist(CoordinateSystem(translation=(0., -1, 0)))
        self.ref_cs = CoordinateSystemArtist(CoordinateSystem(translation=(0., 1, 0)))

        size = 2
        ax.set_xlim3d(-size, size)
        ax.set_ylim3d(-size, size)
        ax.set_zlim3d(-size, size)

        self.startTime = 0
        self.elaspsed = 0

        # need to store FuncAnimation otherwise it doesn't work
        self.anim = animation.FuncAnimation(self.fig, self.animate, frames=self.timeGen, init_func=self.init,
                                       interval=100, blit=True)

    def show(self):
        plt.show()

    def init(self):
        return self.cs.init(self.ax) + self.ref_cs.init(self.ax)

    def timeGen(self):
        for i in range(len(self.data["time"])):
            yield i

    def animate(self, i):
        if i == 0:
            self.startTime = time.time()
        self.elapsed = time.time() - self.startTime

        dt = float(self.data["time"][1]) - float(self.data["time"][0])
        closestIndex = min(int(self.elapsed/dt), len(self.data["time"])-1)
        self.ax.set_title("fps: {}, time: {}/{}".format(round(i/self.elapsed), round(float(self.data["time"][closestIndex]), 1), round(float(self.data["time"][-1]))))
        
        #closestIndex = i
        #self.ax.set_title("Frame: {}".format(i))
        
        self.cs.cs.rotation = R.from_quat(self.data["orientation"][closestIndex]).as_matrix()

        self.ref_cs.cs.rotation = R.from_quat(self.ref_data["orientation"][closestIndex]).as_matrix()
        
        self.fig.canvas.draw() # bug fix
        
        return self.cs.update(i) + self.ref_cs.update(i)

class CubeAnimation:
    def __init__(self):
        pass

    def animateDataset(self, data, referenceData=None):
        
        self.data = data
        self.refData = referenceData
        self.fig = plt.figure()

        self.ax = self.fig.gca(projection='3d')
        ax = self.ax
        ax.clear()
        self.cube = Cube(ax, (0, -2, 0))
        self.cs = CoordinateSystemArtist(CoordinateSystem(translation=(0., -2, 0)))
        self.refCube = None
        if referenceData is not None:
            self.ref_cs = CoordinateSystemArtist(CoordinateSystem(translation=(0., 22, 0)))
            self.refCube = Cube(ax, (0, 2, 0))
        else:
            staticCube = Cube(ax, (0, 2, 0))
            staticCube.draw()

        size = 3
        ax.set_xlim3d(-size, size)
        ax.set_ylim3d(-size, size)
        ax.set_zlim3d(-size, size)

        self.startTime = 0
        self.elaspsed = 0

        # need to store FuncAnimation otherwise it doesn't work
        self.anim = animation.FuncAnimation(self.fig, self.animate, frames=self.timeGen, init_func=self.init,
                                       interval=100, blit=True)

    def show(self):
        plt.show()

    def init(self):
        if self.refCube:
            return self.cube.init() + self.refCube.init()
        return self.cube.init()

    def timeGen(self):
        for i in range(len(self.data["time"])):
            yield i

    def animate(self, i):
        if i == 0:
            self.startTime = time.time()
        self.elapsed = time.time() - self.startTime

        dt = float(self.data["time"][1]) - float(self.data["time"][0])
        closestIndex = min(int(self.elapsed/dt), len(self.data["time"])-1)
        self.ax.set_title("fps: {}, time: {}/{}".format(round(i/self.elapsed), round(float(self.data["time"][closestIndex]), 1), round(float(self.data["time"][-1]))))
        
        #closestIndex = i
        #self.ax.set_title("Frame: {}".format(i))
        
        self.cube.rotate(R.from_quat(self.data["orientation"][closestIndex]).as_matrix())
        if self.refCube:
            self.refCube.rotate(R.from_quat(self.refData["orientation"][closestIndex]).as_matrix())
        
        self.fig.canvas.draw() # bug fix
        
        if self.refCube:
            return self.cube.update(i) + self.refCube.update(i)
        return self.cube.update(i)

if __name__ == "__main__":
    datasets = ['dataset/TStick_Test01_Static.csv',
                'dataset/TStick_Test11_Trial1.csv',
                'dataset/TStick_Test11_Trial2.csv',
                'dataset/TStick_Test11_Trial3.csv',
                'dataset/TStick_Test02_Trial1.csv',
                'dataset/TStick_Test02_Trial2.csv']

    refData = loadDataset(datasets[2])
    #refData = loadAndroid("android_1")
    newData = refData.copy()
    
    kf = Kalman()
    newOrientation, residuals, covariances = kf.generateOrientationData(refData, 
                                                                        update=True, 
                                                                        #q_offset=np.array([0.028, -0.009, 0., 0.999])
                                                                        )

    eulerErrors = []
    for est, true in zip(newOrientation, refData["orientation"]):
        #est = R.from_quat(est)
        #true = R.from_quat(true)
        
        #err = true.inv()*est
        eulerErrors.append(np.linalg.norm(true-est))
        #eulerErrors.append(err.as_euler("ZYX"))

    eulerErrors = np.array(eulerErrors)
    
    #refData["orientation"] = newOrientation

    print("Est. state:", kf.q)
    print("Ref. orientation:", refData["orientation"][-1])
    newData["orientation"] = newOrientation
    print("Mean NDQ:", np.mean(eulerErrors))
    print("Max NDQ:", np.max(eulerErrors))

    if len(eulerErrors) > 0:
        plt.figure()
        
        plt.subplot(4, 2, 1)
        # plot quaternion
        plt.plot(np.array(refData["orientation"])[:, 3])
        plt.plot(newOrientation[:, 3], c="y")
        
        plt.subplot(4, 2, 2)
        plt.plot(np.array(refData["orientation"])[:, 0])
        plt.plot(newOrientation[:, 0], c="r")
        
        plt.subplot(4, 2, 3)
        plt.plot(np.array(refData["orientation"])[:, 1])
        plt.plot(newOrientation[:, 1], c="g")
        
        plt.subplot(4, 2, 4)
        plt.plot(np.array(refData["orientation"])[:, 2])
        plt.plot(newOrientation[:, 2], c="b")
        
        plt.subplot(4, 2, 5)
        plt.plot(eulerErrors)
        #plt.plot(eulerErrors[:, 0], c="b")
        #plt.plot(eulerErrors[:, 1], c="g")
        #plt.plot(eulerErrors[:, 2], c="r")
        #plt.ylim(-.05, .05)
        plt.subplot(4, 2, 6)
        #plt.plot(residuals[:, 0], c="r")
        #plt.plot(residuals[:, 1], c="g")
        #plt.plot(residuals[:, 2], c="b")
        #plt.plot([float(ax) for ax, _, _ in refData["accel"]], c="r")
        #plt.plot([float(ay) for _, ay, _ in refData["accel"]], c="g")
        #plt.plot([float(az) for _, _, az in refData["accel"]], c="b")

        plt.plot([np.linalg.norm(cov, ord=2) for cov in covariances])
        #plt.ylim(-.1, .1)

        #print(np.std([float(ax) for ax, _, _ in refData["accel"]]))
        #print(np.std([float(ay) for ay, _, _ in refData["accel"]]))
        #print(np.std([float(az) for az, _, _ in refData["accel"]]))
        plt.show()

    cani = CSAnimation()
    cani.animateDataset(newData, refData)
    cani.show()
