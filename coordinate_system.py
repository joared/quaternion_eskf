from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

class CoordinateSystem:
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, translation=(0,0,0), euler=(0, 0, 0)):
        r = R.from_euler("XYZ", euler)
        self.rotation = r.as_matrix() # depricated -> as_matrix. Just to make it work with py2
        self.initialRotation = r.as_matrix() # depricated -> as_matrix. Just to make it work with py2
        self.translation = list(translation)
        self.initialTranslation = list(translation)

    def reset(self):
        self.translation = self.initialTranslation.copy()
        self.rotation = self.initialRotation

    def setTransform(self, trans, rot):
        self.translation = [_ for _ in trans]
        self.rotation = [_ for _ in rot]

    def transformedPoints(self, points, referenceTranslation=(0,0,0)):
        """
        Local to global
        """
        transformedPoints = []
        for p in points:
            transformedPoints.append(np.matmul(self.rotation, np.array(p)))

        newPoints = []
        for p in transformedPoints:
            newPoints.append( [p1+p2-p3 for p1,p2,p3 in zip(p, self.translation, referenceTranslation)] )

        return newPoints

    def transformedPointsInv(self, points, referenceTranslation=(0,0,0)):
        """
        Global to local
        """
        points = np.array(points) - np.array(self.translation)
        newPoints = list(np.matmul(np.linalg.inv(np.array(self.rotation)), points.transpose()).transpose())
        #points = np.array(points) - np.array(self.translation)
        #points = np.matmul(np.linalg.inv(np.array(self.rotation)), np.array(points).transpose()).transpose()

        return list(newPoints)

class CoordinateSystemArtist:
    def __init__(self, coordinateSystem=None, scale=1, text=None):
        self.cs = coordinateSystem if coordinateSystem else CoordinateSystem()
        self.scale = scale
        self.xAxis = None
        self.yAxis = None
        self.zAxis = None
        self.text_artist = None
        self.text = text

    def draw(self, ax, referenceTranslation=(0,0,0), colors=("r", "g", "b"), scale=1, alpha=1):
        origin = np.array(self.cs.translation) - np.array(referenceTranslation)
        rotation = np.array(self.cs.rotation)
        self.scale = scale
        ax.plot3D(*zip(*[origin, origin + self.scale*rotation[:, 0]]), color=colors[0], linewidth=2, alpha=alpha)
        ax.plot3D(*zip(*[origin, origin + self.scale*rotation[:, 1]]), color=colors[1], linewidth=2, alpha=alpha)
        ax.plot3D(*zip(*[origin, origin + self.scale*rotation[:, 2]]), color=colors[2], linewidth=2, alpha=alpha)

        return self.artists()

    def drawRelative(self, ax, cs, colors=("r", "g", "b"), scale=1, alpha=1):
        r = R.from_dcm(cs.rotation)
        origin = cs.translation + r.apply(self.cs.translation)
        rotation = np.matmul(r.as_matrix(), self.cs.rotation)
        self.scale = scale
        ax.plot3D(*zip(*[origin, origin + self.scale*rotation[:, 0]]), color=colors[0], linewidth=2, alpha=alpha)
        ax.plot3D(*zip(*[origin, origin + self.scale*rotation[:, 1]]), color=colors[1], linewidth=2, alpha=alpha)
        ax.plot3D(*zip(*[origin, origin + self.scale*rotation[:, 2]]), color=colors[2], linewidth=2, alpha=alpha)

        return self.artists()

    def artists(self):
        return [self.xAxis,
                self.yAxis,
                self.zAxis,
                self.text_artist]

    def init(self, ax):
        self.xAxis = ax.plot3D([], [], [], color="r", linewidth=2)[0]
        self.yAxis = ax.plot3D([], [], [], color="g", linewidth=2)[0]
        self.zAxis = ax.plot3D([], [], [], color="b", linewidth=2)[0]
        x,y,z = self.cs.translation
        self.text_artist = ax.text(x, y, z, self.text, "x")

        return self.artists()

    def update(self, show=True, referenceTranslation=(0,0,0)):
        if show:
            origin = np.array(self.cs.translation) - np.array(referenceTranslation)
            rotation = np.array(self.cs.rotation)
            self.xAxis.set_data_3d(*zip(*[origin, origin + self.scale*rotation[:, 0]]))
            self.yAxis.set_data_3d(*zip(*[origin, origin + self.scale*rotation[:, 1]]))
            self.zAxis.set_data_3d(*zip(*[origin, origin + self.scale*rotation[:, 2]]))
        else:
            self.xAxis.set_data_3d([], [], [])
            self.yAxis.set_data_3d([], [], [])
            self.zAxis.set_data_3d([], [], [])

        return self.artists()

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
        self.cs = CoordinateSystemArtist(CoordinateSystem(translation=(0., -1, 0)), text="Estimated")
        self.ref_cs = CoordinateSystemArtist(CoordinateSystem(translation=(0., 1, 0)), text="True")

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

        #dt = float(self.data["time"][1]) - float(self.data["time"][0])
        #closestIndex = min(int(self.elapsed/dt), len(self.data["time"])-1)
        # new
        closestIndex = np.argmin(abs(np.array(self.data["time"], dtype=np.float32) - self.elapsed))
        closestIndex = min(closestIndex, len(self.data["time"])-1)

        self.ax.set_title("fps: {}, time: {}/{}".format(round(i/self.elapsed), round(float(self.data["time"][closestIndex]), 1), round(float(self.data["time"][-1]))))
        
        #closestIndex = i
        #self.ax.set_title("Frame: {}".format(i))
        
        self.cs.cs.rotation = R.from_quat(self.data["orientation"][closestIndex]).as_matrix()

        self.ref_cs.cs.rotation = R.from_quat(self.ref_data["orientation"][closestIndex]).as_matrix()
        
        self.fig.canvas.draw() # bug fix
        
        return self.cs.update(i) + self.ref_cs.update(i)
