import rotation_utils as Q
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

class CoordinateSystemArtist:
    def __init__(self, translation=(0., 0., 0), scale=1, text=None):
        self.translation = translation
        self.rotation = np.eye(3, dtype=np.float32)
        self.scale = scale
        self.xAxis = None
        self.yAxis = None
        self.zAxis = None
        self.x_trace_artist = None
        self.x_trace = []
        self.text_artist = None
        self.text = text

    def update_rotation(self, rotmat):
        self.rotation = rotmat
        self.x_trace.insert(0, self.translation + rotmat[:, 0])
        self.x_trace = self.x_trace[:30]

    def artists(self):
        return [self.xAxis,
                self.yAxis,
                self.zAxis,
                self.x_trace_artist,
                self.text_artist]

    def init(self, ax):
        self.xAxis = ax.plot3D([], [], [], color="r", linewidth=2)[0]
        self.yAxis = ax.plot3D([], [], [], color="g", linewidth=2)[0]
        self.zAxis = ax.plot3D([], [], [], color="b", linewidth=2)[0]
        self.x_trace_artist = ax.plot3D([], [], [], color="magenta", linewidth=2)[0]
        x,y,z = self.translation
        self.text_artist = ax.text(x, y, z, self.text, "x")

        return self.artists()

    def update(self, i):
        origin = np.array(self.translation)
        rotation = np.array(self.rotation)
        self.xAxis.set_data_3d(*zip(*[origin, origin + self.scale*rotation[:, 0]]))
        self.yAxis.set_data_3d(*zip(*[origin, origin + self.scale*rotation[:, 1]]))
        self.zAxis.set_data_3d(*zip(*[origin, origin + self.scale*rotation[:, 2]]))
        self.x_trace_artist.set_data_3d(*zip(*self.x_trace))

        return self.artists()


class CSAnimation:
    def __init__(self):
        pass

    def animateDataset(self, data, reference_data):
        self.data = data
        self.ref_data = reference_data
        self.fig = plt.figure()

        # 3D plot
        self.ax = self.fig.gca(projection='3d')
        ax = self.ax
        ax.clear()
        self.cs = CoordinateSystemArtist(translation=(0., -1, 0), text="Estimated")
        self.ref_cs = CoordinateSystemArtist(translation=(0., 1, 0), text="True")

        size = 2
        ax.set_xlim3d(-size, size)
        ax.set_ylim3d(-size, size)
        ax.set_zlim3d(-size, size)

        self.startTime = 0
        self.elapsed = 0

        # need to store FuncAnimation otherwise it doesn't work
        self.anim = animation.FuncAnimation(self.fig, 
                                            self.animate, 
                                            frames=self.timeGen, 
                                            init_func=self.init,
                                            interval=100, 
                                            blit=True)

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

        closestIndex = np.argmin(abs(np.array(self.data["time"], dtype=np.float32) - self.elapsed))
        closestIndex = min(closestIndex, len(self.data["time"])-1)

        self.ax.set_title("fps: {}, time: {}/{} sec".format(round(i/self.elapsed), round(float(self.data["time"][closestIndex]), 1), round(float(self.data["time"][-1]), 1)))
        
        self.cs.update_rotation(Q.quat_to_matrix(np.array(self.data["orientation"][closestIndex])))
        self.ref_cs.update_rotation(Q.quat_to_matrix(np.array(self.ref_data["orientation"][closestIndex])))
        
        self.fig.canvas.draw() # bug fix
        
        return self.cs.update(i) + self.ref_cs.update(i)
