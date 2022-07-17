from scipy.spatial.transform import Rotation as R

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from itertools import product, combinations

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d



class Cube:
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, ax, offset=(0,0,0)):
        self.ax = ax
        self.points = [ [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1], 
                        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1] ]

        self._rotation = [ [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self._offset = offset

        self.patches = []

    def draw2(self, ax=None):
        if ax: self.ax = ax
        ax = self.ax
        colors = ["r", "r", "black", "g", "black", "g"]
        points = self.rotatedPoints()
        for i, side in enumerate( [[0,1,2,3], [4,5,6,7], [0,3,7,4], [3,2,6,7], [1,2,6,5], [0,1,5,4]] ):
            ax.plot3D(*zip(points[side[0]], points[side[1]]), color="b")
            ax.plot3D(*zip(points[side[1]], points[side[2]]), color="b")
            ax.plot3D(*zip(points[side[2]], points[side[3]]), color="b")
            ax.plot3D(*zip(points[side[3]], points[side[0]]), color="b")
            ax.plot3D(*zip(points[side[0]], points[side[2]]), color=colors[i])
            ax.plot3D(*zip(points[side[1]], points[side[3]]), color=colors[i])

    def draw(self, ax=None):
        if ax: self.ax = ax
        ax = self.ax
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        points = self.rotatedPoints()
        for i, side in enumerate( [[0,1,2,3], [4,5,6,7], [0,3,7,4], [3,2,6,7], [1,2,6,5], [0,1,5,4]] ):
            rSide = [points[side[0]],
                    points[side[1]],
                    points[side[2]],
                    points[side[3]]]

            side = art3d.Poly3DCollection([rSide]) # important to send in a list for some reason
            side.set_color(colors[i])
            ax.add_collection3d(side)

    def init(self):
        ax = self.ax
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        points = self.rotatedPoints()
        self.patches = []
        for i, side in enumerate( [[0,1,2,3], [4,5,6,7], [0,3,7,4], [3,2,6,7], [1,2,6,5], [0,1,5,4]] ):
            rSide = [points[side[0]],
                    points[side[1]],
                    points[side[2]],
                    points[side[3]]]
            side = art3d.Poly3DCollection([rSide]) # important to send in a list for some reason
            #for i in dir(side):
                #print(i)
            #input()
            side.set_color(colors[i])
            side._facecolors2d = side._facecolor3d
            side._edgecolors2d = side._edgecolor3d
            #side._facecolors2d= ((0.5, 0.5, 0.5, 0.5),(0.5, 0.5, 0.5, 0.5))
            #side._edgecolors2d= ((0.5, 0.5, 0.5, 0.5),(0.5, 0.5, 0.5, 0.5))
            something = ax.add_collection3d(side)
            self.patches.append(side)

        return self.patches

    def update(self, i):
        #https://stackoverflow.com/questions/48794016/animate-a-collection-of-patches-in-matplotlib
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        points = self.rotatedPoints()
        for i, side in enumerate( [[0,1,2,3], [4,5,6,7], [0,3,7,4], [3,2,6,7], [1,2,6,5], [0,1,5,4]] ):
            rSide = [points[side[0]],
                    points[side[1]],
                    points[side[2]],
                    points[side[3]]]

            self.patches[i].set_verts([rSide]) # important to send in a list for some reason
            #https://stackoverflow.com/questions/66113220/how-to-animate-poly3dcollection-using-funcanimation-with-blit-true
            self.patches[i].do_3d_projection(self.patches[i].axes.get_figure().canvas.get_renderer()) # bug fix
        return self.patches

    def rotate(self, r):
        self._rotation = r
        #self._rotation = np.matmul(r, self._rotation)
        #print(self._rotation)

    def rotatedPoints(self):
        rotatedPoints = []
        for p in self.points:
            rotatedPoints.append(np.matmul(self._rotation, np.array(p)))

        points = []
        for p in rotatedPoints:
            points.append( [p1+p2 for p1,p2 in zip(p, self._offset)] )

        return points


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cube = Cube(ax, (0, -2, 0))
    staticCube = Cube(ax, (0, 2, 0))


    a1, a2, a3 = 0, 0, 0
    for i in range(1000):
        a1 += 0.01
        a2 += 0.03
        a3 += 0.01

        r = R.from_euler("XYZ", (a1, a2, a3))
        ax.clear()
        
        size = 3
        ax.set_xlim3d(-size, size)
        ax.set_ylim3d(-size, size)
        ax.set_zlim3d(-size, size)
        ax.set_title("Time: {}".format(i))

        cube.rotate(r.as_matrix())
        staticCube.draw2()
        cube.draw()
        plt.pause(0.001)

    plt.show()

