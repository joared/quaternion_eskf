import matplotlib.pyplot as plt
from cube import Cube
from kalman import ESKF, ESKFG
from coordinate_system import CoordinateSystemArtist, CoordinateSystem

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import time
from loaddata import loadAndroid, loadDataset, load_sensorstream, load_custom

def analyze_sensor_data(data):
    g_data = np.array(data["gyro"])
    a_data = np.array(data["accel"])

    g_mean = np.mean(g_data, axis=0)
    g_std = np.std(g_data, axis=0)
    g_var = np.std(g_data, axis=0)

    dt = .01
    print("STD:", g_std)

    a_mean = np.mean(a_data, axis=0)
    a_std = np.std(a_data, axis=0)

    print("Gyro mean:", g_mean)
    print("Gyro std:", g_std)
    print("Accel mean:", a_mean)
    print("Accel std:", a_std)

    
    plt.subplot(2, 3, 1)
    plt.plot(g_data[:, 0], c="r")
    plt.subplot(2, 3, 2)
    plt.plot(g_data[:, 1], c="g")
    plt.subplot(2, 3, 3)
    plt.plot(g_data[:, 2], c="b")
    plt.subplot(2, 3, 4)
    plt.plot(a_data[:, 0], c="r")
    plt.subplot(2, 3, 5)
    plt.plot(a_data[:, 1], c="g")
    plt.subplot(2, 3, 6)
    plt.plot(a_data[:, 2], c="b")
    plt.show()

    return g_mean, g_std, a_mean, a_std

def calc_orientation_error(refOrientation, estOrientation):
    error_quats = []
    error_norms = []
    error_ndq = []
    for est, true in zip(estOrientation, refOrientation):
        #eulerErrors.append(np.linalg.norm(true-est))
        err_quat = list(np.array(true) - np.array(est))

        est_inv = R.from_quat(list(-est[:3]) + [est[3]])
        true = R.from_quat(true)
        
        err = true*est_inv
        q = err.as_quat()
        
        # 1/2 * Geodesic on the unit sphere
        #v = q[:3]
        #dq = q[3]
        #theta = np.arctan2(np.linalg.norm(v), dq)

        # Geodesic on the unit sphere
        theta = np.linalg.norm(err.as_rotvec())

        error_quats.append(err_quat)
        error_ndq.append(np.linalg.norm(err_quat))
        error_norms.append(theta)

    error_norms = np.array(error_norms)
    error_quats = np.array(error_quats)
    error_ndq = np.array(error_ndq)

    return error_norms, error_ndq, error_quats


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

        # old
        dt = float(self.data["time"][1]) - float(self.data["time"][0])
        closestIndex = min(int(self.elapsed/dt), len(self.data["time"])-1)
        # new
        #closestIndex = np.argmin(abs(np.array(self.data["time"], dtype=np.float32) - self.elapsed))

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
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset', help='Dataset (1-6 or custom)')
    parser.add_argument('--update', "-u", help='Enable accelerometer update', action='store_true')

    args = parser.parse_args()
    custom = False
    dataset_idx = None
    try:
        dataset_idx = int(args.dataset)-1
    except:
        if args.dataset == "custom":
            custom = True
        else:
            raise Exception("Invalid dataset '{}'".format(args.dataset))
    else:
        assert 0 <= dataset_idx <= 5, "Dataset has to be 1-6 (or custom)"

    datasets = ['dataset/TStick_Test01_Static.csv',
                'dataset/TStick_Test02_Trial1.csv',
                'dataset/TStick_Test02_Trial2.csv',
                'dataset/TStick_Test11_Trial1.csv',
                'dataset/TStick_Test11_Trial2.csv',
                'dataset/TStick_Test11_Trial3.csv'
                ]

    if custom:
        dataset = "own_dataset/rec4.csv", "own_dataset/rec4.log"
        loader = load_custom # loadDataset
    else:
        dataset = (datasets[dataset_idx],)
        loader = loadDataset

    refData = loader(*dataset)
    print("Data length:", len(refData["time"]))

    g_mean, g_std, a_mean, a_std = analyze_sensor_data(refData)
    
    newData = refData.copy()
    
    q_init = refData["orientation"][0]
    kf = ESKFG([
               # q
               #0.028, -0.009, 0., 0.999,
               q_init[0], q_init[1], q_init[2], q_init[3], 
               # wb
               0, 0., 0,
               #0.00367097, -0.00232872, 0.00098314,
               # ab
               #-0.01191148, 0.19411188, 0
               0., 0., 0.
              ]) 
    kf = ESKF(q_init)
    states, residuals, covariances, update_on = kf.generateOrientationData(refData, update=args.update)
    newOrientation = states[:, :4]
    error_norms, error_ndq, error_quats = calc_orientation_error(refData["orientation"], newOrientation)

    print("Est. state:", kf.q)
    print("Ref. orientation:", refData["orientation"][-1])
    newData["orientation"] = newOrientation
    print("Mean err:", np.mean(error_norms))
    print("Max err:", np.max(error_norms))
    print("Mean NDQ:", np.mean(error_ndq))
    print("Max NDQ:", np.max(error_ndq))

    bias = None
    accel_bias = None
    gravity = None
    accel_offset = None
    if states.shape[1] > 4:
        bias = states[:, 4:7]
    if states.shape[1] > 7:
        accel_bias = states[:, 7:10]
    if states.shape[1] > 11:
        gravity = states[:, 10:13]
    if states.shape[1] > 14:
        accel_offset = states[:, 13:16]

    rows = 2
    cols = 2
    linewidth = 1.5
    marker_true = "--"
    marker_est = "-"
    c_true = "g"
    c_est = "r"
    c_err = "y"
    lim = 1.05

    if len(error_norms) > 0:
        t = refData["time"]

        # Plot errors
        #fig = plt.figure()
        #plt.plot([np.linalg.norm(r) for r in residuals])
        fig = plt.figure()
        plt.plot(t, error_norms)
        if args.update:
            states_noupdate, _, _, _ = kf.generateOrientationData(refData, update=False)
            newOrientation_noupdate = states_noupdate[:, :4]
            error_norms_noupdate, _, _ = calc_orientation_error(refData["orientation"], newOrientation_noupdate)
            plt.plot(t, error_norms_noupdate)
            plt.legend([r"$|\Delta \theta|$ update", r"$|\Delta \theta|$"])
        else:
            plt.legend([r"$|\Delta \theta|$"])
        
        plt.ylim(-1, 1)
        plt.xlabel("sec")

        # Plot covariances
        rotated_covariances = []
        for s, cov in zip(states, covariances):
            rotmat = R.from_quat(s).as_matrix()
            r_cov = np.matmul(np.matmul(rotmat, cov), rotmat.transpose())
            rotated_covariances.append(r_cov)

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot([np.diag(cov) for cov in covariances])
        plt.subplot(1, 2, 2)
        plt.plot([np.diag(cov) for cov in rotated_covariances])

        # Plot quaternions components and quaternion error
        fig = plt.figure(figsize=(12, 10))
        fig.set_tight_layout(True)

        plt.subplot(rows, cols, 1)
        plt.plot(t, error_quats[:, 0], linewidth=linewidth, c=c_err)
        plt.plot(t, newOrientation[:, 0], marker_est, linewidth=linewidth, c=c_est)
        plt.plot(t, np.array(refData["orientation"])[:, 0], marker_true, linewidth=linewidth, c=c_true)
        plt.legend([r"$Error \; q_x$", r"$Est \; q_x$", r"$True \; q_x$"])
        plt.xlabel("sec")
        plt.ylim(-lim, lim)

        plt.subplot(rows, cols, 2)
        plt.plot(t, error_quats[:, 1], linewidth=linewidth, c=c_err)
        plt.plot(t, newOrientation[:, 1], marker_est, linewidth=linewidth, c=c_est)
        plt.plot(t, np.array(refData["orientation"])[:, 1], marker_true, linewidth=linewidth, c=c_true)
        plt.legend([r"$Error \; q_y$", r"$Est \; q_y$", r"$True \; q_y$"])
        plt.xlabel("sec")
        plt.ylim(-lim, lim)

        plt.subplot(rows, cols, 3)
        plt.plot(t, error_quats[:, 2], linewidth=linewidth, c=c_err)
        plt.plot(t, newOrientation[:, 2], marker_est, linewidth=linewidth, c=c_est)
        plt.plot(t, np.array(refData["orientation"])[:, 2], marker_true, linewidth=linewidth, c=c_true)
        plt.legend([r"$Error \; q_z$", r"$Est \; q_z$", r"$True \; q_z$"])
        plt.xlabel("sec")
        plt.ylim(-lim, lim)
                
        plt.subplot(rows, cols, 4)
        # plot quaternion
        plt.plot(t, error_quats[:, 3], linewidth=linewidth, c=c_err)
        plt.plot(t, newOrientation[:, 3], marker_est, linewidth=linewidth, c=c_est)
        plt.plot(t, np.array(refData["orientation"])[:, 3], marker_true, linewidth=linewidth, c=c_true)
        plt.legend([r"$Error \; q_w$", r"$Est \; q_w$", r"$True \; q_w$"])
        plt.xlabel("sec")
        plt.ylim(-lim, lim)

        #plt.tight_layout(pad=0, h_pad=0, w_pad=0)

        """
        plt.subplot(rows, cols, 5)
        plt.plot(error_vecs)
        plt.ylim(-.1, .1)

        plt.subplot(rows, cols, 6)
        #plt.plot([cov[0,0] for cov in covariances], c="r")
        #plt.plot([cov[1,1] for cov in covariances], c="g")
        #plt.plot([cov[2,2] for cov in covariances], c="b")

        plt.plot([np.linalg.norm(np.diag(cov)[:3], ord=2) for cov in covariances])
        #plt.plot([np.linalg.norm(np.diag(cov)[3:6], ord=2) for cov in covariances])
        #plt.plot([np.linalg.norm(np.diag(cov)[6:9], ord=2) for cov in covariances])
        #plt.plot([np.linalg.norm(np.diag(cov)[9:12], ord=2) for cov in covariances])
        #plt.plot([np.linalg.norm(np.diag(cov)[12:15], ord=2) for cov in covariances])
        #plt.ylim(-.1, .1)
        plt.subplot(rows, cols, 7)
        if bias is not None:
            plt.plot(bias[:, 0], c="r")
            plt.plot(bias[:, 1], c="g")
            plt.plot(bias[:, 2], c="b")

        plt.ylim(-.01, 0.01)

        plt.subplot(rows, cols, 8)
        if accel_bias is not None:
            plt.plot(accel_bias[:, 0], c="r")
            plt.plot(accel_bias[:, 1], c="g")
            plt.plot(accel_bias[:, 2], c="b")
        #plt.ylim(-.5, 0.5)

        plt.subplot(rows, cols, 9)
        if gravity is not None:
            plt.plot(gravity[:, 0], c="r")
            plt.plot(gravity[:, 1], c="g")
            plt.plot(gravity[:, 2], c="b")
        #plt.ylim(-10, 0.5)
        
        plt.subplot(rows, cols, 10)
        if accel_offset is not None:
            plt.plot(accel_offset[:, 0], c="r")
            plt.plot(accel_offset[:, 1], c="g")
            plt.plot(accel_offset[:, 2], c="b")
        #plt.plot(update_on)
        #plt.ylim(-2, 2)
        """
        plt.show()

    cani = CSAnimation()
    #cani = CubeAnimation()
    cani.animateDataset(newData, refData)
    cani.show()
