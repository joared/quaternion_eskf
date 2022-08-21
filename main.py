import matplotlib.pyplot as plt
from kalman import ESKF
from coordinate_system import CSAnimation

import numpy as np
import rotation_utils as Q

from loaddata import load_repoimu, load_custom

def analyze_sensor_data(data):
    g_data = np.array(data["gyro"])
    a_data = np.array(data["accel"])

    g_mean = np.mean(g_data, axis=0)
    g_std = np.std(g_data, axis=0)

    a_mean = np.mean(a_data, axis=0)
    a_std = np.std(a_data, axis=0)

    print("Gyro mean:", g_mean)
    print("Gyro std:", g_std)
    print("Accel mean:", a_mean)
    print("Accel std:", a_std)
    print()
    
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
        err_quat = list(np.array(true) - np.array(est))

        q_err = Q.quat_mult(true, Q.quat_inv(est))
        w = Q.quat_to_rotvec(q_err)

        # Geodesic on the unit sphere
        theta = np.linalg.norm(w)

        error_quats.append(err_quat)
        error_ndq.append(np.linalg.norm(err_quat))
        error_norms.append(theta)

    error_norms = np.array(error_norms)
    error_quats = np.array(error_quats)
    error_ndq = np.array(error_ndq)

    return error_norms, error_ndq, error_quats


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
        loader = load_repoimu

    refData = loader(*dataset)
    print("Data length:", len(refData["time"]))

    g_mean, g_std, a_mean, a_std = analyze_sensor_data(refData)
    
    newData = refData.copy()
    
    q_init = refData["orientation"][0]
    kf = ESKF(q_init)

    states, residuals, covariances = kf.generateOrientationData(refData, update=args.update)
    newOrientation = states[:, :4]
    newData["orientation"] = newOrientation
    error_norms, error_ndq, error_quats = calc_orientation_error(refData["orientation"], newOrientation)

    print("Est. state:", kf.q)
    print("Ref. orientation:", refData["orientation"][-1])
    print("Mean err:", np.mean(error_norms))
    print("Max err:", np.max(error_norms))
    print("Mean NDQ:", np.mean(error_ndq))
    print("Max NDQ:", np.max(error_ndq))

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
        plt.plot(t, error_norms)
        if args.update:
            states_noupdate, _, _ = kf.generateOrientationData(refData, update=False)
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
            rotmat = Q.quat_to_matrix(s)
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
        plt.show()

    cani = CSAnimation()
    cani.animateDataset(newData, refData)
    cani.show()
