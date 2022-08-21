import numpy as np
import time
import rotation_utils as Q
import matplotlib.pyplot as plt


class ESKF:
    def __init__(self, q_init=None, g=9.8255):
        
        # Nominal state
        if q_init is not None:
            self.q = np.array(q_init)
        else:
            self.q = np.array([0., 0, 0, 1])

        self.q_init = self.q.copy()

        self.g = g
        self.Qi = np.eye(3)*5.8e-4**2
        self.RR = np.eye(3)*8.5e-2 * self.g**2
        self.P = np.eye(3)*0

        self.P_init = self.P.copy()

    def reset(self):
        # reset
        self.q = self.q_init.copy()
        self.P = self.P_init.copy()

    def state(self):
        return np.array(list(self.q))

    def transition_matrix(self, w, dt):
        return Q.rodrigues(-w*dt)

    def noise_transition_matrix(self):
        return np.eye(3)

    def measurement_matrix(self, g):
        rotmat = Q.quat_to_matrix(self.q)
        H = Q.skew(np.matmul(rotmat.transpose(), g))
        
        return H

    def predict(self, w, dt):
        # Propagate error state
        Fx = self.transition_matrix(w, dt)
        Fi = self.noise_transition_matrix()
        # self.dx = Fx*self.dx + Fi*self.dx # Propagate mean (always zero)
        # Propagate covariance
        self.P = Fx.dot(self.P).dot(Fx.transpose()) + Fi.dot(self.Qi).dot(Fi.transpose())

        # Propagate nominal state
        #self.q = Q.quat_mult(self.q, np.array(list(0.5*w*dt) + [1])) # might work better sometimes
        self.q = Q.quat_mult(self.q, Q.rotvec_to_quat(w*dt))

    def update(self, a):
        g = np.array([0, 0, self.g])

        H = self.measurement_matrix(g)

        num = np.matmul(self.P, H.transpose())
        den = np.matmul(np.matmul(H, self.P), H.transpose()) + self.RR
        den = np.linalg.inv(den)
        K = np.matmul(num, den)

        residual = a - Q.quat_to_matrix(self.q).transpose().dot(g)
        d_theta = np.matmul(K, residual)

        #self.q = Q.quat_mult(self.q, np.array(list(0.5*d_theta) + [1])) # might work better sometimes
        self.q = Q.quat_mult(self.q, Q.rotvec_to_quat(d_theta))
        
        # Normalize quaternion, normally only needed when using approximate exp_quat
        self.q = Q.quat_norm(self.q)

        # Propagate covariance
        self.P = self.P - np.matmul(np.matmul(K, H), self.P)

        # ESKF reset
        G = np.eye(3) - Q.skew(0.5 * d_theta)
        #G = Q.rodrigues(-d_theta/2.0)
        self.P = np.matmul(np.matmul(G, self.P), G.transpose())

        return residual

    def generateOrientationData(self, data, update=True):
        self.reset()

        orientations = []
        covariances = []
        residuals = []

        start = time.time()
        t_prev = 0
        for i, t in enumerate(data["time"]):
            dt = float(t)-float(t_prev)
            t_prev = float(t)

            orientations.append(self.q.copy())
            covariances.append(self.P.copy())

            w = np.array(list(map(float, data["gyro"][i])))
            self.predict(w, dt)

            if update:
                a = np.array(data["accel"][i])
                residual = self.update(a)
                if residual is None:
                    residual = (0, 0, 0)
                else:
                    residuals.append(residual)

        elapsed = time.time() - start
        avg_time = elapsed / len(data["time"])
        print("Avg time: {} ({} Hz)".format(avg_time, 1./avg_time))
            
        return np.array(orientations), np.array(residuals), np.array(covariances)


if __name__ == "__main__":
    data = {"time": [0.1*i for i in range(100)],
            "gyro": [ [0.1, 0.1, 0.1] for _ in range(100) ]}

    kf = ESKF()
    newData = kf.generateOrientationData(data, update=False)
    print(kf.state())

    plt.plot(data["time"], data["gyro"])
    plt.show()
