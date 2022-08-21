from scipy.spatial.transform import Rotation as R
import scipy
import numpy as np
import time
import rotation_utils as Q
import matplotlib.pyplot as plt


class ESKFG:
    def __init__(self, 
                 init_state, 
                 gyro_bias=True,
                 accel_bias=True,
                 gravity=False,
                 accel_offset=True):
        
        # How to add state
        # 1) Add process noise (Q_i) and initial covariance (P)
        # 2) Modify state()
        # 3) Modify transition_matrix() and noise_transition_matrix()
        # 4) Modify measurement_matrix() (observation model)

        # Nominal state
        self.q = np.array(init_state[:4])
        self.wb = np.array(init_state[4:7])
        self.a = np.array(init_state[7:])
        self.g = np.array([0., 0., 1.]) #9.8255
        #self.a_offset_rot = R.from_quat([0.01031247, 0.00139185, 0.0845246, 0.99636705])
        self.a_offset = np.array([0, 0, 0.])

        # Error state
        #self.d_theta = np.array([0., 0, 0])
        
        ############# Tuning # 3 #############
        # Process noise
        self.theta_i = np.eye(3)*5.8e-4**2
        self.ohmega_i = np.eye(3)*5e-6**2 * gyro_bias
        self.accel_i = np.eye(3)*1e-4**2 * accel_bias
        self.g_i = np.eye(3)*1e-5**2*0 * gravity

        # Measurement noise
        m_noise = 8.5e-3 #/ 9.8255
        self.RR = np.eye(3)*m_noise

        # Error covariance
        self.P = scipy.linalg.block_diag(0*np.eye(3), 
                                         5e-6*np.eye(3)*gyro_bias, 
                                         3e-6*np.eye(3)*accel_bias, 
                                         1e-9*np.eye(3)*gravity, 
                                         1e-2*np.eye(3)*accel_offset
                                         )
        
    def state(self):
        return np.array(list(self.q) + list(self.wb) + list(self.a) + list(self.g) + list(self.a_offset))

    def Qi(self, dt):
        return scipy.linalg.block_diag(self.theta_i,#*dt**2, 
                                       self.ohmega_i,#*dt, 
                                       self.accel_i,#*dt
                                       self.g_i
                                       )

    def transition_matrix(self, w, dt):
        trans_matrix = np.eye(15)
        trans_matrix[:3, :3] = R.from_rotvec((w-self.wb)*dt).as_matrix().transpose()
        trans_matrix[:3, 3:6] = -np.eye(3)*dt
        trans_matrix[3:6, 3:6] = np.eye(3)
        #trans_matrix[6:9, 6:9] = np.eye(3)
        #print(np.diag(trans_matrix))
        
        return trans_matrix

    def noise_transition_matrix(self):
        # TODO: with bias
        Fi = np.zeros((15, 12))
        Fi[:12, :12] = np.eye(12)

        return Fi

    def measurement_matrix(self, dt):
        rotmat = R.from_quat(self.q).as_matrix()
        a_offset_rot = R.from_rotvec(self.a_offset).as_matrix()

        Hq = np.matmul(a_offset_rot.transpose(), Q.skew(np.matmul(rotmat.transpose(), self.g)))
        Ha_offset = Q.skew(np.matmul(a_offset_rot.transpose(), np.matmul(rotmat.transpose(), self.g)))
        Hqa = np.matmul(a_offset_rot.transpose(), Q.skew(np.matmul(rotmat.transpose(), self.a)))
        #Hq = R.from_rotvec(np.matmul(rotmat.transpose(), g)).as_matrix()

        H = np.zeros((3,15))
        H[:, :3] = Hq #+ Hqa
        #H[:, 3:6] = -Hq
        H[:, 6:9] = np.eye(3)
        H[:, 9:12] = rotmat.transpose()
        H[:, 12:15] = Ha_offset

        return H

    def predict(self, w, dt):
        # Propagate error state
        Fx = self.transition_matrix(w, dt)
        Fi = self.noise_transition_matrix()
        # self.dx = Fx*self.dx # Propagate mean (always zero)

        self.P = np.matmul(np.matmul(Fx, self.P), Fx.transpose()) + np.matmul(np.matmul(Fi, self.Qi(dt)), Fi.transpose())# Propagate covariance

        # Propagate nominal state
        rot = R.from_quat(self.q) * R.from_rotvec((w-self.wb)*dt)
        self.q = rot.as_quat()

    def update(self, a, dt):
        a_norm = np.linalg.norm(a)
        thresh = 10#0.1
        if a_norm - 9.8255 > thresh or a_norm - 9.8255 < -thresh:
            print("Outlier")
            return None
        a = a / 9.8255

        H = self.measurement_matrix(dt)

        num = np.matmul(self.P, H.transpose())
        den = np.matmul(np.matmul(H, self.P), H.transpose()) + self.RR
        den = np.linalg.inv(den)
        K = np.matmul(num, den)

        rot = R.from_rotvec(self.a_offset).inv()*R.from_quat(self.q).inv()
        residual = a - rot.apply(self.g) - self.a

        d_x = np.matmul(K, residual)
        d_theta, d_bias, d_accel, d_g, d_aoffset = d_x[:3], d_x[3:6], d_x[6:9], d_x[9:12], d_x[12:15]
        
        # This shuold be < 2
        #norm = np.linalg.norm(d_theta)

        rot = R.from_quat(self.q) * R.from_quat(list((d_theta)/2.0) + [1])
        self.q = rot.as_quat()
        self.wb += d_bias   
        self.a += d_accel
        self.g += d_g
        rot = R.from_rotvec(self.a_offset) * R.from_rotvec(d_aoffset)
        self.a_offset = rot.as_rotvec()

        self.P = self.P - np.matmul(np.matmul(K, H), self.P)
        
        # TODO: reset of d_theta (G*P*G^T)
        G = np.eye(15)
        #G[:3, :3] = R.from_rotvec(d_theta).inv().as_matrix() 
        G[:3, :3] = np.eye(3) -Q.skew(0.5 * d_theta)
        self.P = np.matmul(np.matmul(G, self.P), G.transpose())

        return residual

    def generateOrientationData(self, data, update=True):
        orientations = []
        covariances = []
        residuals = []
        update_on = []
        stop_update = False

        start = time.time()
        t_prev = 0
        for i, t in enumerate(data["time"]):
            dt = float(t)-float(t_prev)
            t_prev = float(t)

            q = self.q.copy()

            state = self.state()
            state[:4] = q
            orientations.append(state)
            covariances.append(self.P)

            w = np.array(list(map(float, data["gyro"][i])))
            self.predict(w, dt)
            updated = 0
            if update and not stop_update:
                a = np.array(data["accel"][i])
                residual = self.update(a, dt)
                if residual is None:
                    residual = (0, 0, 0)
                    #stop_update = True
                else:
                    updated = 1
                    residuals.append(np.abs(residual))
            update_on.append(updated)

        elapsed = time.time() - start
        avg_time = elapsed / len(data["time"])
        print("Avg time: {} ({} Hz)".format(avg_time, 1./avg_time))
            
        return np.array(orientations), np.array(residuals), np.array(covariances), update_on


class ESKF:
    def __init__(self, q_init=None, g=9.8255):
        
        # Nomilan state
        if q_init is not None:
            self.q = np.array(q_init)
        else:
            self.q = np.array([0., 0, 0, 1])

        self.q_init = self.q.copy()

        self.g = g
        self.Qi = np.eye(3)*5.8e-4**2
        self.RR = np.eye(3)*8.5e-2
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
        #self.P = np.matmul(np.matmul(Fx, self.P), Fx.transpose()) + np.matmul(np.matmul(Fi, self.Qi), Fi.transpose())# Propagate covariance
        self.P = Fx.dot(self.P).dot(Fx.transpose()) + Fi.dot(self.Qi).dot(Fi.transpose())

        # Propagate nominal state
        #self.q = Q.quat_mult(self.q, np.array(list(0.5*w*dt) + [1])) # might work better sometimes
        self.q = Q.quat_mult(self.q, Q.rotvec_to_quat(w*dt))

    def update(self, a, dt):
        a_norm = np.linalg.norm(a)
        thresh = 100 #.1
        if a_norm-self.g > thresh or a_norm-self.g < -thresh:
            print("Outlier")
            return None
        
        normalize = False
        if normalize:
            a = a / self.g
            g = np.array([0, 0, 1.0])
            RR = self.RR
        else:
            RR = self.RR * self.g**2
            g = np.array([0, 0, self.g])

        H = self.measurement_matrix(g)

        num = np.matmul(self.P, H.transpose())
        den = np.matmul(np.matmul(H, self.P), H.transpose()) + RR
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
        update_on = []
        stop_update = False

        start = time.time()
        t_prev = 0
        for i, t in enumerate(data["time"]):
            dt = float(t)-float(t_prev)
            t_prev = float(t)

            orientations.append(self.q.copy())
            covariances.append(self.P.copy())

            w = np.array(list(map(float, data["gyro"][i])))
            self.predict(w, dt)
            updated = 0
            if update and not stop_update:
                a = np.array(data["accel"][i])
                residual = self.update(a, dt)
                if residual is None:
                    residual = (0, 0, 0)
                    #stop_update = True
                else:
                    updated = 1
                    residuals.append(residual)
            update_on.append(updated)

        elapsed = time.time() - start
        avg_time = elapsed / len(data["time"])
        print("Avg time: {} ({} Hz)".format(avg_time, 1./avg_time))
            
        return np.array(orientations), np.array(residuals), np.array(covariances), update_on


if __name__ == "__main__":
    data = {"time": [0.1*i for i in range(100)],
            "gyro": [ [0.1, 0.1, 0.1] for i in range(100) ]}


    kf = ESKF()
    newData = kf.generateOrientationData(data, update=False)
    print(kf.state())

    plt.plot(data["time"], data["gyro"])
    plt.show()
