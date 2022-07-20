from scipy.spatial.transform import Rotation as R
import scipy
import numpy as np
import time

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def qL(q):
    qx, qy, qz, qw = q
    return np.array([[qw, -qz, qy, qx],
                     [qz, qw, -qx, qy],
                     [-qy, qx, qw, qz],
                     [-qx, -qy, -qz, qw]
                    ])

class ESKF:
    def __init__(self, init_state=None):
        
        # Nomilan state
        if init_state is not None:
            self.q = np.array(init_state[:4])
            self.wb = np.array(init_state[4:])
        else:
            self.q = np.array([0., 0, 0, 1])    # orientation
            self.wb = np.array([0.0035, -0.0026, 0])      # gyro bias

        # Error state
        #self.d_theta = np.array([0., 0, 0])

        theta_i = np.eye(3)*4.0e-6**2
        ohmega_i = np.eye(3)*1.8e-6**2
        self.Qi =  scipy.linalg.block_diag(theta_i, ohmega_i)# measurement/process covariance TODO: tune

        #self.P = np.eye(3)*0.0005 # error covariance
        self.P = scipy.linalg.block_diag(theta_i*1.8e3, ohmega_i*1e3) # start with initial bias value uncertain
        
        self.is_initialized = False

    def state(self):
        return np.array(list(self.q) + list(self.wb))

    def transition_matrix(self, w, dt):
        rotmat = R.from_rotvec((w-self.wb)*dt).as_matrix().transpose()

        # TODO: with bias
        trans_matrix = np.zeros((6,6))
        trans_matrix[:3, :3] = rotmat
        trans_matrix[:3, 3:7] = -np.eye(3)*dt
        trans_matrix[3:7, 3:7] = np.eye(3)

        return trans_matrix

    def noise_transition_matrix(self):
        # TODO: with bias
        # return np.eye(6)
        return np.eye(6)

    def measurement_matrix(self, g, dt):
        rvec = R.from_quat(self.q).as_rotvec()
        theta = np.linalg.norm(rvec)
        # Eq (143) in micro theory
        thetaSkew = skew(rvec)

        rotmat = R.from_quat(self.q).as_matrix()
    
        #qx, qy, qz, qw = self.q
        #weird_skew = .5 * np.array([[qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qw]])

        Hq = skew(np.matmul(rotmat.transpose(), g))

        H = np.zeros((3,6))
        H[:, :3] = Hq
        H[:, 3:] = -Hq

        return H

    def predict(self, w, dt):
        # Propagate error state
        Fx = self.transition_matrix(w, dt)
        Fi = self.noise_transition_matrix()
        # self.dx = Fx*self.dx # Propagate mean (always zero)

        self.P = np.matmul(np.matmul(Fx, self.P), Fx.transpose()) + np.matmul(np.matmul(Fi, self.Qi), Fi.transpose())# Propagate covariance

        # Propagate nominal state
        rot = R.from_quat(self.q) * R.from_rotvec((w-self.wb)*dt)
        self.q = rot.as_quat()

    def update(self, a, dt):
        a_norm = np.linalg.norm(a)
        thresh = 1000 #.1
        if a_norm-9.82 > thresh or a_norm-9.82 < -thresh:
            print("Ignoring uncertain accel measurement")
            return (0., 0, 0)
        a = a / a_norm
        g = np.array([0, 0, 1.])
    
        a_thresh = 10
        #if np.linalg.norm(a) > 9.82+a_thresh or np.linalg.norm(a) < 9.82-a_thresh:
        #    print("Ignoring outlier")
        #    return (0., 0, 0)

        H = self.measurement_matrix(g, dt)
        
        #RR = np.eye(3)*0.0018 # accel
        RR = np.eye(3)*0.00004 # accel

        num = np.matmul(self.P, H.transpose())
        den = np.matmul(np.matmul(H, self.P), H.transpose()) + RR
        den = np.linalg.inv(den)
        K = np.matmul(num, den)

        # TODO: reset of d_theta (G*P*G^T)
        self.P = self.P - np.matmul(np.matmul(K, H), self.P)

        residual = a - R.from_quat(self.q).inv().apply(g)

        d_x = np.matmul(K, residual)
        d_theta, d_bias = d_x[:3], d_x[3:]
        self.wb += d_bias

        rot = R.from_quat(self.q) * R.from_quat(list((d_theta)/2.0) + [1])
        self.q = rot.as_quat()
        
        #print(self.wb)
        return residual

    def generateOrientationData(self, data, predict=True, update=True, q_offset=None):
        orientations = []
        covariances = []
        residuals = []
        dt = float(data["time"][1]) - float(data["time"][0])

        start = time.time()
        for i, t in enumerate(data["time"]):

            q = self.q.copy()
            if q_offset is not None:
                rot = R.from_quat(list(-q_offset[:3]) + [q_offset[3]]) * R.from_quat(q)
                q = rot.as_quat()
                #self.q = q.copy()

            state = self.state()
            state[:4] = q
            orientations.append(state)
            covariances.append(self.P)

            w = np.array(list(map(float, data["gyro"][i])))
            self.predict(w, dt)
            if update:
                a = np.array(data["accel"][i])
                residual = self.update(a, dt)
                residuals.append(np.abs(residual))
        elapsed = time.time() - start
        avg_time = elapsed / len(data["time"])
        print("Avg time: {} ({} Hz)".format(avg_time, 1./avg_time))
            

            
            
        return np.array(orientations), np.array(residuals), np.array(covariances)


class Kalman:
    def __init__(self, q_init=None):
        
        # Nomilan state
        if q_init is not None:
            self.q = np.array(q_init)
        else:
            self.q = np.array([0., 0, 0, 1])    # orientation
        self.wb = np.array([0., 0, 0])      # gyro bias

        # Error state
        #self.d_theta = np.array([0., 0, 0])


        self.Qi = np.eye(3)*1.8e-5**2 # measurement/process covariance TODO: tune


        #self.P = np.eye(3)*0.0005 # error covariance
        self.P = self.Qi.copy()

        self.is_initialized = False

    def state(self):
        return np.array(list(self.q) + list(self.wb))

    def transition_matrix(self, w, dt):
        rotmat = R.from_rotvec((w-self.wb)*dt).as_matrix().transpose()

        trans_matrix = rotmat

        # TODO: with bias
        #trans_matrix = np.zeros((6,6))
        #trans_matrix[:3, :3] = rotmat
        #trans_matrix[:3, 3:7] = -np.eye(3)*dt
        #trans_matrix[3:7, 3:7] = np.eye(3)

        return trans_matrix

    def noise_transition_matrix(self):
        # TODO: with bias
        # return np.eye(6)
        return np.eye(3)

    def measurement_matrix(self, g):
        rvec = R.from_quat(self.q).as_rotvec()
        theta = np.linalg.norm(rvec)
        # Eq (143) in micro theory
        thetaSkew = skew(rvec)

        if theta < 1e-6:
            JRight = np.eye(3)
        else:
            JRight = np.eye(3) - (1-np.cos(theta))/theta**2*thetaSkew + (theta-np.sin(theta))/theta**3*np.matmul(thetaSkew, thetaSkew)

        rotmat = R.from_quat(self.q).as_matrix()
    
        #q_skew = 0.5 * np.matmul(qL(self.q), np.array([[1., 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]))

        # old 
        H = skew(np.matmul(rotmat.transpose(), g))
        
        # new
        #H = np.matmul(H, rotmat)
        #H = np.matmul(H, skew(rvec))
        
        return H

    def predict(self, w, dt):
        # Propagate error state
        Fx = self.transition_matrix(w, dt)
        Fi = self.noise_transition_matrix()
        # self.dx = Fx*self.dx # Propagate mean (always zero)

        self.P = np.matmul(np.matmul(Fx, self.P), Fx.transpose()) + np.matmul(np.matmul(Fi, self.Qi), Fi.transpose())# Propagate covariance

        # Propagate nominal state
        rot = R.from_quat(self.q) * R.from_rotvec(w*dt)
        self.q = rot.as_quat()

    def update(self, a, dt):
        a_norm = np.linalg.norm(a)
        thresh = 1000 #.1
        if a_norm-9.82 > thresh or a_norm-9.82 < -thresh:
            print("Ignoring uncertain accel measurement")
            return (0., 0, 0)
        a = a / a_norm
        g = np.array([0, 0, 1.])
    
        a_thresh = 10
        #if np.linalg.norm(a) > 9.82+a_thresh or np.linalg.norm(a) < 9.82-a_thresh:
        #    print("Ignoring outlier")
        #    return (0., 0, 0)

        H = self.measurement_matrix(g)
        
        RR = np.eye(3)*0.00018#0.018 # TODO: tune

        num = np.matmul(self.P, H.transpose())
        den = np.matmul(np.matmul(H, self.P), H.transpose()) + RR
        den = np.linalg.inv(den)
        K = np.matmul(num, den)

        # TODO: reset of d_theta (G*P*G^T)
        self.P = self.P - np.matmul(np.matmul(K, H), self.P)

        residual = a - R.from_quat(self.q).inv().apply(g)
        d_theta = np.matmul(K, residual)
        rot = R.from_quat(self.q) * R.from_quat(list(d_theta/2.0) + [1])
        self.q = rot.as_quat()

        G = R.from_rotvec(-d_theta).as_matrix()
        #self.P = np.matmul(np.matmul(G, self.P), G.transpose())

        return residual

    def generateOrientationData(self, data, predict=True, update=True, q_offset=None):
        orientations = []
        covariances = []
        residuals = []
        dt = float(data["time"][1]) - float(data["time"][0])

        start = time.time()
        for i, t in enumerate(data["time"]):

            q = self.q
            if q_offset is not None:
                rot = R.from_quat(list(-q_offset[:3]) + [q_offset[3]]) * R.from_quat(self.q)
                q = rot.as_quat()
                #self.q = q.copy()

            orientations.append(self.state())
            covariances.append(self.P)

            w = np.array(list(map(float, data["gyro"][i])))
            self.predict(w, dt)
            if update:
                a = np.array(data["accel"][i])
                residual = self.update(a, dt)
                residuals.append(np.abs(residual))
        elapsed = time.time() - start
        avg_time = elapsed / len(data["time"])
        print("Avg time: {} ({} Hz)".format(avg_time, 1./avg_time))
            

            
            
        return np.array(orientations), np.array(residuals), np.array(covariances)

if __name__ == "__main__":
    data = {"time": [0.1*i for i in range(100)],
            "gyro": [ [0.1, 0.1, 0.1] for i in range(100) ]}

    kf = Kalman()
    newData = kf.generateOrientationData(data, update=False)
    print(kf.state)
