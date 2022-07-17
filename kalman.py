from scipy.spatial.transform import Rotation as R
import numpy as np

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


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

        self.P = np.eye(3)*0.0005 # error covariance

        self.Qi = np.eye(3)*1.8e-5**2 # measurement/process covariance TODO: tune

        self.is_initialized = False

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
    
        qx, qy, qz, qw = self.q
        weird_skew = .5 * np.array([[qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qw]])

        H = skew(np.matmul(rotmat.transpose(), g))
        #H = np.matmul(H, JRight)
        #H = np.matmul(H, weird_skew)

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
        
        #RR = np.eye(3)*0.00836**2 # TODO: tune
        RR = np.eye(3)*.018 # TODO: tune

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
        return residual

    def generateOrientationData(self, data, predict=True, update=True, q_offset=None):
        orientations = []
        covariances = []
        residuals = []
        dt = float(data["time"][1]) - float(data["time"][0])

        for i, t in enumerate(data["time"]):

            q = self.q
            if q_offset is not None:
                rot = R.from_quat(list(-q_offset[:3]) + [q_offset[3]]) * R.from_quat(self.q)
                q = rot.as_quat()
                #self.q = q.copy()

            orientations.append(q.copy())
            covariances.append(self.P)

            w = np.array(list(map(float, data["gyro"][i])))
            self.predict(w, dt)
            if update:
                a = np.array(data["accel"][i])
                residual = self.update(a, dt)
                residuals.append(np.abs(residual))

            

            
            
        return np.array(orientations), np.array(residuals), np.array(covariances)

if __name__ == "__main__":
    data = {"time": [0.1*i for i in range(100)],
            "gyro": [ [0.1, 0.1, 0.1] for i in range(100) ]}

    kf = Kalman()
    newData = kf.generateOrientationData(data, update=False)
    print(kf.state)
