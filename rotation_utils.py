import numpy as np


def quat_norm(q):
    return q / np.linalg.norm(q)

def qL(q):
    qx, qy, qz, qw = q
    return np.array([[qw, -qz, qy, qx],
                     [qz, qw, -qx, qy],
                     [-qy, qx, qw, qz],
                     [-qx, -qy, -qz, qw]
                    ])

def quat_mult(q1, q2):
    return np.matmul(qL(q1), q2)

def exp_quat(w):
    # exponential map to S3
    theta = np.linalg.norm(w)
    if theta == 0:
        u = np.array([0., 0., 0.])
    else:
        u = w / theta
    return np.array(list(np.sin(theta) * u) + [np.cos(theta)])

def rotvec_to_quat(w):
    # Capitalized exponential map to S3
    return exp_quat(0.5 * w)

def quat_to_matrix(q):
    qw = q[3]
    qv = q[:3]
    qv_dot = qv.dot(qv)
    qv_dot_trans = qv.reshape((3,1)).dot(qv.reshape((1,3)))

    return (qw**2 - qv_dot)*np.eye(3) + 2*qv_dot_trans + 2*qw*skew(qv)

def rodrigues(w):
    # exponential map to SO(3)
    theta = np.linalg.norm(w)
    if theta == 0:
        u = np.array([0., 0., 0.])
    else:
        u = w / theta
    u_skew = skew(u)
    return np.eye(3) + np.sin(theta)*u_skew + (1.0-np.cos(theta))*np.matmul(u_skew, u_skew)

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

if __name__ == "__main__":
    import sys
    from scipy.spatial.transform import Rotation as R
    epsilon = sys.float_info.epsilon

    r1 = R.from_rotvec([0.1, 0.4, -0.654])
    w = np.array([-0.3, 0.2, -0.145])
    r2 = R.from_rotvec(w)
    
    # Test qL

    # Test quat_mul
    q_test = quat_mult(r1.as_quat(), r2.as_quat())
    r_true = r1*r2
    q_true = r_true.as_quat()
    assert np.allclose(q_test, q_true, atol=epsilon), "qL failed"
    
    # Test exp_quat
    q_test = exp_quat(0.5 * w)
    q_true = r2.as_quat()
    assert np.allclose(q_test, q_true, atol=epsilon), "exp_quat failed"
    
    # Test quat_to_matrix
    rotmat_test = quat_to_matrix(r1.as_quat())
    rotmat_true = r1.as_matrix()
    assert np.allclose(rotmat_test, rotmat_true, atol=epsilon), "quat_to_matrix failed"

    # Test rodrigues
    rotmat_test = rodrigues(w)
    rotmat_true = r2.as_matrix()
    assert np.allclose(rotmat_test, rotmat_true, atol=epsilon), "rodrigues failed"

    # Test skew