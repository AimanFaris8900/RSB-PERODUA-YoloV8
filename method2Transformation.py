import numpy as np
from scipy.spatial.transform import Rotation

def inv_T(T):
    Ri = T[:3, :3].T
    out = np.eye(4)
    out[:3, :3] = Ri
    out[:3, 3] = -Ri @ T[:3, 3]
    return out

def tcp_to_matrix(x, y, z, rx, ry, rz):
    """
    Convert Dobot TCP pose to 4x4 homogeneous transform.
    x, y, z in mm (converted to meters internally)
    rx, ry, rz in degrees
    """
    # Rotation -- use whatever convention your Dobot uses
    def make_rot(angle_deg, axis):
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        if axis == 'x': return np.array([[1,0,0],[0,c,-s],[0,s,c]])
        if axis == 'y': return np.array([[c,0,s],[0,1,0],[-s,0,c]])
        if axis == 'z': return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    R = make_rot(rz,'z') @ make_rot(ry,'y') @ make_rot(rx,'x')   # ZYX extrinsic

    # Build 4x4
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = np.array([x, y, z]) / 1000.0  # mm -> meters

    return T

def matrix_to_tcp(T):
    """
    Convert 4x4 homogeneous matrix back to TCP pose.
    Returns translation in mm, rotation in degrees.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # Translation -- back to mm
    x, y, z = t * 1000.0

    # Rotation to Euler -- match your Dobot's convention (ZYX extrinsic)
    r = Rotation.from_matrix(R)
    rz, ry, rx = r.as_euler('zyx', degrees=True)  # ZYX extrinsic

    return x, y, z, rx, ry, rz

def origin_final_transformation(t_cam, t_final):

    T_Cam2Final = inv_T(t_cam) @ t_final

    return T_Cam2Final

def camera_origin(path):
    camera_coord = []
    pcd_center = np.load(path) * 1000   #convert from meter to mm

    for axis in pcd_center:
        camera_coord.append(0-axis)

    print(camera_coord)
    return camera_coord

def cam_offset():
    he = np.load("numpy_trans/hand_eye_result.npz")

    T_cam2flange = np.eye(4)
    T_cam2flange[:3, :3] = he['R']
    T_cam2flange[:3, 3]  = he['t'].flatten()

    T_cam2flange_inv = inv_T(T_cam2flange)

    return T_cam2flange

def main():
    # get camera coordinate relative to PCD center
    camera_coord = camera_origin("numpy_trans/pcd_center.npy")

    t_cam = tcp_to_matrix(
        x=camera_coord[0],
        y=camera_coord[1],
        z=camera_coord[2],
        rx=0,
        ry=0,
        rz=0
    )
    print(t_cam)

    T_pcd = np.load("numpy_trans/estimated_transform.npy")

    # new camera coordinate after PCD tranform in virtual space
    t_final = T_pcd @ t_cam

    P_final = matrix_to_tcp(t_final)

    print("Final P: \n")
    for tcp in P_final:
        print(f"\n{tcp}")

    # get transformatiom from camera coord to final camera coord virtual
    T_Cam2Final = origin_final_transformation(t_cam, t_final)

    t_dobot = tcp_to_matrix(
        x=198.033127,
        y=-352.80838,
        z=210.772324,
        rx=-98.440735,
        ry=15.355934,
        rz=158.735733
    )

    # get depth camera offset to flange
    T_dobot_cam = cam_offset() @ t_dobot

    # apply transformation to TCP
    T_dobot = T_Cam2Final @ t_dobot

    print("DOBOT COORDINATE: \n",)
    P_dobot = matrix_to_tcp(T_dobot)
    for dobot in P_dobot:
        print(f"\n{dobot}")

    

if __name__ == "__main__":
    main()