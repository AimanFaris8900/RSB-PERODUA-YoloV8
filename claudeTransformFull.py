
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

def final_T():
    T_pcd = np.load("estimated_transform.npy")

    P_pcd = matrix_to_tcp(T_pcd)

    print("ESTIMATE TRANSFORM TCP: ", P_pcd)

    he = np.load("hand_eye_result.npz")

    T_cam2flange = np.eye(4)
    T_cam2flange[:3, :3] = he['R']
    T_cam2flange[:3, 3]  = he['t'].flatten()

    T_cam2flange_inv = inv_T(T_cam2flange)

    T_final = T_cam2flange @ T_pcd @ T_cam2flange_inv


    t_dobot = tcp_to_matrix(
        x=-479.8356,
        y=-485.8820,
        z=80.2956,
        rx=-87.9513,
        ry=14.7616,
        rz=167.9480
    )

    T_tcp = T_final @ t_dobot

    P_tcp = matrix_to_tcp(T_tcp)

    print(f"T_PCD: \n{T_pcd}, \nHE (R): \n{T_cam2flange}, \nHE (T): \n{he['t']}")

    print(f"\n\nT_final: \n{T_final}")

    print(f"\n\nDobot TCP: \n{P_tcp}")

def get_crop_center_3d(depth_image_cropped, intrinsics_original, bbox_x1, bbox_y1):
    """
    Get 3D coordinates of the center of an already-cropped depth image.

    Args:
        depth_image_cropped:  cropped depth image (H x W numpy array)
        intrinsics_original:  dict with fx, fy, cx, cy from original full image
        bbox_x1, bbox_y1:     top-left corner of the crop in original image coords
                              (needed to correct the principal point offset)
    Returns:
        center_3d: (X, Y, Z) in camera frame
    """
    h, w = depth_image_cropped.shape

    # center pixel in cropped image
    u_crop = w // 2
    v_crop = h // 2

    # convert back to original image pixel coords for correct deprojection
    u_orig = u_crop + bbox_x1
    v_orig = v_crop + bbox_y1

    depth = depth_image_cropped[v_crop, u_crop]

    if depth == 0:
        print("Warning: center pixel depth is 0, using patch average")
        patch = depth_image_cropped[v_crop-5:v_crop+5, u_crop-5:u_crop+5]
        valid = patch[patch > 0]
        depth = valid.mean() if len(valid) > 0 else None

    if depth is None:
        print("No valid depth found!")
        return None

    fx, fy = intrinsics_original['fx'], intrinsics_original['fy']
    cx, cy = intrinsics_original['cx'], intrinsics_original['cy']

    X = (u_orig - cx) * depth / fx
    Y = (v_orig - cy) * depth / fy
    Z = depth

    print(f"Crop size:          {w} x {h}")
    print(f"Center pixel (crop):  ({u_crop}, {v_crop})")
    print(f"Center pixel (orig):  ({u_orig}, {v_orig})")
    print(f"Depth:              {depth:.4f}")
    print(f"3D (camera frame):  X={X:.4f}, Y={Y:.4f}, Z={Z:.4f}")

    return np.array([X, Y, Z])

def load_intrinsics(path):
    intrinsics = np.load(path)
    intrinsics_matrix = intrinsics["camera_matrix"]
    fx = intrinsics_matrix[0,0]
    fy = intrinsics_matrix[1,1]
    cx = intrinsics_matrix[0,2]
    cy = intrinsics_matrix[1,2]

    print(f"{fx}, {fy}, {cx}, {cy}")

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy
    }

def main():
    # Load your data
    T_icp = np.load("estimated_transform.npy")

    he = np.load("hand_eye_result.npz")
    
    T_cam2flange = np.eye(4)
    T_cam2flange[:3, :3] = he['R']
    T_cam2flange[:3, 3]  = he['t'].flatten()

    # Step 1 — camera origin in object frame
    T_cam_in_object = inv_T(T_icp)

    # Step 2 — flange/TCP in object frame
    T_tcp_in_object = T_cam_in_object @ T_cam2flange

    # Read out position and orientation
    tcp_pos = T_tcp_in_object[:3, 3]
    tcp_euler = Rotation.from_matrix(T_tcp_in_object[:3, :3]).as_euler('xyz', degrees=True)

    print("TCP position relative to object (m):", tcp_pos)
    print("TCP rotation relative to object (deg):", tcp_euler)

if __name__ == "__main__":
    # final_T()
    load_intrinsics("camera_intrinsics.npz")