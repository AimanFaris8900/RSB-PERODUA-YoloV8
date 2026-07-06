"""
Euler Convention Tester
-----------------------
Tries all 6 possible axis orders (XYZ, XZY, YXZ, YZX, ZXY, ZYX)
for both extrinsic and intrinsic conventions, runs hand-eye calibration
for each, and reports the consistency spread (lower = better convention match).
 
Run this on your current dataset to find the correct Euler convention
your Dobot uses, rather than guessing.
"""
 
import cv2
import numpy as np
import pandas as pd
import itertools
 
# ---- paste your actual config here ----
CAMERA_MATRIX = np.array([
    [1.39273660e+03, 0.00000000e+00, 7.94918797e+02],
    [0.00000000e+00, 1.39263253e+03, 5.96387540e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
])
DIST_COEFFS = np.array([1.86553625e-01, -9.04931834e-01, -1.37049273e-03,
                         -7.48187569e-04,  1.29838443e+00])
 
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_LENGTH = 0.030
MARKER_LENGTH = 0.022
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
BOARD = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT
)
DETECTOR = cv2.aruco.CharucoDetector(BOARD)
 
CSV_PATH = "poses.csv"  # your CSV file
 
 
def make_rot(angle_deg, axis):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 'x':
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    elif axis == 'y':
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    elif axis == 'z':
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])
 
 
def euler_to_rotmat(rx, ry, rz, order):
    """
    Build rotation matrix for given axis order string e.g. 'xyz', 'zyx'.
    Extrinsic (fixed-axis) application: applied right-to-left.
    Intrinsic (body-axis) application: same matrix result but axes named differently.
    We test both by also trying the reversed order string.
    """
    axes = list(order)
    angles = {'x': rx, 'y': ry, 'z': rz}
    R = np.eye(3)
    for ax in reversed(axes):  # extrinsic: apply rightmost first
        R = make_rot(angles[ax], ax) @ R
    return R
 
 
def detect_charuco_pose(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    charuco_corners, charuco_ids, _, _ = DETECTOR.detectBoard(img)
    if charuco_ids is None or len(charuco_ids) < 6:
        return None
    obj_points, img_points = BOARD.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or len(obj_points) < 6:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, CAMERA_MATRIX, DIST_COEFFS)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)
 
 
def consistency_spread(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list):
    """Run TSAI and return mean std deviation across xyz of consistency check."""
    R_b2g, t_b2g = [], []
    for R, t in zip(R_g2b_list, t_g2b_list):
        R_inv = R.T
        t_inv = -R_inv @ t
        R_b2g.append(R_inv)
        t_b2g.append(t_inv)
 
    try:
        R_c2b, t_c2b = cv2.calibrateHandEye(
            R_b2g, t_b2g, R_t2c_list, t_t2c_list,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
    except cv2.error:
        return float('inf'), None, None
 
    positions = []
    for i in range(len(R_g2b_list)):
        t_target_base = R_c2b @ t_t2c_list[i] + t_c2b.ravel()
        positions.append(t_target_base)
 
    positions = np.array(positions)
    spread = np.std(positions, axis=0)
    mean_spread_mm = np.mean(spread) * 1000
    return mean_spread_mm, R_c2b, t_c2b
 
 
def main():
    # Load CSV
    df = pd.read_csv(CSV_PATH, header=None)
    df.columns = ['path', 'x', 'y', 'z', 'rx', 'ry', 'rz']
 
    # Detect board poses
    R_t2c_list, t_t2c_list = [], []
    valid_rows = []
    for _, row in df.iterrows():
        result = detect_charuco_pose(row['path'])
        if result is None:
            print(f"  [!] Skip {row['path']}")
            continue
        R_t2c_list.append(result[0])
        t_t2c_list.append(result[1])
        valid_rows.append(row)
 
    print(f"{len(valid_rows)} valid samples\n")
 
    # Try all 6 axis orders
    all_orders = [''.join(p) for p in itertools.permutations('xyz')]
    results = []
 
    for order in all_orders:
        R_g2b_list, t_g2b_list = [], []
        for row in valid_rows:
            R = euler_to_rotmat(row['rx'], row['ry'], row['rz'], order)
            t = np.array([row['x'], row['y'], row['z']]) / 1000.0
            R_g2b_list.append(R)
            t_g2b_list.append(t)
 
        spread, R_best, t_best = consistency_spread(
            R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list
        )
        results.append((spread, order, R_best, t_best))
        print(f"Order {order.upper()} (extrinsic): mean spread = {spread:.1f} mm")
 
    # Sort by spread (lower is better)
    results.sort(key=lambda x: x[0])
    best_spread, best_order, best_R, best_t = results[0]
 
    print(f"\n=== BEST CONVENTION: {best_order.upper()} (extrinsic) ===")
    print(f"Mean consistency spread: {best_spread:.1f} mm")
    print("R (base->cam):\n", best_R)
    print("t (base->cam, meters):\n", best_t.ravel())
 
    if best_R is not None:
        np.savez("hand_eye_result.npz", R=best_R, t=best_t, convention=best_order)
        print(f"\nSaved hand_eye_result.npz with convention={best_order.upper()}")
 
 
if __name__ == "__main__":
    main()