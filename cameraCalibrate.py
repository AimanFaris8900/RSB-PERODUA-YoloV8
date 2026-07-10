import cv2
import csv
import numpy as np
import os
import glob
import pandas as pd
from scipy.spatial.transform import Rotation

# Defining the dimensions of checkerboard
CHECKERBOARD = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

CAMERA_MATRIX = np.array([
    [1.39273660e+03, 0.00000000e+00, 7.94918797e+02],
    [0.00000000e+00, 1.39263253e+03, 5.96387540e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
])
DIST_COEFFS = np.array([1.86553625e-01, -9.04931834e-01, -1.37049273e-03,
                         -7.48187569e-04, 1.29838443e+00])
 
# ----------------------------------------------------------------------------
# ChArUco board definition -- must match generate_charuco_board.py
# ----------------------------------------------------------------------------
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_LENGTH = 0.0275   # meters -- confirm with calipers after printing
MARKER_LENGTH = 0.02
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
BOARD = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)
DETECTOR = cv2.aruco.CharucoDetector(BOARD)
 
# ----------------------------------------------------------------------------
# MANUALLY ENTERED ROBOT TCP POSES
# Format: (image_filename, x, y, z, rx, ry, rz)
#   x,y,z in mm (will be converted to meters below)
#   rx,ry,rz in degrees -- CONFIRM your Dobot's Euler convention in the manual
#   (commonly intrinsic ZYX or extrinsic XYZ -- this matters!)
# ----------------------------------------------------------------------------
POSES = [
    # ("imgs/pose1.png", 250.0, -100.0, 300.0, 178.0, 2.0, 90.0),
    # ("imgs/pose2.png", 220.0, -50.0,  280.0, 160.0, 10.0, 75.0),
    # ... add all your captured poses here
]


def calibrate_camera():
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    
    # Extracting path of individual image stored in a given directory
    images = glob.glob('./imgs/*.png')
    for fname in images:
        print("IMAGE NAME: ", fname)
        img = cv2.imread(fname)
        img = sharpen_image(img)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display
        them on the images of checker board
        """
        
        if ret == True:
            print("DETECT CHECKERBOARD")
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        

        cv2.imshow('img',img)
        cv2.waitKey(0)
        print("NEXT IMAGE")
    
    cv2.destroyAllWindows()
    
    h,w = img.shape[:2]
    
    """
    Performing camera calibration by
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"Total reprojection error: {mean_error / len(objpoints)}")
    
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)

    np.savez("camera_intrinsics.npz", camera_matrix=mtx, dist_coeffs=dist)

def capture_image():
    cap = cv2.VideoCapture(1)

    num = 1

    while cap.isOpened():
        success, img = cap.read()

        k = cv2.waitKey(5)

        if k == 27:
            break
        elif k == ord('S'):
            cv2.imread(f"imgs/img_{str(num)}.jpg", img)
            num += 1

        cv2.imshow("img",img)

    cap.release()
    cv2.destroyAllWindows()

def sharpen_image(image):

    # Define a 3x3 sharpening kernel
    kernel = np.array([[0, -1, 0], 
                    [-1, 5, -1], 
                    [0, -1, 0]])

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Blend the original and blurred image to sharpen
    # Formula: sharpened = original * alpha + blurred * beta + gamma
    sharpened_usm = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    

    # #Display results
    # cv2.imshow("Original", blurred)
    # cv2.imshow("Sharpened", sharpened_usm)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return sharpened_usm

def hand_eye_calibration():
    pass
 
def euler_to_rotmat(rx, ry, rz, degrees=True):
    """
    Convert Euler angles to rotation matrix.
    ADJUST THIS to match your Dobot's actual convention (check manual).
    This default assumes extrinsic XYZ (R = Rz @ Ry @ Rx applied in that order).
    """
    if degrees:
        rx, ry, rz = np.radians([rx, ry, rz])
 
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
 
    return Rz @ Ry @ Rx
 
def detect_charuco_pose(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [!] Could not read {image_path}")
        return None
 
    charuco_corners, charuco_ids, marker_corners, marker_ids = DETECTOR.detectBoard(img)
    if charuco_ids is None or len(charuco_ids) < 6:
        print(f"  [!] Insufficient ChArUco detection in {image_path}")
        return None
 
    obj_points, img_points = BOARD.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or len(obj_points) < 6:
        return None
 
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, CAMERA_MATRIX, DIST_COEFFS)
    if not ok:
        return None
 
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)
 
def hand_eye_main():
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    poses = read_csv()
 
    for image_path, x, y, z, rx, ry, rz in poses:
        result = detect_charuco_pose(image_path)
        if result is None:
            print(f"  Skipping {image_path} -- detection failed")
            continue
 
        R_t2c, t_t2c = result
        R_g2b = euler_to_rotmat(rx, ry, rz)
        t_g2b = np.array([x, y, z]) / 1000.0  # mm -> m
 
        R_gripper2base_list.append(R_g2b)
        t_gripper2base_list.append(t_g2b)
        R_target2cam_list.append(R_t2c)
        t_target2cam_list.append(t_t2c)
        print(f"  [+] {image_path} -- OK")
 
    n = len(R_gripper2base_list)
    print(f"\n{n} valid samples collected")
    if n < 8:
        print("WARNING: fewer than 8 samples -- result may be unstable. "
              "Aim for 12-15+ with varied orientations.")
 
    # Eye-in-hand (CORRECT for your setup) -- pass directly:
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list,  # pass as-is, no inversion
        R_target2cam_list, t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    print("\n--- Consistency check (board position in base frame) ---")
    for i in range(len(R_gripper2base_list)):
        R_board_base = R_gripper2base_list[i] @ R_cam2gripper @ R_target2cam_list[i]
        t_board_base = (R_gripper2base_list[i] @ R_cam2gripper @ t_target2cam_list[i]
                        + R_gripper2base_list[i] @ t_cam2gripper.ravel()
                        + t_gripper2base_list[i])
        print(f"Sample {i}: {t_board_base}")
    print("These should all be nearly identical.")
 
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
 
    results = {}
    for name, flag in methods.items():
        try:
            R_cam2base, t_cam2base = cv2.calibrateHandEye(
                R_gripper2base_list, t_gripper2base_list,
                R_target2cam_list, t_target2cam_list,
                method=flag,
            )
            results[name] = (R_cam2base, t_cam2base)
            print(f"\n--- {name} ---")
            print("R:\n", R_cam2base)
            print("t:\n", t_cam2base.ravel())
        except cv2.error as e:
            print(f"\n--- {name} FAILED, skipping ---")
 
    # Consistency check: target position in base frame should be near-identical
    # across all samples since the board is rigidly fixed to the gripper
    R_best, t_best = results["PARK"]
    print("\n--- Consistency check (target position in base frame, meters) ---")
    for i in range(n):
        t_target_base = R_best @ t_target2cam_list[i] + t_best.ravel()
        print(f"Sample {i}: {t_target_base}")
    print("Values should be close together. Large spread = bad sample(s) or "
          "wrong Euler convention in euler_to_rotmat().")
 
    np.savez("hand_eye_result.npz", R=R_best, t=t_best)
    print("\nSaved hand_eye_result.npz")


def read_csv():
    df = pd.read_csv("poses.csv", header=None)
    print(df)
    poses = []

    for index, rows in df.iterrows():
        rows = rows.to_list()
        print(f"INDEX: {index}, ROWS: {rows}")
        poses.append(rows)

    return poses

def read_npz():
    with np.load('hand_eye_result.npz') as data:
        # See the names of the arrays inside
        print(data.files)
        
        # Access a specific array by its name
        my_array = data['t']
        print(my_array)

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

    R = make_rot(rz,'z') @ make_rot(ry,'y') @ make_rot(rx,'x')  # ZYX extrinsic

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

def transform_multiply():
    dobot = tcp_to_matrix(
        x=-479.8356,
        y=-485.8820,
        z=80.2956,
        rx=-87.9513,
        ry=14.7616,
        rz=167.9480
    )

    print("DOBOT: ", dobot)
    
    estimate = np.load("estimated_transform.npy")

    print("ESTIMATED TRANSFORM TCP: ", matrix_to_tcp(estimate))
    end_transform = dobot @ estimate

    print(end_transform)

    return end_transform

if __name__ == "__main__":
    # img = cv2.imread("imgs/color_1280x720_1_1782793011.0.png")
    # sharpen_image(img)
    # calibrate_camera()
    # capture_image()
    # read_csv()
    # hand_eye_main()
    # read_npz()
    # Example -- paste your current TCP reading here:
    end_transform = transform_multiply()
    x, y, z, rx, ry, rz = matrix_to_tcp(end_transform)

    print(f"x  = {x:.4f} mm")
    print(f"y  = {y:.4f} mm")
    print(f"z  = {z:.4f} mm")
    print(f"rx = {rx:.4f} deg")
    print(f"ry = {ry:.4f} deg")
    print(f"rz = {rz:.4f} deg")
