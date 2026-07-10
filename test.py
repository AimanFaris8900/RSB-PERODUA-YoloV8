from ultralytics import YOLO
import cv2
import numpy as np
import torch
import mediapipe as mp
import math

BOX_SCALE = 5

def yolo_model(frames):
    model = YOLO("weights/best.pt")

    # result = model.predict(source="test_images/test_6.jpeg")
    result = model.predict(source=frames)

    if result[0].masks:
        print(result[0].boxes.xyxy)
        data_boxes = [result[0].boxes.xyxy[0].tolist(), 0]
        return data_boxes
    
    return None

def crop_detection(img_path, depth_path):
    results = yolo_model(img_path)
    frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    print("depth shape:", depth.shape)
    print("depth dtype:", depth.dtype)
    print("depth max:", depth.max())

    if results:
        bounding_box = results[0]

        h_img, w_img = frame.shape[:2]

        top_left  = [
            max(0, int(bounding_box[0]) - BOX_SCALE),
            max(0, int(bounding_box[1]) - BOX_SCALE)
        ]
        btm_right = [
            min(w_img, int(bounding_box[2]) + BOX_SCALE),
            min(h_img, int(bounding_box[3]) + BOX_SCALE)
        ]

        print("TOP LEFT: ", top_left)
        print("BTM RIGHT: ", btm_right)

        cropped_img = frame[top_left[1]:btm_right[1], top_left[0]:btm_right[0]]       #startY:endY, startX:endX
        cropped_depth = depth[top_left[1]:btm_right[1], top_left[0]:btm_right[0]] 

        intrinsics = load_intrinsics("numpy_trans/camera_intrinsics.npz")
        center_3d = get_center_3d(cropped_depth, top_left, intrinsics)

    np.save("cropped_img/pcd_center.npy", center_3d)
    cv2.imwrite("cropped_img/img.jpg", cropped_img)
    cv2.imwrite("cropped_img/depth.png", cropped_depth)
    cv2.imshow('Camera', cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Get depth camera PCD coordinates
def get_center_3d(cropped_depth, top_left, intrinsics):
    """
    Get 3D coordinates of the center of a cropped depth image.

    Args:
        cropped_depth:  cropped depth image (H x W numpy array, uint16 mm)
        top_left:       [x, y] top-left corner of crop in original image coords
        intrinsics:     dict with fx, fy, cx, cy

    Returns:
        center_3d: numpy array [X, Y, Z] in meters (camera frame), or None
    """
    # force single channel
    if cropped_depth.ndim == 3:
        cropped_depth = cropped_depth[:, :, 0]

    h, w = cropped_depth.shape
    u_crop, v_crop = w // 2, h // 2

    # offset back to original image coords
    u_orig = u_crop + top_left[0]
    v_orig = v_crop + top_left[1]

    depth_val = cropped_depth[v_crop, u_crop]

    if depth_val == 0:
        print("Warning: center depth is 0, using patch average")
        patch = cropped_depth[v_crop-5:v_crop+5, u_crop-5:u_crop+5]
        valid = patch[patch > 0]
        depth_val = valid.mean() if len(valid) > 0 else None

    if depth_val is None:
        print("Could not compute 3D center — no valid depth")
        return None

    depth_m = depth_val / 1000.0  # mm → meters
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    X = (u_orig - cx) * depth_m / fx
    Y = (v_orig - cy) * depth_m / fy
    Z = depth_m

    center_3d = np.array([X, Y, Z])
    print(f"Crop center pixel (orig): ({u_orig}, {v_orig})")
    print(f"Depth: {depth_m:.4f} m")
    print(f"3D center (camera frame): X={X:.4f}, Y={Y:.4f}, Z={Z:.4f} m")

    return center_3d

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

def visualize_result(img_path):
    results = yolo_model(img_path)
    frame = cv2.imread(img_path)

    if results:
        bounding_box = results[0]
        top_left = [int(bounding_box[0])-BOX_SCALE,int(bounding_box[1])-BOX_SCALE]
        btm_right = [int(bounding_box[2])+BOX_SCALE, int(bounding_box[3])+BOX_SCALE]

        print("BOUNDING BOX: ", bounding_box)
        frame = cv2.rectangle(frame, (int(bounding_box[0])-BOX_SCALE,int(bounding_box[1])-BOX_SCALE), (int(bounding_box[2])+BOX_SCALE, int(bounding_box[3])+BOX_SCALE), color=(0,0,255))
        center = calculate_center(bounding_box)
        print("CENTER VAR: ", center)
        # CENTER
        frame = cv2.circle(frame, (center[0], center[1]), 5, (0,0,255), -1)

        # BOX POINTS
        frame = cv2.circle(frame, (int(bounding_box[0]), int(bounding_box[1])), 5, (255,0,0), -1)
        frame = cv2.circle(frame, (int(bounding_box[2]), int(bounding_box[3])), 5, (0,255,0), -1)

    # Display the captured frame
    
    cv2.imshow('Camera', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def camera_feed():
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FRAME HEIGHT: ", frame_height)
    print("FRAME WIDTH: ", frame_width)

    while True:
        ret, frame = cam.read()

        results = yolo_model(frame)

        if results:
            bounding_box = results[0]
            print("BOUNDING BOX: ", bounding_box)
            frame = cv2.rectangle(frame, (int(bounding_box[0]),int(bounding_box[1])), (int(bounding_box[2]), int(bounding_box[3])), color=(0,0,255))
            center = calculate_center(bounding_box)
            print("CENTER VAR: ", center)
            # CENTER
            frame = cv2.circle(frame, (center[0], center[1]), 5, (0,0,255), -1)

            # BOX POINTS
            frame = cv2.circle(frame, (int(bounding_box[0]), int(bounding_box[1])), 5, (255,0,0), -1)
            frame = cv2.circle(frame, (int(bounding_box[2]), int(bounding_box[3])), 5, (0,255,0), -1)

        # Display the captured frame
        
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def calculate_center(bounding_box: list):
    x = round((bounding_box[2] + bounding_box[0]))/2
    y = round((bounding_box[3] + bounding_box[1]))/2

    center_coor = [int(x), int(y)]
    # print("CENTER: ",center_coor)

    return center_coor

def hand_pose():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # For static images:
    IMAGE_FILES = []
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.flip(cv2.imread(file), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                print('hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            cv2.imwrite(
                '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
            if not results.multi_hand_world_landmarks:
                continue
            for hand_world_landmarks in results.multi_hand_world_landmarks:
                mp_drawing.plot_landmarks(
                    hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

if __name__ == "__main__":
    crop_detection("test_images/rgb.png", "test_images/raw_depth.png")
    # visualize_result("test_images/test_6.jpg")
    # camera_feed()
    # result = yolo_model()
    # print(result)