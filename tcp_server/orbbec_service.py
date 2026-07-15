from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, FrameSet, OBStreamType, AlignFilter

BOX_SCALE = 5
MIN_DEPTH_MM = 100  # Clip depth closer than this (mm)
MAX_DEPTH_MM = 5000  # Clip depth farther than this (mm)

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

def realtime_bb_data(bb_results):
    center = None
    if bb_results:
        bounding_box = bb_results[0]
        center = calculate_center(bounding_box)

    return center

def draw_bounding_box(bb_results, frame, frame_size):
    frame_width = frame_size[0]
    frame_height = frame_size[1]

    if bb_results:
        bounding_box = bb_results[0]
        print("BOUNDING BOX: ", bounding_box)
        frame = cv2.rectangle(frame, (int(bounding_box[0]),int(bounding_box[1])), (int(bounding_box[2]), int(bounding_box[3])), color=(0,0,255))
        center = calculate_center(bounding_box)
        print("CENTER VAR: ", center)

        # BB CENTER
        frame = cv2.circle(frame, (center[0], center[1]), 5, (0,0,255), -1)

        # FRAME CENTER
        frame = cv2.circle(frame, (int(frame_width/2), int(frame_height/2)), 5, (0,255,0), -1)

        # BOX POINTS
        frame = cv2.circle(frame, (int(bounding_box[0]), int(bounding_box[1])), 5, (255,0,0), -1)
        frame = cv2.circle(frame, (int(bounding_box[2]), int(bounding_box[3])), 5, (0,255,0), -1)

        # BB CENTER TEXT
        frame = cv2.putText(frame, f"BB Center: {center[0]}x{center[1]}", (500,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 250), 3)
    
    # TEXTS
    frame = cv2.putText(frame, f"Frame Center: {frame_width/2}x{frame_height/2}", (500,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    return frame

def frame_to_bgr(color_frame) -> np.ndarray:
    """Convert an Orbbec color frame to an OpenCV BGR image, handling common formats."""
    width = color_frame.get_width()
    height = color_frame.get_height()
    data = np.asarray(color_frame.get_data())
    fmt = color_frame.get_format()
 
    if fmt == OBFormat.RGB:
        img = data.reshape((height, width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        return data.reshape((height, width, 3))
    elif fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.YUYV:
        img = data.reshape((height, width, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    else:
        raise ValueError(f"Unsupported color format: {fmt}")

def camera_orrbec_stream():
    pipeline, align_filter = start_camera_pipeline()
 
    try:
        while True:
            frames: FrameSet = pipeline.wait_for_frames(5000)  # timeout ms
            if frames is None:
                continue

            aligned_frames = align_filter.process(frames)
            if aligned_frames is None:
                continue
 
            color_frame = aligned_frames.get_color_frame()
            if color_frame is None:
                continue

            depth_frame = aligned_frames.get_depth_frame()
            if depth_frame is None:
                continue

            get_depth_data(depth_frame)
 
            try:
                bgr_image = frame_to_bgr(color_frame)
                results = yolo_model(bgr_image)

                frame_size = [color_frame.get_width(), color_frame.get_height()]
                bb_img = draw_bounding_box(results, bgr_image, frame_size)

            except ValueError as e:
                print(e)
                continue
 
            cv2.imshow("Orbbec RGB Stream", bb_img)
            # cv2.imshow("Orbbec RGB Stream", bgr_image)
            if cv2.waitKey(1) in (ord('q'), 27):  # 'q' or ESC to quit
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def start_camera_pipeline():
    pipeline = Pipeline()
    config = Config()
 
    try:
        # Pick the color stream profile (default resolution/fps from the device)
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        # Explicitly select Y16 depth profile — default_video_stream_profile() picks RLE
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = None
        for i in range(depth_profile_list.get_count()):
            p = depth_profile_list.get_stream_profile_by_index(i)
            if (p.get_format() == OBFormat.Y16
                    and p.get_width() == 1600
                    and p.get_height() == 1200
                    and p.get_fps() == 30):
                depth_profile = p
                break

        if depth_profile is None:
            raise RuntimeError("Y16 1600x1200@30fps depth profile not found")

        config.enable_stream(depth_profile)
    except Exception as e:
        print(f"Error accessing camera streams: {e}")
        return

    pipeline.start(config)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    print(f"Streaming color: {color_profile.get_width()}x{color_profile.get_height()} "
          f"@ {color_profile.get_fps()}fps, format={color_profile.get_format()}")

    return pipeline, align_filter

def camera_data_stream(pipeline: Pipeline, align_filter: AlignFilter, depth = False):

    frames: FrameSet = pipeline.wait_for_frames(5000)  # timeout ms
    if frames is None:
        return None

    aligned_frames = align_filter.process(frames)
    if aligned_frames is None:
        return None

    color_frame = aligned_frames.get_color_frame()
    if color_frame is None:
        return None

    depth_frame = aligned_frames.get_depth_frame()
    if depth_frame is None:
        return None

    center_dist = get_depth_data(depth_frame)

    if depth:
        return depth_frame
    
    return color_frame
    

def get_port_bbox(color_frame):
    try:
        bgr_image = frame_to_bgr(color_frame)
        results = yolo_model(bgr_image)

        frame_size = [color_frame.get_width(), color_frame.get_height()]
        center_port = realtime_bb_data(results)

        return frame_size, center_port

    except ValueError as e:
        pass

def get_depth_data(frame: FrameSet):
    depth_data = frame.get_data()
    width = frame.get_width()
    height = frame.get_height()
    scale = frame.get_depth_scale()  # e.g. 0.1 → 1 unit = 0.1 mm

    raw = np.frombuffer(frame.get_data(), dtype=np.uint16)
    depth_mm = raw.reshape(height, width).astype(np.float32) * scale

    print(f"Depth format: {frame.get_format()}")
    print(f"Data length: {len(frame.get_data())}, expected raw: {width*height*2}")

    print(f"DEPTH CAM WIDTH HEIGHT: {width} {height} {depth_data}")

    cy, cx = height // 2, width // 2
    center_dist = depth_mm[cy, cx]
    in_range = MIN_DEPTH_MM <= center_dist <= MAX_DEPTH_MM
    dist_label = f"{center_dist:.0f} mm" if in_range else "out of range"
    print(f"Distance at ({cx}, {cy}): {center_dist} mm")

    return center_dist

    # x, y = int(width/2), int(height/2)

    # # Calculate 1D index
    # if 0 <= x < width and 0 <= y < height:
    #     index = y * width + x
    #     depth_distance_mm = depth_data[index]
    #     print(f"Distance at ({x}, {y}): {depth_distance_mm} mm")

def calculate_center(bounding_box: list):
    x = round((bounding_box[2] + bounding_box[0]))/2
    y = round((bounding_box[3] + bounding_box[1]))/2

    center_coor = [int(x), int(y)]
    # print("CENTER: ",center_coor)

    return center_coor

if __name__ == "__main__":
    # crop_detection("test_images/rgb.png", "test_images/raw_depth.png")
    # visualize_result("test_images/test_6.jpg")
    # camera_feed()
    # result = yolo_model()
    # print(result)
    camera_orrbec_stream()
