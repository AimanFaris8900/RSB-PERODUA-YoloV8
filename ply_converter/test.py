import cv2
import numpy as np
import open3d as o3d


def convert_depth_to_ply(depth_path, ply_path, scale_factor=1000.0, fov=60.0):
    """
    Converts a raw depth image to a .ply point cloud.
    
    :param depth_path: Path to the raw depth image (e.g., 16-bit PNG or raw).
    :param ply_path: Output .ply file path.
    :param scale_factor: Factor to convert depth units to meters (default is mm to m, 1000.0).
    :param fov: Field of View in degrees to calculate camera intrinsics.
    """
    # 1. Load the depth image
    # Note: Use -1 to keep raw 16-bit depth values (do not load as RGB)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if depth_raw is None:
        raise ValueError("Could not open or find the depth image.")
    
    print("Shape:", depth_raw.shape, "Dtype:", depth_raw.dtype)

    # 2. Convert raw values to actual depth (meters)
    # This example assumes your raw values represent millimeters.
    depth_meters = depth_raw.astype(np.float32) / scale_factor

    # 3. Create Open3D Image object
    o3d_depth = o3d.geometry.Image(depth_meters)

    # 4. Define Camera Intrinsics
    # Automatically estimate intrinsic parameters based on image size and FOV
    height, width= depth_raw.shape

    # Read Orbbec Astra 2 camera intrinsics
    fx, cx, fy, cy = load_intrinsics("test_data/intrinsics.txt")
    print("HEIGHT: ", height)
    # center_x = width / 2.0
    # center_y = height / 2.0
    # # Approximate focal length from FOV
    # focal_length = width / (2 * np.tan(np.deg2rad(fov / 2)))

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )

    # 5. Create the Point Cloud from the depth image
    print("Generating point cloud...")
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=o3d_depth,
        intrinsic=intrinsic,
        depth_scale=1.0, # 1.0 because we already converted to meters
        depth_trunc=3.0, # Max depth to consider (in meters)
        stride=1         # Use stride=2 or 3 to downsample dense clouds
    )

    # Optional: Estimate normals for smoother rendering
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 6. Save as PLY
    print(f"Saving point cloud to {ply_path}")
    o3d.io.write_point_cloud(ply_path, pcd)
    print("Export complete!")

def load_intrinsics(txt_path):
    intrinsics = []

    with open(txt_path, 'r') as file:
        for line in file:
            # Strip newline characters and check if a space exists
            line = line.strip()
            if ' ' in line:
                # split(delimiter, maxsplit)
                parts = line.split(' ', 2) 
                content_after_space = parts
                print(content_after_space)

                for val in content_after_space:
                    if float(val) > 1.000:
                        intrinsics.append(float(val))

    return intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]   #fx cx fy cy

# --- Execution Example ---
if __name__ == "__main__":
    DEPTH_IMAGE = "test_data/raw_depth.png"
    OUTPUT_PLY = "test_output/output.ply"
    
    convert_depth_to_ply(DEPTH_IMAGE, OUTPUT_PLY)
