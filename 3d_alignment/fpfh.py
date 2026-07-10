import copy
import numpy as np
import open3d as o3d
import argparse
from scipy.spatial.transform import Rotation as R

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source point cloud (.ply)")
    parser.add_argument("--ref", required=True, help="reference/target point cloud (.ply)")
    parser.add_argument("--voxel_size", type=float, default=0.01,
                         help="downsampling voxel size in meters (tune to your object's scale)")
    return parser

def preprocess_point_cloud(pcd, voxel_size):
    """Downsample + estimate normals + compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
 
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
 
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, fpfh
 
 
def global_registration_ransac(src_down, ref_down, src_fpfh, ref_fpfh, voxel_size):
    """Coarse global alignment via FPFH feature matching + RANSAC."""
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        ref_down,
        src_fpfh,
        ref_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result
 
 
def refine_registration_icp(src, ref, initial_transform, voxel_size):
    """Fine-tune alignment via point-to-plane ICP, starting from the RANSAC result."""
    distance_threshold = voxel_size * 0.4
 
    # ICP needs normals for point-to-plane; estimate on the full-resolution clouds
    src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    ref.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
 
    result = o3d.pipelines.registration.registration_icp(
        src,
        ref,
        distance_threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result
 
 
def visualize(src, ref, transform=None, window_name="Registration result"):
    src_vis = copy.deepcopy(src)
    ref_vis = copy.deepcopy(ref)
    if transform is not None:
        src_vis.transform(transform)
    src_vis.paint_uniform_color([0.2, 0.5, 0.9])   # blue
    ref_vis.paint_uniform_color([0.9, 0.7, 0.2])   # yellow/orange
    o3d.visualization.draw_geometries([src_vis, ref_vis], window_name=window_name)
 
def decompose_transform(T):
    """
    Split a 4x4 homogeneous transform into rotation matrix + translation vector.
 
    Args:
        T: (4, 4) numpy array, or path to a .npy file containing one.
 
    Returns:
        rotation_matrix: (3, 3) numpy array
        translation: (3,) numpy array, in the same units as the input (usually meters)
    """
    if isinstance(T, str):
        T = np.load(T)
 
    rotation_matrix = T[:3, :3]
    translation = T[:3, 3]
    return rotation_matrix, translation
 
 
def decompose_transform_verbose(T):
    """
    Same as decompose_transform, but also returns human-readable rotation info
    (Euler angles in degrees, total rotation angle in degrees, translation magnitude).
    Useful for sanity-checking / debugging, not for feeding into downstream robot code.
 
    Args:
        T: (4, 4) numpy array, or path to a .npy file containing one.
 
    Returns:
        dict with keys:
            rotation_matrix    (3, 3) raw rotation matrix
            translation        (3,) raw translation vector (meters)
            translation_norm   float, total translation distance (meters)
            euler_xyz_deg      (3,) Euler angles around x, y, z axes (degrees)
            total_angle_deg    float, total rotation magnitude (degrees)
    """
    rotation_matrix, translation = decompose_transform(T)
 
    r = R.from_matrix(rotation_matrix)
    euler_xyz_deg = r.as_euler('xyz', degrees=True)
    total_angle_deg = r.magnitude() * 180 / np.pi
 
    return {
        "rotation_matrix": rotation_matrix,
        "translation": translation,
        "translation_norm": np.linalg.norm(translation),
        "euler_xyz_deg": euler_xyz_deg,
        "total_angle_deg": total_angle_deg,
    }

def start_convert_transform(transforms):

    info = decompose_transform_verbose(transforms)
 
    print("Rotation matrix:\n", info["rotation_matrix"])
    print("\nTranslation (m):", info["translation"])
    print("Translation magnitude (m):", info["translation_norm"])
    print("\nEuler angles XYZ (deg):", info["euler_xyz_deg"])
    print("Total rotation angle (deg):", info["total_angle_deg"])

def main():
    args = make_parser().parse_args()
 
    src = o3d.io.read_point_cloud(args.src)
    ref = o3d.io.read_point_cloud(args.ref)
 
    print(f"Loaded src: {len(src.points)} points, ref: {len(ref.points)} points")
 
    # --- before alignment ---
    visualize(src, ref, transform=None, window_name="BEFORE alignment")
 
    # --- preprocess: downsample + FPFH ---
    src_down, src_fpfh = preprocess_point_cloud(src, args.voxel_size)
    ref_down, ref_fpfh = preprocess_point_cloud(ref, args.voxel_size)
    print(f"Downsampled src: {len(src_down.points)} points, ref: {len(ref_down.points)} points")
 
    # --- global registration (RANSAC on FPFH matches) ---
    ransac_result = global_registration_ransac(
        src_down, ref_down, src_fpfh, ref_fpfh, args.voxel_size
    )
    print("\n--- RANSAC global registration result ---")
    print(ransac_result)
    print("Transform:\n", ransac_result.transformation)
 
    visualize(src, ref, transform=ransac_result.transformation, window_name="AFTER RANSAC (coarse)")
 
    # --- local refinement (ICP) ---
    icp_result = refine_registration_icp(
        src, ref, ransac_result.transformation, args.voxel_size
    )
    print("\n--- ICP refinement result ---")
    print(icp_result)
    print("Final transform:\n", icp_result.transformation)
 
    visualize(src, ref, transform=icp_result.transformation, window_name="AFTER ICP (refined)")
 
    # Save the final transform for downstream use (e.g. feeding into your robot arm pipeline)
    start_convert_transform(icp_result.transformation)
    np.save("estimated_transform.npy", icp_result.transformation)
    print("\nSaved final transform to estimated_transform.npy")

if __name__ == "__main__":
    print("starting")
    main()