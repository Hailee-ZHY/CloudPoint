import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os

def cloud_sample(path_to_mesh, N_1 = 2048, N_2 = 6144, output_roots = "output"):
    # path_to_mesh = "ModelNet10/bathtub/train/bathtub_0001.off" 

    # Load the mesh (supports .off, .obj, .stl, etc.)
    mesh = trimesh.load_mesh(path_to_mesh, force_mesh=True)  # Complete this! 这里是不是只读一个就可以了？
    # Export the mesh to a new PLY file for use in MeshLab
    mesh_name = os.path.splitext(os.path.basename(path_to_mesh))[0]
    save_dir = os.path.join(output_roots, mesh_name)
    os.makedirs(save_dir, exist_ok=True)
    mesh_output_path = os.path.join(save_dir, "mesh.ply")
    mesh.export(mesh_output_path)

    # Sample N points uniformly on the surface
    # N = 2048
    # Method 1:
    points_1, _ = trimesh.sample.sample_surface(mesh, N_1)

    # Method 2:
    # Compute face areas
    areas = mesh.area_faces
    probabilities = areas / areas.sum()  # Normalize

    # Sample faces proportionally to area
    face_indices = np.random.choice(len(mesh.faces), size=N_2, p=probabilities)

    # Barycentric coordinate sampling within faces
    barycentric_coords = np.random.dirichlet((1,1,1), size=N_2)
    vertices = mesh.vertices[mesh.faces[face_indices]]
    points_2 = (vertices * barycentric_coords[:, :, None]).sum(axis=1)

    # Convert sampled points to Open3D PointCloud for saving
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points_1)
    pcd2.points = o3d.utility.Vector3dVector(points_2)
    o1_path = os.path.join(save_dir, "sampled_points_1.ply")
    o2_path = os.path.join(save_dir, "sampled_points_2.ply")

    o3d.io.write_point_cloud(o1_path, pcd1)
    o3d.io.write_point_cloud(o2_path, pcd2)

    # print(f"saved mesh and could points to: {save_dir}")


if __name__ == "__main__":
    path_to_mesh = [
        r".\ModelNet10\bathtub\train\bathtub_0001.off",
        r".\ModelNet10\bed\train\bed_0003.off",
        r".\ModelNet10\chair\train\chair_0005.off",
        r".\ModelNet10\desk\train\desk_0006.off",
        r".\ModelNet10\sofa\train\sofa_0009.off",
    ]

    for mesh_path in path_to_mesh:
        cloud_sample(mesh_path)