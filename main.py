from model import PointNetAttn
from sample_pc import cloud_sample
from dataloader import DataLoader

import torch
import numpy as np
import random
from glob import glob
import os 
import open3d as o3d
import trimesh

class run_main():
    def __init__(self):
        # load model 
        self.loader_test = DataLoader(data_root = "ModelNet10", split = "test", N = 1024, use_cache = True, cache_dir = "cache/test") # 这里要导一下train的数据
        self.loader_train = DataLoader(data_root = "ModelNet10", split = "train", N = 1024, use_cache = True, cache_dir = "cache/train")
        self.model = PointNetAttn(num_classes=self.loader_test.num_classes).to(self.loader_test.device)
        self.model.load_state_dict(torch.load(r"checkpoints\best_model.pth"))
        self.out_puts = "pre_submit_file"
        self.path_to_off_test = [random.choice(glob(f"ModelNet10/{cat}/test/*.off")) for cat in self.loader_test.categories]
        self.path_to_off_train = [random.choice(glob(f"ModelNet10/{cat}/train/*.off")) for cat in self.loader_train.categories]
        self.aug_methods = ["jitter", "scale", "rotate", "dropout", "flip"]

    # 1. create visualization file for meshlab
    def densities_compare(self):
        save_path = os.path.join(self.out_puts, "2densities")
        os.makedirs(save_path, exist_ok=True)
        for off_path in self.path_to_off_train[:5]:
            cloud_sample(off_path, N_1 = 2048, N_2 = 6144, output_roots = save_path)

    # 2. create visualization file for augmentation
    def augmentation_visualization(self):
        output_root = os.path.join(self.out_puts, "augmentation_results")

        for path in self.path_to_off_train[:5]:
            category = path.split(os.sep)[1].replace(".off", "")
            base_name = os.path.splitext(os.path.basename(path))[0]
            original_pc = self.loader_train.sample_pc(path)
            # print("category", category)
            # print("base_name", base_name)

            mesh = trimesh.load_mesh(path, force = "mesh")
            mesh_save_dir = os.path.join(output_root, category, "mesh")
            os.makedirs(mesh_save_dir, exist_ok=True)

            mesh_name = os.path.splitext(os.path.basename(path))[0]
            mesh.export(os.path.join(mesh_save_dir, f"{mesh_name}.ply"))
            
            for method in self.aug_methods:
                aug1 = self.loader_train.augmentation(original_pc, method = method)
                aug2 = self.loader_train.augmentation(original_pc, method = method)

                save_dir = os.path.join(output_root, category, method)
                # print("save_dir", save_dir)
                os.makedirs(save_dir, exist_ok=True)
            
                pcd_0 = o3d.geometry.PointCloud()
                pcd_1 = o3d.geometry.PointCloud()
                pcd_2 = o3d.geometry.PointCloud()

                pcd_0.points = o3d.utility.Vector3dVector(original_pc.astype(np.float32))
                pcd_1.points = o3d.utility.Vector3dVector(aug1.astype(np.float32))
                pcd_2.points = o3d.utility.Vector3dVector(aug2.astype(np.float32))

                o3d.io.write_point_cloud(os.path.join(save_dir, f"{base_name}_original.ply"), pcd_0)  
                o3d.io.write_point_cloud(os.path.join(save_dir, f"{base_name}_{method}_1.ply"), pcd_1) 
                o3d.io.write_point_cloud(os.path.join(save_dir, f"{base_name}_{method}_2.ply"), pcd_2) 


    # 3. sample data and classification via model 
    def sample_prediction(self):
        result_dir = os.path.join(self.out_puts, "predictions")
        os.makedirs(result_dir, exist_ok= True)

        save_path = os.path.join(self.out_puts, "prediction_results.txt")
        with open(save_path, "w") as f:
            for idx, path in enumerate(self.path_to_off_test):
                points = self.loader_test.sample_pc(path)
                base_name = os.path.splitext(os.path.basename(path))[0]
                cat = os.path.normpath(path).split(os.sep)[1]
                sample_dir = os.path.join(result_dir, cat)
                os.makedirs(sample_dir, exist_ok=True)

                # save mesh
                mesh = trimesh.load_mesh(path, force = "mesh")
                mesh.export(os.path.join(sample_dir, f"{base_name}_mesh.ply"))

                # save sampled point cloud
                points = self.loader_test.sample_pc(path)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
                o3d.io.write_point_cloud(os.path.join(sample_dir, f"{base_name}_points.ply"), pcd)

                # in model
                self.model.eval()
                points_tensor = torch.tensor(points).unsqueeze(0).float().to(self.loader_test.device)
                with torch.no_grad():
                    logits = self.model(points_tensor)
                    probs = torch.softmax(logits, dim = 1)
                    top3_probs, top3_indx = torch.topk(probs, k = 3)

                f.write(f"{idx+1}-{cat} {base_name}:\n")
                for i in range(3):
                    pred_class = self.loader_test.categories[top3_indx[0][i].item()]
                    score = top3_probs[0][i].item()
                    f.write(f"Top {i+1}: {pred_class}({score: .2f})\n")
                f.write("\n")
            

if __name__ == "__main__":
    run = run_main()
    run.densities_compare()
    run.augmentation_visualization()
    run.sample_prediction()