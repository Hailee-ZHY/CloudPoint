import os
import numpy as np
from glob import glob
import trimesh
import matplotlib.pyplot as plt
import torch
import open3d as o3d

class DataLoader:
    def __init__(self, data_root = "ModelNet10", split = "train", N = 2048, use_cache = False, cache_dir = "cache"):
        ## 路径和数据读取捆绑在实例上写在这里
        self.data_root = data_root
        self.split = split
        self.N = N
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.categories = self._get_categories()
        self.file_path = self._get_file_path()
        self.label_map = {cat:i for i, cat in enumerate(self.categories)}
        self.num_classes = len(self.categories)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _get_categories(self):
        cats = sorted(
            [d for d in os.listdir(self.data_root)
             if os.path.isdir(os.path.join(self.data_root,d))]
             ) # os.path.isdir 用来判断一个路径是否是一个文件夹，在这里，如果不是文件夹的话在这里我们不要

        return cats 
    
    def _get_file_path(self):
        paths = []
        for cat in self.categories:
            pattern = os.path.join(self.data_root, cat, self.split, "*.off")
            paths.extend(glob(pattern)) # glob会根据的提供的带通配符的路径返回一个所有匹配文件路径的列表
        return paths 
    
    def sample_pc(self, path):
        filename = os.path.basename(path).replace(".off", f"_{self.N}.npy")
        cache_path = os.path.join(self.cache_dir, filename)

        if self.use_cache and os.path.exists(cache_path): #如果允许缓存并且缓存文件已经存在的话，就直接读取缓存的文件，而不重新采样
            return np.load(cache_path) # np.load()是从本地读取一个文件并变成一个python中的数组，符合这里的情况

        mesh = trimesh.load_mesh(path, force_mesh = True)
        points, _ = trimesh.sample.sample_surface(mesh, self.N)

        if self.use_cache:
            np.save(cache_path, points)

        return points

    def get_label(self, path):
        category = path.split(os.sep)[-3]
        return self.label_map[category] # 返回这个类对应的编号

    def get_batch(self, path):
        pcs = []
        labels = []
        for p in path:
            pc = self.sample_pc(p)
            label = self.get_label(p)
            pcs.append(pc)
            labels.append(label)
        return np.stack(pcs), np.array(labels)  # pcs本身就是2维(N,3)，stack之后会变成3维(batch_size, N, 3)

    def augmentation(self, pointcloud, method, **kwargs):
        # at least 5 methods, and from scatch, cnanot just use 3d library
        if method == "jitter":
            noise = np.random.normal(scale = 0.05, size = pointcloud.shape) # 生成正态分布，scale代表标准差，均值默认为0
            return pointcloud + noise
        
        elif method == "scale":
            factor = np.random.uniform(0.5,1.5, size = (1,3)) # 对(x,y,z)三个维度进行不同比例的缩放
            return pointcloud*factor
        
        elif method == "rotate":
            theta = np.random.uniform(0,2*np.pi) # np.random.uniform(a,b): 从区间(a,b)生成的均匀随机分布中取一个数
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0], 
                [np.sin(theta), np.cos(theta), 0],
                [0,0,1]
            ])
            return pointcloud @ rot_matrix
        
        elif method =="flip":
            axis = kwargs.get("axis", "x")
            flipped = np.copy(pointcloud)
            if axis == 'x':
                flipped[:, 0] = -flipped[:, 0]
            elif axis == 'y':
                flipped[:, 1] = -flipped[:, 1]
            elif axis == 'z':
                flipped[:, 2] = -flipped[:, 2]
            return flipped
        
        elif method == "dropout":
            keep_ratio = kwargs.get("keep_ratio", 0.75) # default的设置是保留75%
            N = pointcloud.shape[0]
            num_keep = int(N * keep_ratio)
            indices = np.random.choice(N, num_keep, replace = False) # 从N中抽取num_keep个点
            pointcloud = pointcloud[indices]

            # 为了保持返回维度一致，比如仍然是（2048 * 3），我们可以补零或者补重复点
            if num_keep < N:
                pad = pointcloud[np.random.choice(num_keep, N-num_keep)]
                pointcloud = np.concatenate([pointcloud, pad], axis = 0)
            
            return pointcloud

    def eval_split(self, val_ratio = 0.2, shuffle = True, seed = 42):
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.file_path)

        split_idx = int(len(self.file_path) * (1-val_ratio))
        train_set = self.file_path[:split_idx]
        val_set = self.file_path[split_idx:] 
        return train_set, val_set

    def dataloader_test(self):
        print(f"Load {len(self.file_path)} files.")
        print(f"First file: {self.file_path[0]}")

        pc = self.sample_pc(self.file_path[0])
        print(f"Point cloud shape: {pc.shape}")
    

# result test
if __name__ == "__main__":
    loader = DataLoader(data_root = "ModelNet10", split = "train", N = 1024, use_cache = False, cache_dir = "cache")
    loader.dataloader_test()

