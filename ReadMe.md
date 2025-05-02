# Overview

This model is designed based on traditional PointNet architecture, my creative points are as below: 

1. Simplifying the original PointNet MLP pipeline

2. Removing the T-Net transformation block

3. Introducing multi-head self-attention modules

4. Applying local neighbor aggregation using k-nearest neighbors (KNN)  

# Code Organization
```
├── model.py              # Model architecture with PointNetAttn  
├── dataloader.py         # DataLoader with .off parser, sampler, augmentation  
├── utils.py              # Helper functions like get_knn and index_points  
├── train.py              # Training and evaluation scripts  
├── main.py               # Inference and visualization export (for Meshlab)  
├── sample_pc.py          # Mesh sampling function for visualization  
├── run.sh                # Bash script for training and testing  
├── checkpoints/          # Stores best_model.pth  
├── pre_submit_file/      # Generated visualization and result files (only when you run the main.py, it will apear)
```  

# Install requirements
```
pip install -r requirements.txt
```
# Pretrained Model File
Feel free to download the pretranied file from my github link:  
https://github.com/Hailee-ZHY/best_model_result.pth

# Run code 
1. Install dataset at first, and save it to the current dictionary (save level as model.py, and do NOT change the name of folder)  
2. Once dependencies are installed, run the following script in your terminal:  
``` 
    bash run.sh
```
