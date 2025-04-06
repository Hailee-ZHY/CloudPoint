# Overview

This project is an extension of the PointNet architecture, integrating attention mechanisms and local feature modeling inspired by DGCNN. The main objective is to improve point cloud classification accuracy while maintaining efficiency. This is done by:

Simplifying the original PointNet MLP pipeline

Removing the T-Net transformation block

Introducing multi-head self-attention modules

Applying local neighbor aggregation using k-nearest neighbors (KNN)

# Key Features

Simplified Backbone:

Two-layer MLP for initial feature embedding

Global max pooling after extracting local-global features

Self-Attention Module:

Two stacked multi-head self-attention blocks

Query-Key-Value projection and residual connections

KNN used to gather local neighboring features

Training Enhancements:

On-the-fly data augmentation (jittering, scaling, rotation, shifting, dropout)

Per-class accuracy computation

Automatic best model checkpointing based on validation accuracy

# File Structure

.
├── model.py               # Model architecture with PointNetAttn
├── dataloader.py         # DataLoader with .off parser, sampler, augmentation
├── utils.py              # Helper functions like get_knn and index_points
├── train.py              # Training and evaluation scripts
├── main.py               # Inference and visualization export (for Meshlab)
├── sample_pc.py          # Mesh sampling for visualization
├── run.sh                # Bash script for training and testing
├── checkpoints/          # Stores best_model.pth
├── pre_submit_file/      # Generated visualization and result files

# Requirements

Python >= 3.7

PyTorch >= 1.10

numpy, trimesh, open3d, matplotlib

tqdm

# Training

bash run.sh

This will:

Train the model and save best weights to checkpoints/best_model.pth

Run the main.py script to sample point clouds, perform prediction, and export .ply files and results.

# Evaluation Output

Classification results with top-3 predictions and confidence scores per test mesh

Visualization-ready files for Meshlab:

Mesh and two densities (2048, 6144 points)

Augmentation variations per method

# Unique Contributions

Introduced self-attention to PointNet for better global-local feature fusion

Used KNN to build local neighborhoods for attention (inspired by DGCNN)

Achieved modular and readable architecture for experimentation

# Reference

PointNet: Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"

DGCNN: Wang et al., "Dynamic Graph CNN for Learning on Point Clouds"

