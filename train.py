import torch
import torch.nn as nn 
import os 
from dataloader import DataLoader
import numpy as np
import argparse
import torch.optim as optim
from tqdm import tqdm 
from collections import defaultdict
import random

from dataloader import DataLoader
from model import PointNetAttn

def train(model, optimizer, loss_fn, train_paths, batch_size, loader, device, num_classes, use_aug):
    model.train()
    total_loss, total_correct = 0, 0
    correct_per_class, total_per_class = defaultdict(int), defaultdict(int)
    aug_methods = ["jitter", "scale", "rotate", "dropout", "shift"]

    np.random.shuffle(train_paths) # 这里不是split train和val的时候的shuffle, 而是在每个epoch内部shuffle，因为这里的train写的是一个epoch的逻辑
    for i in tqdm(range(0,len(train_paths),batch_size)):
        batch_paths = train_paths[i:i+batch_size]
        points, labels = loader.get_batch(batch_paths)

        if use_aug:   # 这里的if后面不能用continue，因为continue就会跳过本次循环，不要忘记了，这个if外面还有一个循环
            for i in range(len(points)):  
                method = random.choice(aug_methods)
                points[i] = loader.augmentation(points[i], method = method)

        points = torch.tensor(points, dtype = torch.float32).to(device)
        labels = torch.tensor(labels, dtype = torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(points) # 调用的是forward, 所以传入的是数据
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step() # 根据当前的梯度信息更新模型参数，e.g. Adam

        total_loss += loss.item() * points.size(0)  # loss.item() 得到的是当前batch的 mean loss
        preds = outputs.argmax(dim = 1)
        total_correct += (preds == labels).sum().item() # 这里 .item() 的作用是将一个0维的标量tensor转换为python数值，方便和普通变量做加法等操作
        
        for l, p in zip(labels, preds):
                total_per_class[l.item()] += 1
                if l == p:
                    correct_per_class[l.item()] += 1
        
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_paths)
    acc = total_correct / len(train_paths)

    per_class_acc = {}
    for cls in range(num_classes):
        if total_per_class[cls] > 0:
            per_class_acc[cls] = correct_per_class[cls] / total_per_class[cls]
        else:
            per_class_acc[cls] = None # no samples


    return avg_loss, acc, per_class_acc

def evaluation(model, loss_fn, eval_paths, batch_size, loader, device, num_classes):
    model.eval()

    total_loss, total_correct = 0, 0 
    correct_per_class, total_per_class = defaultdict(int), defaultdict(int)

    with torch.no_grad():
        for i in range(0, len(eval_paths), batch_size):
            batch_paths = eval_paths[i:i+batch_size]
            points, labels = loader.get_batch(batch_paths)
            points = torch.tensor(points, dtype = torch.float32).to(device)
            labels = torch.tensor(labels, dtype = torch.long).to(device)

            outputs = model(points)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * points.size(0) 
            preds = outputs.argmax(dim = 1)
            total_correct += (preds == labels).sum().item()

            for l, p in zip(labels, preds):
                total_per_class[l.item()] += 1
                if l == p:
                    correct_per_class[l.item()] += 1

            torch.cuda.empty_cache()

    avg_loss = total_loss / len(eval_paths)
    acc = total_correct / len(eval_paths)

    per_class_acc = {}
    for cls in range(num_classes):
        if total_per_class[cls] > 0:
            per_class_acc[cls] = correct_per_class[cls] / total_per_class[cls]
        else:
            per_class_acc[cls] = None # no samples

    return avg_loss, acc, per_class_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  ## 创建一个解析器对象
    parser.add_argument('--epoches', type = int, default=20)  # 这里的epoch相当于就是调用的变量名了
    parser.add_argument('--batch_size', type = int, default=8)
    parser.add_argument('--lr', type = float, default=0.001)
    parser.add_argument('--device', type = str, default= 'cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_aug', type = bool, default=True)
    args = parser.parse_args() #将上面写入的hyperparameter都打包到args这个对象里

    loader = DataLoader()
    train_paths, eval_paths = loader.eval_split()
    model = PointNetAttn(num_classes=len(loader.categories)).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr) # 把model的所有参数都传入Adam进行优化
    loss_fn = nn.CrossEntropyLoss()
    num_classes = loader.num_classes

    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epoches+1):
        train_loss, train_acc, train_per_class_acc = train(model, optimizer,loss_fn, train_paths, args.batch_size, loader, args.device, num_classes, args.use_aug)
        eval_loss, eval_acc, eval_per_class_acc = evaluation(model, loss_fn, eval_paths, args.batch_size, loader, args.device, num_classes)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {eval_loss:.4f}, Acc: {eval_acc:.4f}")

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")

            # save loss, acc, per-class acc for the best model
            with open("checkpoints/best_model_metrics.txt", "w") as f:
                f.write(f"Best Model Mtrics:\n")
                f.write(f"Train Loss: {train_loss:.4f}\n")
                f.write(f"Train Accuracy: {train_acc*100: 2f}%\n")
                f.write(f"Evaluarion Loss: {eval_loss: 4f}\n")
                f.write(f"Evaluation Accuracy: {eval_acc*100: 2f}%\n")

                f.write(f"Per-Class Accuracy (Train):\n")
                for cls_id, acc in train_per_class_acc.items():
                    class_name = loader.categories[cls_id]
                    f.write(f"{class_name:12s}:{acc*100:.2f}%\n")
                
                f.write(f"Per-Class Accuracy (Evaluarion):\n")
                for cls_id, acc in eval_per_class_acc.items():
                    class_name = loader.categories[cls_id]
                    f.write(f"{class_name:12s}:{acc*100:.2f}%\n")

    
    print("training finshed and best model has been saved!")

