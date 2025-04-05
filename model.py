# reference PointNet++, DGCNN 

import torch
import torch.nn as nn 
import torch.nn.functional as F 

from utils import get_knn, index_points

## multi-head sel-attention, the key part in my model

class SelfAttention(nn.Module):
    def __init__(self, dim, dropout = 0.25, k = 16, heads = 4, exclude_self = False): # i set defaul as not exclude self
        super(SelfAttention, self).__init__() 

        self.query_proj = nn.Linear(dim,dim)
        self.key_proj = nn.Linear(dim,dim)
        self.value_proj = nn.Linear(dim,dim)

        self.softmax = nn.Softmax(dim = -1)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.k = k 
        self.exclude_self = exclude_self

        self.heads = heads
        self.head_dim = dim//heads
        self.merge_heads = nn.Linear(dim,dim)
    
    def forward(self, x): # x = [B,N,dim]
        B, N, D = x.shape

        Q = self.query_proj(x).view(B,N,self.heads, self.head_dim).permute(0,2,1,3) # [B,H,N,d_k]
        K = self.key_proj(x).view(B,N,self.heads, self.head_dim).permute(0,2,1,3)
        V = self.value_proj(x).view(B,N,self.heads, self.head_dim).permute(0,2,1,3)

        idx = get_knn(x, k=self.k, exclude_self=self.exclude_self)  # [B, N, K]
        idx = idx.unsqueeze(1).expand(-1, self.heads, -1, -1)  # [B, H, N, K]

        neighbours_K = index_points(K, idx)  # [B, H, N, K, d_k]
        neighbours_V = index_points(V, idx)

        d_k = self.head_dim

        # Q: [B,N,d_k] -> [B,N,1,d_k] 
        Q = Q.unsqueeze(3)
        attn_scores = (Q*neighbours_K).sum(-1) / (d_k**0.5) # [B,N,K]
        # attn_scores = torch.bmm(Q, K.transpose(1,2)) / (Q.shape[-1]**0.5) # [B,N,N] -- bmm = batch matrix multiplication
        attn_weights = self.softmax(attn_scores) # [B,H,N,K]
        attn_output = (attn_weights.unsqueeze(-1) * neighbours_V).sum(dim=3)  # (B, H, N, d_k)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, D)  # (B, N, D)
        out = self.norm(x + self.dropout(self.merge_heads(attn_output))) # residual add, avoid overfitting
        return out 


## 目前pointnet和原文不太一样，是简化版本，先测试自己的功能
class PointNetAttn(nn.Module):
    def __init__(self, num_classes = 10):
        super(PointNetAttn, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(3,64),
            nn.ReLU(), 
            nn.Linear(64,128),
            nn.ReLU()
        )

        self.attn1 = SelfAttention(dim = 128, k = 16, heads = 4, exclude_self=False)
        self.attn2 = SelfAttention(dim = 128, k = 16, heads = 4, exclude_self=False)

        self.mlp2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x): # x = [B,N,3]
        x = self.mlp1(x) # [B,N,128]
        # print(f"[debug] Input to attn1: {x.shape}")
        x = self.attn1(x) # [B,N,128]
        x = self.attn2(x) # [B,N,128]
        x = self.mlp2(x) # [B,N,1024]
        x = torch.max(x, dim = 1)[0]
        out = self.fc(x) # [B,num_classes]
        return out

# dummy test
if __name__ == "__main__":
    model = PointNetAttn(num_classes=10)
    dummy_input = torch.randn(4,2048,3) # [b,N,dim]
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}")

