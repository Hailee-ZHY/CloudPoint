# get knn
import torch

def get_knn(x,k, exclude_self = False):
    B,N,D = x.shape
    dist = torch.cdist(x,x) # x中两两计算的E距离

    if exclude_self:
        _, idx = torch.topk(dist, k = k+1, dim = -1, largest=False, sorted=False) 
        idx = idx[:,:,1:]
    else:
        _, idx = torch.topk(dist, k = k, dim = -1, largest=False, sorted=False) # return [B,N,K]
    return idx

def index_points(points, idx):

    B, H, N, D = points.shape
    _, _, _, K = idx.shape

    idx = idx.reshape(B * H, N * K)
    points = points.reshape(B * H, N, D)

    gathered = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, D))
    return gathered.view(B, H, N, K, D)