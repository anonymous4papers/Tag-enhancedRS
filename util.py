import torch
import numpy as np

def clip_by_norm(x, clip_norm, dim=-1, inplace=True):
    norm = (x ** 2).sum(dim, keepdim=True)
    output = torch.where(norm > clip_norm ** 2, x, x * clip_norm / (norm + 1e-6))
    if inplace:
        x = output
    return output

def mrr(ranks):
    return torch.reciprocal(ranks.float()).sum().item()

def _dcg_support(size):
    arr = np.arange(1, size+1)+1
    return 1./np.log2(arr)

def ndcg_one(vector_true_dense, vector_predict, hits):
    idcg = np.sum(_dcg_support(len(vector_true_dense)))
    dcg_base = _dcg_support(len(vector_predict))
    dcg_base[np.logical_not(hits)] = 0
    dcg = np.sum(dcg_base)
    return dcg/idcg

def map_one(vector_predict, hits):
    precisions = np.cumsum(hits, dtype=np.float32)/range(1, len(vector_predict)+1)
    return np.mean(precisions)

def ndcg(ranklists, gts):
    ndcgs = []
    ranklists = ranklists.cpu()
    gts = gts.cpu()
    for i in range(len(ranklists)):
        gt_list = gts[i].nonzero()[:, -1].numpy()
        ranklist = ranklists[i].numpy()
        hits = np.isin(ranklist, gt_list)
        if len(gt_list) != 0:
            ndcgs.append(ndcg_one(gt_list, ranklist, hits))
    return ndcgs

def maps(ranklists, gts):
    maps = []
    ranklists = ranklists.cpu()
    gts = gts.cpu()
    for i in range(len(ranklists)):
        gt_list = gts[i].nonzero()[:, -1].numpy()
        ranklist = ranklists[i].numpy()
        hits = np.isin(ranklist, gt_list)
        if len(gt_list) != 0:
            maps.append(map_one(ranklist, hits))
    return maps


def padding(batch, pad=0):
    lens = [len(session) for session in batch]
    len_max = max(lens)
    batch = [session + [pad] * (len_max - l) for session, l in zip(batch, lens)]
    return batch

def mod_value(matrix):
    return torch.norm(matrix, 2, dim=-1).unsqueeze(-1)

def expmap(x, u):
    v = u - x
    norm_v = v.norm(p=2, dim=-1, keepdim=True) / x[0, -1]
    exp = x * torch.cos(norm_v) + v * torch.sin(norm_v) / norm_v
    return exp

def euclidean_distances(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return dist
