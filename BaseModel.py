import math
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from util import *
from tqdm import tqdm
import numpy_indexed as npi
import scipy.sparse as sps
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from geoopt.optim import RiemannianAdam, RiemannianSGD


class BaseModel(nn.Module):
    def __init__(self, device=0):
        super(BaseModel, self).__init__()
        self.device = device

    def train_test(self, train_task, test_task, train_data, valid_data, test_data=None,
                   n_epochs=10, lr=0.01, n_metric=2, savepath=None, small_better=None,
                   **kwargs):
        if small_better is None:
            small_better = [False] * n_metric
        best_epoch = [-1] * n_metric
        best_metrics = [1e5 if small else 0 for small in small_better]
        self.cuda(self.device)

        if self.args.model == 'TSML':
            self.optimizer = RiemannianAdam(self.parameters(), lr=lr)
            self.load_state_dict(torch.load('restore/{}_{}_pretrain.pt'.format(self.args.model, self.args.dataset, self.embed_dim),
                                                map_location="cuda:{}".format(self.device)), strict=False)

            self.cuda(self.device)

        else:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(n_epochs):
            loss = self.fit(train_data, train_task, epoch)
            if np.isnan(loss):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                return
            print('Epoch{}\tloss: {:.6f}'.format(epoch, loss))

            metric = test_task(valid_data, **kwargs)
            for i, m in enumerate(metric):
                m = m[1]
                if (best_metrics[i] < m) ^ small_better[i]:
                    best_metrics[i], best_epoch[i] = m, epoch
                    if savepath and i == 0:
                        torch.save(self.state_dict(), savepath)
            print('Recall@[5, 10]', list(metric[0]))

        self.load_state_dict(torch.load(savepath,
                                        map_location="cuda:{}".format(self.device)), strict=False)
        self.cuda(self.device)

        metric = test_task(valid_data, True, **kwargs)
        print('final record', 'Recall@[5, 10]', metric[0], 'NDCG@[5, 10]', metric[2])


    def fit(self, data, task):
        self.train()
        c = []
        for input_batch in data:
            self.optimizer.zero_grad()
            cost = task(input_batch)
            c.append(cost.item())
            cost.backward()
            self.optimizer.step()
        return np.mean(c)

    def test_rank(self, data, final=False, **kwargs):
        self.eval()
        data, train_pos, test_pos = data
        top = kwargs['topk']
        Recall, MRR, NDCG = [[] for t in range(len(top))], [[] for t in range(len(top))], [[] for t in range(len(top))]
        with torch.no_grad():
            for input_batch in data:
                preds = self.item_scores(input_batch)
                current_pos = test_pos[input_batch.cpu().numpy()].todense()
                current_train_pos = torch.from_numpy(train_pos[input_batch.cpu().numpy()].todense()).to(
                    preds.device).float()
                preds = preds * (1 - current_train_pos) + (-1e6) * current_train_pos
                pred_list = preds.argsort(dim=1, descending=True)
                ranks = pred_list.argsort(dim=1) + 1
                current_pos_torch = torch.from_numpy(current_pos).to(ranks.device).float()
                gts = current_pos_torch
                ranks = ranks.float() * current_pos_torch + self.n_items * (1 - current_pos_torch)
                ranks = ranks.long()
                gt_num = torch.from_numpy(current_pos.sum(axis=1)).to(ranks.device).squeeze().float()
                for k in range(len(top)):
                    pred_sub_list = pred_list[:, :top[k]]
                    rank_ok = (ranks <= top[k])
                    recall = rank_ok.float().sum(dim=1) / gt_num
                    Recall[k] += list(recall.cpu().numpy())
                    if final:
                        MRR[k] += maps(pred_sub_list, gts)
                        NDCG[k] += ndcg(pred_sub_list, gts)
                    else:
                        MRR[k] += [0]
                        NDCG[k] += [0]

        for k in range(len(top)):
            Recall[k] = np.array(Recall[k]).mean()
            MRR[k] = np.array(MRR[k]).mean()
            NDCG[k] = np.array(NDCG[k]).mean()
        return [np.array(metric, dtype=float) for metric in [Recall, MRR, NDCG]]
