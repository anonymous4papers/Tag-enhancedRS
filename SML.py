from torch import nn
import torch.nn.functional as F
from util import *
from BaseModel import BaseModel
from tqdm import tqdm
import numpy as np
import torch
import math
import pandas as pd

class SML(BaseModel):
    def __init__(self, n_users, n_items, device=0, embed_dim=20, margin=1.5, rand_seed=12345, args=None):
        super(SML, self).__init__(device)
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.rand_seed = rand_seed
        self.clip_norm = 1.0
        self.args = args
        self.margin = margin

        self.user_embeddings = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.n_items, self.embed_dim)
        self.user_margin = nn.Parameter(self.margin * torch.ones(n_users))
        self.item_margin = nn.Parameter(self.margin * torch.ones(n_items))
        for emb in [self.user_embeddings, self.item_embeddings]:
            nn.init.normal_(emb.weight.data,  mean=0.0, std=0.01)

    def _embedding_loss(self, X):
        users = self.user_embeddings(X[:, 0])
        pos_items = self.item_embeddings(X[:, 1])
        pos_distances = torch.sum((users - pos_items) ** 2, 1)

        neg_items = self.item_embeddings(X[:, 2:]).transpose(-2, -1)
        distance_to_neg_items = torch.sum((users.unsqueeze(-1) - neg_items) ** 2, 1)

        closest_negative_item_distances = distance_to_neg_items.min(1)[0]
        closest_negative_item_indices = distance_to_neg_items.min(1)[1]
        closest_neg_items = torch.gather(X[:, 2:], dim=1, index=closest_negative_item_indices.view(-1, 1)).squeeze()
        closest_neg_items_embedding = self.item_embeddings(closest_neg_items)
        pos_to_neg_distances = torch.sum((pos_items - closest_neg_items_embedding) ** 2, 1)

        u_margin = self.user_margin[X[:, 0]]
        p_margin = self.item_margin[X[:, 1]]

        distance1 = pos_distances - closest_negative_item_distances + u_margin
        distance2 = pos_distances - pos_to_neg_distances + p_margin
        distance1 = F.relu(distance1)
        distance2 = F.relu(distance2)
        margin_reg = self.user_margin.mean() + self.item_margin.mean()

        loss = distance1.sum() + 0.01 * distance2.sum() - 10 * margin_reg

        return loss

    def loss(self, X):
        X = X.to(self.device)
        embedding_loss = self._embedding_loss(X)
        loss = embedding_loss
        return loss

    def clip_by_norm_op(self):
        return [clip_by_norm(self.user_embeddings.weight.data, self.clip_norm),
                clip_by_norm(self.item_embeddings.weight.data, self.clip_norm)]

    def fit(self, data, task, epoch):
        self.train()
        c = []
        for input_batch in tqdm(data):
            self.optimizer.zero_grad()
            loss = task(input_batch)
            c.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.clip_by_norm_op()
            self.user_margin.data.clamp_(min=0.0, max=self.margin)
            self.item_margin.data.clamp_(min=0.0, max=self.margin)
        return np.mean(c)
    def item_scores(self, score_user_ids):
        score_user_ids = score_user_ids.to(self.device)
        user = self.user_embeddings(score_user_ids)
        items = self.item_embeddings.weight.data

        score = -euclidean_distances(user, items)

        return score