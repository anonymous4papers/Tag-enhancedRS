from torch import nn
import torch.nn.functional as F
from util import *
from BaseModel import BaseModel
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import torch
import math
import pandas as pd

class TSML(BaseModel):
    def __init__(self, n_users, n_items, device=0, embed_dim=20, margin=1.5, rand_seed=12345, args=None):
        super(TSML, self).__init__(device)
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.rand_seed = rand_seed
        self.args = args
        self.margin = margin

        tags = pd.read_csv('data/' + args.dataset + '/item_cat.dat', sep='\t')
        n_tags = 0
        for i in range(len(tags)):
            tag = str(tags.iloc[i]['tag'])
            tag_list = tag.split(',')
            for t in tag_list:
                if int(t) + 1 > n_tags:
                    n_tags = int(t) + 1

        self.n_tags = n_tags
        self.tag_labels = torch.zeros(len(tags), n_tags)
        for i in range(len(tags)):
            item = tags.iloc[i]['iid']
            tag = str(tags.iloc[i]['tag'])
            tag_list = tag.split(',')
            for t in tag_list:
                self.tag_labels[int(item), int(t)] = 1
            self.tag_labels[int(item)] = self.tag_labels[int(item)] / self.tag_labels[int(item)].sum()
        self.tag_labels = self.tag_labels.cuda(self.device)
        self.user_pref = nn.Embedding(self.n_users, self.n_tags)
        self.tag_embeddings = nn.Embedding(self.n_tags, self.embed_dim)
        for emb in [self.user_pref, self.tag_embeddings]:
            nn.init.normal_(emb.weight.data, mean=0.0, std=0.01)

        self.user_embeddings = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.n_items, self.embed_dim)
        self.user_margin = nn.Parameter(torch.ones(n_users))
        self.item_margin = nn.Parameter(torch.ones(n_items))
        for emb in [self.user_embeddings, self.item_embeddings]:
            nn.init.normal_(emb.weight.data,  mean=0.0, std=0.01)


    def _embedding_loss(self, X):
        users = self.user_embeddings(X[:, 0])
        pos_items = self.item_embeddings(X[:, 1])
        neg_items = self.item_embeddings(X[:, 2:])
        distance_to_neg_items = torch.sum((users.unsqueeze(1) * neg_items), dim=-1)

        user_tag_pref = self.user_pref(X[:, 0])
        user_tag_embedding = torch.softmax(user_tag_pref, dim=1).mm(self.tag_embeddings.weight)
        item_tags = self.tag_labels[X[:, 1:]]
        item_tag_embeddings = item_tags.matmul(self.tag_embeddings.weight)

        user_tag_embedding = user_tag_embedding / mod_value(user_tag_embedding)
        item_tag_embeddings = item_tag_embeddings / mod_value(item_tag_embeddings)

        neg_tag_items = item_tag_embeddings[:, 1:]
        distance_to_neg_items_tag = torch.sum(user_tag_embedding.unsqueeze(1) * neg_tag_items, dim=-1)
        distance_to_neg_items += distance_to_neg_items_tag

        closest_negative_item_indices = distance_to_neg_items.max(1)[1]
        closest_neg_items = torch.gather(X[:, 2:], dim=1, index=closest_negative_item_indices.view(-1, 1)).squeeze()
        closest_neg_items_embedding = self.item_embeddings(closest_neg_items)


        pos_distances = torch.sum(users * pos_items, dim=1)
        neg_distances = torch.sum(users * closest_neg_items_embedding, dim=1)
        pos_to_neg_distances = torch.torch.sum(pos_items * closest_neg_items_embedding, dim=-1)

        neg_tag_embedding = self.tag_labels[closest_neg_items].matmul(self.tag_embeddings.weight)
        neg_tag_embedding = neg_tag_embedding / mod_value(neg_tag_embedding)

        pos_to_neg_distances_tag = torch.sum(item_tag_embeddings[:, 0, :] * neg_tag_embedding, dim=-1)
        pos_distances_tag = torch.sum(user_tag_embedding * item_tag_embeddings[:, 0, :], dim=-1)
        neg_distances_tag = torch.sum(user_tag_embedding * neg_tag_embedding, dim=-1)

        pos_distances += pos_distances_tag
        neg_distances += neg_distances_tag
        pos_to_neg_distances += pos_to_neg_distances_tag

        u_margin = self.user_margin[X[:, 0]]
        p_margin = self.item_margin[X[:, 1]]

        distance2 = - pos_distances + pos_to_neg_distances + p_margin
        distance1 = - pos_distances + neg_distances + u_margin
        distance1 = F.relu(distance1)
        distance2 = F.relu(distance2)

        margin_reg = self.user_margin.mean() + self.item_margin.mean()
        loss = distance1 + 0.01 * distance2 - 10 * margin_reg
        loss += - self.args.nuc_reg * torch.norm(self.user_pref(X[:, 0]), 'nuc')
        return loss.sum()

    def tag_pair_loss(self, user_tag_embeddings):
        tag_embeddings = user_tag_embeddings / mod_value(user_tag_embeddings)
        distances = tag_embeddings.matmul(tag_embeddings.t())
        tag_pair_loss = distances
        tag_pair_loss = torch.log(1 + torch.exp(tag_pair_loss))
        return self.args.tag_pair * torch.triu(tag_pair_loss, diagonal=1).sum()

    def loss(self, X):
        X = X.to(self.device)
        embedding_loss = self._embedding_loss(X)
        loss = embedding_loss

        return loss

    def fit(self, data, task, epoch):
        self.train()
        c = []
        for input_batch in tqdm(data):
            self.optimizer.zero_grad()
            loss = task(input_batch)
            c.append(loss.item())
            loss += self.tag_pair_loss(self.tag_embeddings.weight.data)
            loss.backward()

            self.optimizer.step()
            self.user_margin.data.clamp_(min=0.0, max=self.margin)
            self.item_margin.data.clamp_(min=0.0, max=self.margin)

        return np.mean(c)
    def item_scores(self, score_user_ids):
        score_user_ids = score_user_ids.to(self.device)
        user = self.user_embeddings(score_user_ids)
        item = self.item_embeddings.weight.data
        score = user.matmul(item.t())

        user_tag_pref = self.user_pref(score_user_ids)
        user_tag_embedding = (torch.softmax(user_tag_pref, dim=1)).mm(self.tag_embeddings.weight)
        user_tag_embedding = user_tag_embedding / mod_value(user_tag_embedding)
        item_tags = self.tag_labels
        item_tag_embeddings = item_tags.mm(self.tag_embeddings.weight)
        item_tag_embeddings = item_tag_embeddings / mod_value(item_tag_embeddings)
        score += user_tag_embedding.matmul(item_tag_embeddings.t())

        return score

