import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import scipy.sparse as sp
from tqdm import tqdm


class TrainDataset(Dataset):
    def __init__(self, data_path, negative_path, neg_num=20):
        data = pd.read_csv(data_path, delimiter='\t')
        self.n_item = data['iid'].max()
        negs = dict()
        with open(negative_path) as f:
            for line in f:
                l = line.strip().split('\t')
                items = [int(item) for item in l[1:]]
                negs[int(l[0])] = items
                self.n_item = max(self.n_item, max(items))

        self.n_item += 1
        self.n_user = data['uid'].max() + 1

        self.origin_data = data

        data = data.groupby('uid').iid.apply(list)
        self.data = []
        self.item_tags = []
        for uid, iids in data.items():
            uid = int(uid)
            neg = negs[uid]
            for i, iid in enumerate(iids):
                self.data.append([uid, int(iid)] + neg[i * neg_num: (i+1) * neg_num])

        self.data = np.array(self.data)

    def get_pos(self, n_item):
        pos = sp.lil_matrix((self.n_user, n_item), dtype=np.float)
        data = self.origin_data
        for uid, group in tqdm(data.groupby('uid')):
            for iid in group['iid'].tolist():
                pos[uid, iid] = 1

        return pos


    def __getitem__(self, indice):

        return self.data[indice]

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self, data_path):

        data = pd.read_csv(data_path, delimiter='\t')

        self.n_item = data['iid'].max() + 1
        self.n_user = data['uid'].max() + 1

        data_list = []
        for uid, group in data.groupby('uid'):
            data_list.append(uid)
        self.origin_data = data
        self.data = np.array(data_list)

    def get_pos(self, n_item):
        pos = sp.lil_matrix((self.n_user, n_item), dtype=np.float)
        data = self.origin_data
        for uid, group in tqdm(data.groupby('uid')):
            for iid in group['iid'].tolist():
                pos[uid, iid] = 1

        return pos
    def __getitem__(self, indice):
        return self.data[indice]

    def __len__(self):
        return len(self.data)