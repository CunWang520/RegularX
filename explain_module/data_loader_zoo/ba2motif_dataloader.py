import os.path as osp
import pickle as pkl

import torch
import random
import numpy as np
from torch_geometric.data import InMemoryDataset, Data


class BA2Motif(InMemoryDataset):
    splits = ['training', 'evaluation', 'testing']

    def __init__(self, root, mode='testing', transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.mode = mode
        super(BA2Motif, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['BA-2motif.pkl']

    @property
    def processed_file_names(self):
        return ['training.pt', 'evaluation.pt', 'testing.pt']

    def download(self):
        if not osp.exists(osp.join(self.raw_dir, 'raw', 'BA-2motif.pkl')):
            print("raw data of `BA-2motif` doesn't exist, please redownload from our github.")
            raise FileNotFoundError

    def process(self):
        with open(osp.join(self.raw_dir, self.raw_file_names[0]), 'rb') as fin:
            adjs, features, labels = pkl.load(fin)

        data_list = []
        for idx, (adj, feature, label) in enumerate(zip(adjs, features, labels)):
            edge_index = torch.from_numpy(np.array(np.nonzero(adj)))
            x = torch.from_numpy(feature)
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.from_numpy(label).argmax().unsqueeze(dim=0)
            ground_truth_mask = (edge_index[0] >= 20) & (edge_index[1] >= 20)
            data = Data(x=x, y=y,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        ground_truth_mask=ground_truth_mask,
                        name=f'BA-2Motif{idx}', idx=idx)


            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        assert len(data_list) == 1000

        random.shuffle(data_list)
        torch.save(self.collate(data_list[400:]), self.processed_paths[0])
        torch.save(self.collate(data_list[200:400]), self.processed_paths[1])
        torch.save(self.collate(data_list[:200]), self.processed_paths[2])
