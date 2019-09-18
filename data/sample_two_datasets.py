"""
Returns a relative equal sample from two datasets. Most samplers sample probabilistically from two datasets such that
the samples are still imbalanced. This sampler returns a data based on the ratio of availability.
"""
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import math


class DummyDataset(Dataset):
    def __init__(self, data):
        super(DummyDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EqualSampler(Sampler):
    def __init__(self, dataset, num_data, num_other_data, batch_size):
        super(EqualSampler, self).__init__(data_source=dataset)
        self.num_data = num_data
        self.num_other_data = num_other_data
        self.ratio = num_data / (num_data + num_other_data)
        self.batch_size = batch_size
        self.num_total = math.ceil((self.num_data + self.num_other_data) / self.batch_size) * self.batch_size

    def __iter__(self):
        idxs = list()
        data = torch.randperm(self.num_data)
        other_data = torch.randperm(self.num_other_data) + self.num_data  # add an offset

        num_data_used = 0
        num_other_data_used = 0
        data_idx = 0
        other_data_idx = 0
        for _ in range(self.num_total):
            num_used = num_data_used + num_other_data_used
            if (num_used * self.ratio - num_data_used) >= (num_used * (1 - self.ratio) - num_other_data_used):
                if data_idx >= self.num_data:  # reset to 0, basically wrap around
                    data_idx = 0
                idxs.append(data[data_idx].item())
                data_idx += 1
                num_data_used += 1
            else:
                if other_data_idx >= self.num_other_data:
                    other_data_idx = 0  # reset to 0
                idxs.append(other_data[other_data_idx].item())
                other_data_idx += 1
                num_other_data_used += 1
        return iter(idxs)

    def __len__(self):
        return self.num_total


def execute():
    data = np.arange(0, 999)
    batch_size = 100
    other_data = np.arange(1000, 5000)
    all_data = np.concatenate([data, other_data], axis=0)
    dataset = DummyDataset(all_data)
    sampler = EqualSampler(dataset=dataset, num_data=len(data), num_other_data=len(other_data), batch_size=batch_size)
    data_loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    print(sampler.ratio)
    for i, data in enumerate(data_loader):
        print(data.shape)
        print('iteration {}'.format(i))
        batch_num_data = torch.sum(data < 999)
        batch_num_other_data = torch.sum(data >= 1000)
        print('batch_num_data: {}'.format(batch_num_data.item()))
        print('batch_other_num_data: {}'.format(batch_num_other_data.item()))


if __name__ == '__main__':
    execute()
