'''
DLRM Facebookresearch Debloating
author: sjoon-oh @ Github
source: dlrm/dlrm_data_pytorch.py
'''


from __future__ import absolute_import, division, print_function, unicode_literals

# others
from os import path
import sys
import bisect
import collections

import data_utils

# numpy
import numpy as np
from numpy import random as ra
from collections import deque


# pytorch
import torch
from torch.utils.data import Dataset, RandomSampler


# Kaggle Display Advertising Challenge Dataset
# dataset (str): name of dataset (Kaggle or Terabyte)
# split (bool) : to split into train, test, validation data-sets
class CriteoDataset(Dataset):

    def __init__(
            self,
            args,
            sub_sample_rate,
            split="train",
            raw_path="",
            pro_data="",
    ):
        # dataset
        # tar_fea = 1   # single target
        den_fea = args.den_feature_num  # 13 dense  features

        partitions = 7
        out_file = args.processed_data_file # --processed-data-file


        # split the datafile into path and filename
        lstr = raw_path.split("/")
        self.d_path = "/".join(lstr[0:-1]) + "/"
        self.d_file = lstr[-1].split(".")[0]
        self.npzfile = self.d_path + (
            (self.d_file + "_split")
        )
        self.trafile = self.d_path + (
            (self.d_file + "_fea")
        )

        # Added.
        self.transfer_map = None

        # check if pre-processed data is available
        data_ready = True

        if not path.exists(str(pro_data)):
            data_ready = False

        # pre-process data if needed
        # WARNNING: when memory mapping is used we get a collection of files
        if data_ready:
            print("Reading pre-processed data=%s" % (str(pro_data)))
            file = str(pro_data)
        else:
            print("Reading raw data=%s" % (str(raw_path)))
            file = data_utils.getCriteoAdData(
                args,
                raw_path,
                out_file,
                sub_sample_rate,
                partitions,
                split,
            )

        # get a number of samples per day
        lstr = pro_data.split("/")
        pro_path = "/".join(lstr[0:-1]) + "/"
        pro_file = lstr[-1].split(".")[0]
        total_file = pro_path + pro_file + "_part_count.npz"

        with np.load(total_file) as data:
            total_per_file = data["total_per_file"]
        # compute offsets per file
        self.offset_per_file = np.array([0] + [x for x in total_per_file])
        for i in range(partitions):
            self.offset_per_file[i + 1] += self.offset_per_file[i]
        # print(self.offset_per_file)

        # setup data
        # load and preprocess data
        with np.load(file) as data:
            X_int = data["X_int"]  # continuous  feature
            X_cat = data["X_cat"]  # categorical feature
            y = data["y"]          # target
            self.counts = data["counts"]
            
        self.m_den = X_int.shape[1]  # den_fea
        self.m_fea = X_cat.shape[1]  # Added.
        self.n_emb = len(self.counts)
        print("Sparse fea = %d, Dense fea = %d" % (self.n_emb, self.m_den))

        # create reordering
        indices = np.arange(len(y))

        if split == "none":
            # randomize all data
            
            indices = np.random.permutation(indices)
            print("Randomized indices...")

            X_int[indices] = X_int
            X_cat[indices] = X_cat
            y[indices] = y

        else:
            indices = np.array_split(indices, self.offset_per_file[1:-1])

            train_indices = np.concatenate(indices[:-1])
            test_indices = indices[-1]
            test_indices, val_indices = np.array_split(test_indices, 2)

            print("Defined %s indices..." % (split))

            # randomize train data (across partitions)
            
            train_indices = np.random.permutation(train_indices)
            print("Randomized indices across partitions ...")

            # create training, validation, and test sets
            if split == 'train':
                self.X_int = [X_int[i] for i in train_indices]
                self.X_cat = [X_cat[i] for i in train_indices]
                self.y = [y[i] for i in train_indices]
            elif split == 'val':
                self.X_int = [X_int[i] for i in val_indices]
                self.X_cat = [X_cat[i] for i in val_indices]
                self.y = [y[i] for i in val_indices]
            elif split == 'test':
                self.X_int = [X_int[i] for i in test_indices]
                self.X_cat = [X_cat[i] for i in test_indices]
                self.y = [y[i] for i in test_indices]

        print("Split data according to indices...")

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        i = index

        # return self.X_int[i], self.X_cat[i], self.y[i]
        # Added.
        return self.X_int[i], self.X_cat[i], self.y[i], self.transfer_map

    # def _default_preprocess(self, X_int, X_cat, y):
    #     X_int = torch.log(torch.tensor(X_int, dtype=torch.float) + 1)

    #     X_cat = torch.tensor(X_cat, dtype=torch.long)
    #     y = torch.tensor(y.astype(np.float32))

    #     return X_int, X_cat, y

    def __len__(self):
        return len(self.y)


def collate_wrapper_criteo_offset(list_of_tuples):
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = [X_cat[:, i] for i in range(featureCnt)]
    lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]

    return X_int, torch.stack(lS_o), torch.stack(lS_i), T


# Conversion from offset to length
def offset_to_length_converter(lS_o, lS_i):
    def diff(tensor):
        return tensor[1:] - tensor[:-1]

    return torch.stack(
        [
            diff(torch.cat((S_o, torch.tensor(lS_i[ind].shape))).int())
            for ind, S_o in enumerate(lS_o)
        ]
    )


def collate_wrapper_criteo_length(list_of_tuples):
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = torch.stack([X_cat[:, i] for i in range(featureCnt)])
    lS_o = torch.stack(
        [torch.tensor(range(batchSize)) for _ in range(featureCnt)]
    )

    lS_l = offset_to_length_converter(lS_o, lS_i)

    return X_int, lS_l, lS_i, T


def make_criteo_data_and_loaders(args, offset_to_length_converter=False):

    train_data = CriteoDataset(
        args,
        args.data_sub_sample_rate,
        "train",
        args.raw_data_file,
        args.processed_data_file,
    )

    test_data = CriteoDataset(
        args,
        args.data_sub_sample_rate,
        "test",
        args.raw_data_file,
        args.processed_data_file,
    )

    collate_wrapper_criteo = collate_wrapper_criteo_offset
    if offset_to_length_converter:
        collate_wrapper_criteo = collate_wrapper_criteo_length

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_mini_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    return train_data, train_loader, test_data, test_loader




#
# Added.

# This function should be called right after the generation of DLRM_NET
def new_make_criteo_loaders(args, train_data, test_data, transfer_map):

    # Arguments are originla train_data and test_data
    print("new_make_criteo_loaders")

    # Rearrange
    # test_data.X_cat is the list of np.array
    print("Rearranging train_data, test_data indices")
    print(f"transfer_map: {type(transfer_map)}")
    print(f"transfer_map: len - {[len(lst) for lst in transfer_map]}")
    # Transfermap : list of list

    X_cat = train_data.X_cat
    print(f"X_cat (train_data): {len(X_cat)}")
    for line in X_cat:
        for fea in range(line.shape[0]):
            if len(transfer_map[fea]) != 0:
                line[fea] = transfer_map[fea][int(line[fea])]

    X_cat = test_data.X_cat
    print(f"X_cat (test_data): {len(X_cat)}")
    for line in X_cat:
        for fea in range(line.shape[0]):
            if len(transfer_map[fea]) != 0:
                line[fea] = transfer_map[fea][int(line[fea])]

    print("Generating new train_loader and test_loader.")

    new_train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_criteo_offset,
        pin_memory=False,
        drop_last=False,  # True
    )

    new_test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_mini_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        collate_fn=collate_wrapper_criteo_offset,
        pin_memory=False,
        drop_last=False,  # True
    )

    return new_train_loader, new_test_loader


# WARNING: global define, must be consistent across all synthetic functions
cache_line_size = 1

if __name__ == "__main__":
    pass