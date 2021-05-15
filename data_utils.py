'''
DLRM Facebookresearch Debloating
author: sjoon-oh @ Github
source: dlrm/data_utils.py
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
# import os
from os import path
from multiprocessing import Process, Manager

import numpy as np

def processCriteoAdData(args, d_path, d_file, npzfile, i, convertDicts, pre_comp_counts):
    # Process Kaggle Display Advertising Challenge or Terabyte Dataset
    # by converting unicode strings in X_cat to integers and
    # converting negative integer values in X_int.
    #
    # Loads data in the form "{kaggle|terabyte}_split_i.npz" where i is the split.
    #
    # Inputs:
    #   d_path (str): path for {kaggle|terabyte}_split_i.npz files
    #   i (int): splits in the dataset (typically 0 to 7 or 0 to 24)

    # process data if not all files exist
    filename_i = npzfile + "_{0}_processed.npz".format(i)

    if path.exists(filename_i):
        print("Using existing " + filename_i, end="\n")
    else:
        print("Not existing " + filename_i)
        with np.load(npzfile + "_{0}.npz".format(i)) as data:
            # categorical features

            # Approach 2a: using pre-computed dictionaries
            X_cat_t = np.zeros(data["X_cat_t"].shape)
            for j in range(args.cat_feature_num):
                for k, x in enumerate(data["X_cat_t"][j, :]):
                    X_cat_t[j, k] = convertDicts[j][x]
            # continuous features
            X_int = data["X_int"]
            X_int[X_int < 0] = 0
            # targets
            y = data["y"]

        np.savez_compressed(
            filename_i,
            # X_cat = X_cat,
            X_cat=np.transpose(X_cat_t),  # transpose of the data
            X_int=X_int,
            y=y,
        )
        print("Processed " + filename_i, end="\n")
    return


def concatCriteoAdData(
        args,
        d_path,
        d_file,
        npzfile,
        trafile,
        splits,
        data_split,
        total_per_file,
        total_count,
        o_filename
):
    # Concatenates different splits and saves the result.
    #
    # Inputs:
    #   splits (int): total number of splits in the dataset (typically 7 or 24)
    #   d_path (str): path for {kaggle|terabyte}_split_i.npz files
    #   o_filename (str): output file name
    #
    # Output:
    #   o_file (str): output file path

    print("Concatenating multiple splits into %s file" % str(o_filename))

    # load and concatenate data
    for i in range(splits):
        filename_i = npzfile + "_{0}_processed.npz".format(i)
        with np.load(filename_i) as data:
            if i == 0:
                X_cat = data["X_cat"]
                X_int = data["X_int"]
                y = data["y"]
            else:
                X_cat = np.concatenate((X_cat, data["X_cat"]))
                X_int = np.concatenate((X_int, data["X_int"]))
                y = np.concatenate((y, data["y"]))
        print("Loaded split:", i, "y = 1:", len(y[y == 1]), "y = 0:", len(y[y == 0]))

    with np.load(d_path + d_file + "_fea_count.npz") as data:
        counts = data["counts"]
    print("Loaded counts!")

    np.savez_compressed(
        o_filename,
        X_cat=X_cat,
        X_int=X_int,
        y=y,
        counts=counts,
    )

    return o_filename



def getCriteoAdData(
        args,
        datafile, # raw path
        o_filename,
        sub_sample_rate=0.0,
        splits=7,
        data_split='train'
):

    #split the datafile into path and filename

    # lstr = datafile.split("/")
    lstr = o_filename.split("/")
    d_path = "/".join(lstr[0:-1]) + "/"

    print(f"d_path: {d_path}")

    d_file = lstr[-1].split(".")[0]
    npzfile = d_path + (d_file + "_part")
    trafile = d_path + (d_file + "_fea")

    # count number of datapoints in training set
    total_file = d_path + d_file + "_part_count.npz"

    if path.exists(total_file):
        with np.load(total_file) as data:
            total_per_file = list(data["total_per_file"])
        total_count = np.sum(total_per_file)
        print("Skipping counts per file (already exist)")

    else:
        total_count = 0
        total_per_file = []

        # WARNING: The raw data consists of a single train.txt file
        # Each line in the file is a sample, consisting of 13 continuous and
        # 26 categorical features (an extra space indicates that feature is
        # missing and will be interpreted as 0).

        if path.exists(datafile):
            print("Reading data from path=%s" % (datafile))
            with open(str(datafile)) as f:
                for _ in f:
                    total_count += 1
            total_per_file.append(total_count)
            # reset total per file due to split
            num_data_per_split, extras = divmod(total_count, splits)
            total_per_file = [num_data_per_split] * splits
            for j in range(extras):
                total_per_file[j] += 1
            # split into splits (simplifies code later on)
            file_id = 0
            boundary = total_per_file[file_id]
            nf = open(npzfile + "_" + str(file_id), "w")
            with open(str(datafile)) as f:
                for j, line in enumerate(f):
                    if j == boundary:
                        nf.close()
                        file_id += 1
                        nf = open(npzfile + "_" + str(file_id), "w")
                        boundary += total_per_file[file_id]
                    nf.write(line)
            nf.close()
        else:
            sys.exit("ERROR: Criteo Kaggle Display Ad Challenge Dataset path is invalid; please download from https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset")
        
        
    # process a file worth of data and reinitialize data
    # note that a file main contain a single or multiple splits
    def process_one_file(
            datfile,
            npzfile,
            split,
            num_data_in_split,
            convertDictssplit=None,
            resultsplit=None
    ):

        with open(str(datfile)) as f:
            y = np.zeros(num_data_in_split, dtype="i4")  # 4 byte int
            X_int = np.zeros((num_data_in_split, args.den_feature_num), dtype="i4")  # 4 byte int
            X_cat = np.zeros((num_data_in_split, args.cat_feature_num), dtype="i4")  # 4 byte int
            if sub_sample_rate == 0.0:
                rand_u = 1.0
            else:
                rand_u = np.random.uniform(low=0.0, high=1.0, size=num_data_in_split)

            i = 0
            percent = 0
            for k, line in enumerate(f):
                # process a line (data point)
                line = line.split('\t')
                # set missing values to zero
                for j in range(len(line)):
                    if (line[j] == '') or (line[j] == '\n'):
                        line[j] = '0'
                # sub-sample data by dropping zero targets, if needed
                target = np.int32(line[0])
                if target == 0 and \
                   (rand_u if sub_sample_rate == 0.0 else rand_u[k]) < sub_sample_rate:
                    continue

                y[i] = target
                X_int[i] = np.array(line[1:args.den_feature_num + 1], dtype=np.int32)

                X_cat[i] = np.array(
                    list(map(lambda x: int(x, 16), line[args.den_feature_num + 1:])),
                    dtype=np.int32
                )

                # count uniques
                for j in range(args.cat_feature_num):
                    convertDicts[j][X_cat[i][j]] = 1
                # debug prints
                print(
                    "Load %d/%d  Split: %d  Label True: %d  Stored: %d"
                    % (
                        i,
                        num_data_in_split,
                        split,
                        target,
                        y[i],
                    ),
                    end="\r",
                )
                i += 1

            # store num_data_in_split samples or extras at the end of file
            filename_s = npzfile + "_{0}.npz".format(split)
            if path.exists(filename_s):
                print("\nSkip existing " + filename_s)
            else:
                np.savez_compressed(
                    filename_s,
                    X_int=X_int[0:i, :],
                    # X_cat=X_cat[0:i, :],
                    X_cat_t=np.transpose(X_cat[0:i, :]),  # transpose of the data
                    y=y[0:i],
                )
                print("\nSaved " + npzfile + "_{0}.npz!".format(split))


        return i

    # create all splits (reuse existing files if possible)
    recreate_flag = False
    convertDicts = [{} for _ in range(args.cat_feature_num)]
    # WARNING: to get reproducable sub-sampling results you must reset the seed below

    # in this case there is a single split in each split
    for i in range(splits):
        npzfile_i = npzfile + "_{0}.npz".format(i)
        npzfile_p = npzfile + "_{0}_processed.npz".format(i)
        if path.exists(npzfile_i):
            print("Skip existing " + npzfile_i)
        elif path.exists(npzfile_p):
            print("Skip existing " + npzfile_p)
        else:
            recreate_flag = True

    if recreate_flag:
        for i in range(splits):
            total_per_file[i] = process_one_file(
                npzfile + "_{0}".format(i),
                npzfile,
                i,
                total_per_file[i],
            )

    # report and save total into a file
    total_count = np.sum(total_per_file)
    if not path.exists(total_file):
        np.savez_compressed(total_file, total_per_file=total_per_file)
    print("Total number of samples:", total_count)
    print("Divided into splits/splits:\n", total_per_file)

    # dictionary files
    counts = np.zeros(args.cat_feature_num, dtype=np.int32)
    if recreate_flag:
        # create dictionaries
        for j in range(args.cat_feature_num):
            for i, x in enumerate(convertDicts[j]):
                convertDicts[j][x] = i
            dict_file_j = d_path + d_file + "_fea_dict_{0}.npz".format(j)
            if not path.exists(dict_file_j):
                np.savez_compressed(
                    dict_file_j,
                    unique=np.array(list(convertDicts[j]), dtype=np.int32)
                )
            counts[j] = len(convertDicts[j])
        # store (uniques and) counts
        count_file = d_path + d_file + "_fea_count.npz"
        if not path.exists(count_file):
            np.savez_compressed(count_file, counts=counts)
    else:
        # create dictionaries (from existing files)
        for j in range(args.cat_feature_num):
            with np.load(d_path + d_file + "_fea_dict_{0}.npz".format(j)) as data:
                unique = data["unique"]
            for i, x in enumerate(unique):
                convertDicts[j][x] = i
        # load (uniques and) counts
        with np.load(d_path + d_file + "_fea_count.npz") as data:
            counts = data["counts"]

    # process all splits

    for i in range(splits):
        processCriteoAdData(args, d_path, d_file, npzfile, i, convertDicts, counts)

    o_file = concatCriteoAdData(
        args,
        d_path,
        d_file,
        npzfile,
        trafile,
        splits,
        data_split,
        total_per_file,
        total_count,
        o_filename
    )

    return o_file

if __name__ == "__main__":
    pass
