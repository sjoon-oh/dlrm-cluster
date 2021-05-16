'''
DLRM Facebookresearch K-Means++
author: sjoon-oh @ Github
source: -
'''

import torch
import torch.nn as nn

import numpy as np
from sklearn.cluster import KMeans

import os.path
import json

class ClusterManager():
    def __init__(self, 
            n_clusters, # Given as integer
            n_embeddings, # Given as list
            lS_i, # Given as list of torch.Tensor 
            # (data_size, # of categorical features)
            file_name='transfer_map'
        ):

        self.n_clusters = n_clusters # Number of cluster set
        self.n_embeddings = n_embeddings # List of # of embeddings
        self.n_features = len(n_embeddings)
        self.index_transfer_maps = [[] for _ in range(self.n_features)] # Initialize to empty one.
        self.transfer_offsets = [] # 

        self.file_name = file_name

        self.queries = lS_i

        # Assumes lS_i is not batch, but a full query
        self.k_means_models = [
            KMeans(n_clusters=self.n_clusters, init='k-means++') \
                for _ in range(len(n_embeddings))
        ]

    def doClusterSingle2(self, index):
        print(f"Training K-Means using GPU for {index}!")

        fpath = f'{self.file_name}-{index}.json'

        print(f'Searching for {fpath}')

        if os.path.exists(fpath): 
            print(f"Skipping clustering for feature {index}")
            with open(fpath, 'r') as json_file:
                tmap = json.load(json_file)
                self.index_transfer_maps[index] = tmap[str(index)]

        else:

            from libKMCUDA import kmeans_cuda

            train_q = self.queries[index].numpy().reshape(-1, 1)
            dummy_axis = np.zeros(train_q.shape[0]).reshape(-1, 1) 

            train_q = np.concatenate((train_q, dummy_axis), axis=1).astype(np.float32)

            centroids, assignments = kmeans_cuda(
                train_q,
                self.n_clusters,
                verbosity=1,
                yinyang_t=0,
                tolerance=0.002
            )

            cid = assignments.tolist()
            
            _ = [{} for nc in range(max(cid) + 1)] # Extract only unique indices

            for idx in train_q[:][0]:
                _[cid[idx]][idx] = 1

            print(f"  Reconstructing unique maps done.")
            del train_q
            _ = [list(cid_s.keys()) for cid_s in _]

            for cluster in _:
                self.index_transfer_maps[index].extend(cluster)
            del _

            _ = list(range(1, self.n_embeddings[index] + 1))
            for idx in self.index_transfer_maps[index]: 
                _[idx] = 0
                print(f"\r  Searching unseen... {idx}", end='')
            self.index_transfer_maps[index].extend([x for x in _ if x != 0])

            # Generate Backup

            with open(fpath, 'w') as json_file:
                json.dump({index: self.index_transfer_maps[index]}, json_file)
            del _
        
        print(f"  Embeddings: {self.n_embeddings[index]}")
        print(f"  index_transfer_maps[{index}] size: {len(self.index_transfer_maps[index])}")
        print(f"Feature {index} done.\n")


    # Deprecated. Too slow.
    def doClusterSingle(self, index):
        print(f"Training K-Means for {index}!")
        fpath = f'{self.file_name}-{index}.json'

        print(f'Searching for {fpath}')

        if os.path.exists(fpath): 
            print(f"Skipping clustering for feature {index}")
            with open(fpath, 'r') as json_file:
                tmap = json.load(json_file)
                self.index_transfer_maps[index] = tmap[str(index)]

        else:

            train_q = self.queries[index].numpy().reshape(-1, 1)

            self.k_means_models[index].fit(train_q) # Train
            cid = self.k_means_models[index].predict(train_q) # Record

            print(f"  cid size: {len(cid)}")
            print(f"  cid max: {max(cid)}")

            _ = [{} for nc in range(max(cid) + 1)] # Extract only unique indices
            train_q = train_q.reshape(1, -1).tolist()[0]

            for idx in train_q:
                _[cid[idx]][idx] = 1

            print(f"  Reconstructing unique maps done.")
            del train_q
            _ = [list(cid_s.keys()) for cid_s in _]

            for cluster in _:
                self.index_transfer_maps[index].extend(cluster)
            del _

            _ = list(range(1, self.n_embeddings[index] + 1))
            for idx in self.index_transfer_maps[index]: 
                _[idx] = 0
                print(f"\r  Searching unseen... {idx}", end='')
            self.index_transfer_maps[index].extend([x for x in _ if x != 0])

            # Generate Backup

            with open(fpath, 'w') as json_file:
                json.dump({index: self.index_transfer_maps[index]}, json_file)
            del _
        
        print(f"  Embeddings: {self.n_embeddings[index]}")
        print(f"  index_transfer_maps[{index}] size: {len(self.index_transfer_maps[index])}")
        print(f"Feature {index} done.\n")

    # Deprecated
    def doCluster(self):
        for fea in range(self.n_features):
            self.doClusterSingle(fea)


    def visualize(self):
        pass


    def get_transfer_map(self):
        return self.index_transfer_maps





# Below here lies test code. 
#

#
# Assumes this runs on CPU
# def initialize():
    
import dlrm_run as run

show_plt = False

#
# Reuse the dlrm_run
def initialize():

    parser = run.prepare_parser()
    args = parser.parse_args()

    args.arch_sparse_feature_size = 16
    args.arch_mlp_bot = "13-512-256-64-16"
    args.arch_mlp_top = "512-256-1"
    
    args.raw_data_file = "dataset/Kaggle/train.txt"
    args.processed_data_file = "dataset-processed/Kaggle/kaggle.npz"

    args.loss_function = "bce"
    args.round_targets = True
    args.learning_rate = 0.1
    args.mini_batch_size = 512

    args.test_mini_batch_size = 1024
    args.test_num_workers = 2
    args.den_feature_num = 13
    args.cat_feature_num = 26

    return args


#
# Console
def console(msg):
    print(f"> {msg}")


#
#
def generate_transfer_map(train_dataset, ln_emb, n_clusters, enable, file_name):
    console("generate_transfer_map")

    X_cat = torch.tensor(train_dataset.X_cat, dtype=torch.long)
    console(f"Local X_cat: {X_cat.shape}")

    #
    # Prepare query
    # Test is done for feature 0 only.
    lS_i = [X_cat[:, i] for i in range(train_dataset.n_emb)]

    console(f"lS_i len: {len(lS_i)}")
    console(f"lS_i element type: {type(lS_i[0])}")

    cl_manager = ClusterManager(
        n_clusters=n_clusters, 
        n_embeddings=ln_emb, 
        lS_i=lS_i,
        file_name=file_name
    )

    for fea in range(len(ln_emb)):
        if enable[fea]:
            cl_manager.doClusterSingle2(fea)
        
    print(f"Returning index_transfer_maps")
    return cl_manager.index_transfer_maps



#
# Unit test
if __name__ == '__main__':

    # This is a test code!
    print('K-Means Clustering Test Snippet\nAuthor: SukJoon Oh\n')
    exit()

    args = initialize() # Load
    nbatches = args.mini_batch_size

    model_path = 'model/model-kaggle.pt'    
    dlrm = torch.load(model_path)
    dlrm_state_dict = dlrm['state_dict']

    # Embeddingbags Key:
    # emb_l.0.weight
    # emb_l.1.weight
    # emb_l.2.weight
    # emb_l.3.weight amd so on.    
    fea_keys = [f'emb_l.{feature}.weight' for feature in range(26)]

    #
    # Reconstruct Embeddings
    emb_weights = [
        dlrm_state_dict[fea_keys[_]] for _ in range(args.cat_feature_num)
    ]

    emb_l = []
    for fea in range(len(emb_weights)):
        emb_l.append(
            nn.EmbeddingBag(
                num_embeddings=emb_weights[fea].shape[0],
                embedding_dim=emb_weights[fea].shape[1],
            )
        )
        emb_l[fea].weight.data = emb_weights[fea]

    console(f"EmbeddingBag {fea_keys} loaded:\n  {emb_l}")
    console(f"Calling dp.make_criteo_data_and_loaders...\n")

    import dlrm_data as dp
    
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    console(f"Done.")

    generate_transfer_map(
        train_data, 
        [int(_.num_embeddings) for _ in emb_l],
        3
        )

    exit()