'''
DLRM Facebookresearch Debloating
author: sjoon-oh @ Github
source: dlrm/dlrm_s_pytorch.py
'''

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from torch.autograd.profiler import record_function
# from gen_hm import *

import numpy as np

### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):

    # Ctor
    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        ndevices=-1,
        loss_function="bce"
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_function=loss_function

            self.transfer_map = None # Added.
            self.idx_2_cpu = []

            # create operators
            if ndevices <= 1:
                self.emb_l, w_list = self.create_emb(m_spa, ln_emb)
                self.v_W_l = w_list

            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )


    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()

        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)

            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)

            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            n = ln[i]

            # construct embedding operator
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            # initialize embeddings
            
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)   
            ).astype(np.float32)

            # approach 1
            EE.weight.data = torch.tensor(W, requires_grad=True)

            v_W_l.append(None)

            emb_l.append(EE)
        return emb_l, v_W_l


    def apply_mlp(self, x, layers):
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups
        # print(f"apply_emb:\n  lS_o - {type(lS_o)}\n  lS_i - {type(lS_i)}")

        heatmap_on = False
        figs = []

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            
            # Added for Testing
            if heatmap_on: 
                try:
                    label = sorted(list(dict.fromkeys(sparse_index_group_batch.numpy())))
                    figs.append(HeatMapFeature(len(label)))

                    figs[-1].set_label(label)
                    figs[-1].record_map(sparse_index_group_batch)
                    figs[-1].sum_all()
                    figs[-1].show_plt()

                except Exception as e: 
                    print(e.message)

            # if k == 0: print(f"feature: {k}, index batch: \n{sparse_index_group_batch}")

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]

            if v_W_l[k] is not None:
                per_sample_weights = v_W_l[k].gather(0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            E = emb_l[k]
            V = E(
                sparse_index_group_batch,
                sparse_offset_group_batch,
                per_sample_weights=per_sample_weights,
            )

            if self.top_l.is_cuda == True:
                if k in self.idx_2_cpu:
                    ly.append(V.cuda())
                else: ly.append(V.cuda())
            else:
                ly.append(V)

        return ly

    def interact_features(self, x, ly):

        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)

            _, ni, nj = Z.shape

            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]

            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):

        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        return p


# Model Split
def model_split_cpu_gpu(
    dlrm,
    use_gpu=False,
    idx2cpu=[]
):
    print(f"model_split_cpu_gpu")
    if use_gpu:

        device = torch.device("cuda", 0) # Single GPU

        print(f"Current model: {next(dlrm.parameters()).device}")
        dlrm.top_l.cuda()
        dlrm.bot_l.cuda()

        for idx in range(len(dlrm.emb_l)):
            if idx not in idx2cpu:
                dlrm.emb_l[idx].to(device)
                # Move to GPU

    print(f"model_split_cpu_gpu done.")


# Query Split
def query_split_cpu_gpu(
    dlrm, lS_o, lS_i, idx2cpu
):
    new_lS_i = []
    new_lS_o = []

    print(f"query_split_cpu_gpu")

    for idx in range(len(dlrm.emb_l)):
        if idx not in idx2cpu:
            new_lS_o.append(lS_o[idx].cuda())
            new_lS_i.append(lS_i[idx].cuda())
        else:
            new_lS_o.append(lS_o[idx])
            new_lS_i.append(lS_i[idx])

    print(f"query_split_cpu_gpu done.")




def dlrm_wrap(dlrm, X, lS_o, lS_i, use_gpu, device, idx_2_cpu):
    if use_gpu:  # .cuda()
        # lS_i can be either a list of tensors or a stacked tensor.
        # Handle each case below:
        model_split_cpu_gpu(
            dlrm,
            use_gpu=use_gpu,
            idx2cpu=idx_2_cpu
        )

        query_split_cpu_gpu(
            dlrm,
            lS_o=lS_o,
            lS_i=lS_i,
            idx2cpu=idx_2_cpu
        )

    

    return dlrm(X.to(device), lS_o, lS_i)



def loss_fn_wrap(args, dlrm, Z, T, use_gpu, device):
    with record_function("DLRM loss compute"):
        if args.loss_function == "mse" or args.loss_function == "bce":
            return dlrm.loss_fn(Z, T.to(device))
        elif args.loss_function == "wbce":
            loss_ws_ = dlrm.loss_ws[T.data.view(-1).long()].view_as(T).to(device)
            loss_fn_ = dlrm.loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            return loss_sc_.mean()


# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr


