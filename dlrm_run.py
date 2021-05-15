'''
DLRM Facebookresearch Debloating
author: sjoon-oh @ Github
source: dlrm/dlrm_s_pytorch.py
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# miscellaneous
import builtins
import datetime
import json
import sys
import time

# data generation
import dlrm_data as dp
from dlrm_net import *

# numpy
import numpy as np

# pytorch
import torch
import torch.nn as nn
from torch._ops import ops
from torch.nn.parameter import Parameter

import optim.rwsadagrad as RowWiseSparseAdagrad

exc = getattr(builtins, "IOError", "FileNotFoundError")

#
# ---
# Function: time_wrap
def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()

#
# ---
# Function: dash_separated_ints
def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value

#
# ---
# Function: dash_separated_floats
def dash_separated_floats(value):
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )

    return value

#
# ---
# Function: get_split_lengths
# Moved from extended_distributed.py
def get_split_lengths(n):
    k, m = divmod(n, 1)
    if m == 0:
        splits = None
        my_len = k
    else:
        splits = [(k + 1) if i < m else k for i in range(1)]
        my_len = splits[my_rank]
    return (my_len, splits)


#
# ---
# Function: inference
def inference(
    args,
    dlrm,
    best_acc_test,
    best_auc_test,
    test_ld,
    device,
    use_gpu,
    log_iter=-1,
):
    test_accu = 0
    test_samp = 0

    for i, testBatch in enumerate(test_ld):
        # early exit if nbatches was set by the user and was exceeded
        if nbatches > 0 and i >= nbatches:
            break

        X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
            testBatch
        )

        # forward pass
        Z_test = dlrm_wrap(
            dlrm,
            X_test,
            lS_o_test,
            lS_i_test,
            use_gpu,
            device,
            ndevices=ndevices,
        )

        if Z_test.is_cuda:
            torch.cuda.synchronize()

        (_, batch_split_lengths) = get_split_lengths(X_test.size(0))

        # compute loss and accuracy
        S_test = Z_test.detach().cpu().numpy()  # numpy array
        T_test = T_test.detach().cpu().numpy()  # numpy array

        mbs_test = T_test.shape[0]  # = mini_batch_size except last
        A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

        test_accu += A_test
        test_samp += mbs_test

    acc_test = test_accu / test_samp
    # writer.add_scalar("Test/Acc", acc_test, log_iter)

    model_metrics_dict = {
        "nepochs": args.nepochs,
        "nbatches": nbatches,
        "nbatches_test": nbatches_test,
        "state_dict": dlrm.state_dict(),
        "test_acc": acc_test,
    }

    is_best = acc_test > best_acc_test
    if is_best:
        best_acc_test = acc_test
    print(
        " accuracy {:3.3f} %, best {:3.3f} %".format(
            acc_test * 100, best_acc_test * 100
        ),
        flush=True,
    )
    return model_metrics_dict, is_best

#
# ---
# Function: prepare_parser
def prepare_parser():
    
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)

    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)

    # activations and loss
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-workers", type=int, default=0)

    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time", action="store_true", default=False)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")

    parser.add_argument("--max-ind-range", type=int, default=-1)

    parser.add_argument("--cat-feature-num", type=int, default=0)
    parser.add_argument("--den-feature-num", type=int, default=0)

    parser.add_argument("--idx-2-gpu", nargs='+', default=None)

    return parser

#
# ---
# Function: run
def run():

    ### parse arguments ###
    parser = prepare_parser()

    global args
    global nbatches
    global nbatches_test
    
    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()

    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True

        ngpus = torch.cuda.device_count()
        device = torch.device("cuda", 0)

        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data

    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    table_feature_map = {idx: idx for idx in range(len(train_data.counts))}
    nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
    nbatches_test = len(test_ld)

    ln_emb = train_data.counts
    # enforce maximum limit on number of vectors per embedding

    print(f"Embeddings count: {ln_emb}")

    ln_emb = np.array(ln_emb)
    m_den = train_data.m_den
    ln_bot[0] = m_den


    args.ln_emb = ln_emb.tolist()

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )

    if m_spa != m_den_out:
        sys.exit(
            "ERROR: arch-sparse-feature-size "
            + str(m_spa)
            + " does not match last dim of bottom mlp "
            + str(m_den_out)
        )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    global ndevices
    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1
    # added
    ndevices = 1

    print(f"ndevices: {ndevices}")

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    global dlrm
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        ndevices=ndevices,
        loss_function=args.loss_function
    )

    # if use_gpu:
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        # dlrm = dlrm.to(device)  # .cuda()


    
    #
    #
    # Added.
    if args.idx_2_gpu == None: args.idx_2_gpu = []
    else:
        args.idx_2_gpu = [int(_) for _ in args.idx_2_gpu]

    print(f"idx_2_gpu: {args.idx_2_gpu}")
    
    idx_2_cpu = [_ for _ in range(len(dlrm.emb_l)) if _ not in args.idx_2_gpu]
    print(f"idx_2_cpu: {idx_2_cpu}")
    dlrm.idx_2_cpu = idx_2_cpu # Register

    import k_means
    embedding_sizes = [int(_.num_embeddings) for _ in dlrm.emb_l]
    cluster_enable = [False for _ in dlrm.emb_l]
    # Set embedding_sizes to False if not want to cluster
    
    for idx in range(len(cluster_enable)):
        cluster_enable[idx] = True if idx in args.idx_2_gpu else False
    
    transfer_map = \
        k_means.generate_transfer_map(
            train_dataset=train_data,
            ln_emb=embedding_sizes,
            n_clusters=100,
            enable=cluster_enable
        )    

    # Exchange to index-transferred DataLoaders
    train_ld, test_ld = \
        dp.new_make_criteo_loaders(
            args, 
            train_data=train_data,
            test_data=test_data,
            transfer_map=transfer_map
        )

    # Register to dlrm_net
    if dlrm.transfer_map == None:
        dlrm.transfer_map = transfer_map

    #
    # -- end

    if not args.inference_only:
        if use_gpu and args.optimizer in ["rwsadagrad", "adagrad"]:
            sys.exit("GPU version of Adagrad is not supported by PyTorch.")
        # specify the optimizer algorithm
        opts = {
            "sgd": torch.optim.SGD,
            "rwsadagrad": RowWiseSparseAdagrad.RWSAdagrad,
            "adagrad": torch.optim.Adagrad,
        }

        parameters = (
            dlrm.parameters()
        )
        optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)
        lr_scheduler = LRPolicyScheduler(
            optimizer,
            0, # args.lr_num_warmup_steps,
            0, # args.lr_decay_start_step,
            0, # args.lr_num_decay_steps,
        )

    ### main loop ###

    # training or inference
    best_acc_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_iter = 0
    total_samp = 0


    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device("cuda")
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(args.load_model, map_location=torch.device("cpu"))
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_train_loss = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]

        ld_acc_test = ld_model["test_acc"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_acc_test = ld_acc_test
            total_loss = ld_total_loss
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}".format(
                ld_train_loss,
            )
        )

        print("Testing state: accuracy = {:3.3f} %".format(ld_acc_test * 100))

    if args.inference_only:
        # Currently only dynamic quantization with INT8 and FP16 weights are
        # supported for MLPs and INT4 and INT8 weights for EmbeddingBag
        # post-training quantization during the inference.
        # By default we don't do the quantization: quantize_{mlp,emb}_with_bit == 32 (FP32)

        pass

    print("time/loss/accuracy (if enabled):")


    # tb_file = "./" + args.tensor_board_filename
    # writer = SummaryWriter(tb_file)

    with torch.autograd.profiler.profile(
        enabled=False,
        use_cuda=use_gpu, record_shapes=True
    ) as prof:
        if not args.inference_only:
            k = 0
            total_time_begin = 0
            while k < args.nepochs:
                if k < skip_upto_epoch:
                    continue

                for j, inputBatch in enumerate(train_ld):

                    if j < skip_upto_batch:
                        continue

                    X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

                    t1 = time_wrap(use_gpu)

                    # early exit if nbatches was set by the user and has been exceeded
                    if nbatches > 0 and j >= nbatches:
                        break

                    mbs = T.shape[0]  # = args.mini_batch_size except maybe for last

                    # forward pass
                    Z = dlrm_wrap(
                        dlrm,
                        X,
                        lS_o,
                        lS_i,
                        use_gpu,
                        device,
                        ndevices=ndevices,
                    )

                    # loss
                    E = loss_fn_wrap(args, dlrm, Z, T, use_gpu, device)

                    # compute loss and accuracy
                    L = E.detach().cpu().numpy()  # numpy array
                    
                    optimizer.zero_grad()

                    # backward pass
                    E.backward()

                    # optimizer
                    optimizer.step()
                    lr_scheduler.step()

                    t2 = time_wrap(use_gpu)
                    total_time += t2 - t1

                    total_loss += L * mbs
                    total_iter += 1
                    total_samp += mbs

                    should_print = ((j + 1) % args.print_freq == 0) or (
                        j + 1 == nbatches
                    )
                    should_test = (
                        (args.test_freq > 0)
                        and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                    )

                    # print time, loss and accuracy
                    if should_print or should_test:
                        gT = 1000.0 * total_time / total_iter if args.print_time else -1
                        total_time = 0

                        train_loss = total_loss / total_samp
                        total_loss = 0

                        str_run_type = (
                            "inference" if args.inference_only else "training"
                        )

                        wall_time = ""
                        if args.print_wall_time:
                            wall_time = " ({})".format(time.strftime("%H:%M"))

                        print(
                            "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                                str_run_type, j + 1, nbatches, k, gT
                            )
                            + " loss {:.6f}".format(train_loss)
                            + wall_time,
                            flush=True,
                        )

                        log_iter = nbatches * k + j + 1
                        # writer.add_scalar("Train/Loss", train_loss, log_iter)

                        total_iter = 0
                        total_samp = 0

                    # testing
                    if should_test:
                        epoch_num_float = (j + 1) / len(train_ld) + k + 1
                        print(
                            "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k)
                        )
                        model_metrics_dict, is_best = inference(
                            args,
                            dlrm,
                            best_acc_test,
                            best_auc_test,
                            test_ld,
                            device,
                            use_gpu,
                            log_iter,
                        )

                        if (
                            is_best
                            and not (args.save_model == "")
                            and not args.inference_only
                        ):
                            model_metrics_dict["epoch"] = k
                            model_metrics_dict["iter"] = j + 1
                            model_metrics_dict["train_loss"] = train_loss
                            model_metrics_dict["total_loss"] = total_loss
                            model_metrics_dict[
                                "opt_state_dict"
                            ] = optimizer.state_dict()
                            print("Saving model to {}".format(args.save_model))
                            torch.save(model_metrics_dict, args.save_model)

                k += 1  # nepochs
            
        else:
            print("Testing for inference only")
            inference(
                args,
                dlrm,
                best_acc_test,
                best_auc_test,
                test_ld,
                device,
                use_gpu,
            )

    total_time_end = time_wrap(use_gpu)


if __name__ == "__main__":
    run()
