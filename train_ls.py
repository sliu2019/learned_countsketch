"""
Same as train_speedup.py file; attempting applying gradients directly to sketch_value, instead of manual gradient application
"""
import argparse
import os
import torch
from evaluate import evaluate,evaluate_both,getbest,evaluate_dense
from pathlib import Path
import sys
import time
from misc_utils import *
import warnings
from tqdm import tqdm
import numpy as np
from torch import autograd
from global_variables import *
from regression_init_utils import run_greedy_opt

def make_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--alg", type=str, default="gd", help="method for constructing sketch", choices=["greedy_gd", "gd", "random"]) # "best" is automatically computed
    aa("--data", type=str, default="tech") # choices=["video", "hyper", "social_network"]
    aa("--dataname", type=str, default="logo", choices=["eagle", "logo", "friends"])
    aa("--m", type=int, default=10, help="m for S")
    aa("--size", type=int, default=-1, help="dataset size")
    aa("--transpose", default=False, action='store_true', help="Run on transposed dataset; same as computing right sketch for non-transposed A")
    aa("--gpu_cpu_bs", type=int, default=100, help="Set smaller if lilogoed space on device")

    aa("--overwrite", default=False, action='store_true', help="Overwrite all trials?")
    aa("--num_exp", type=int, default=1, help="number of times to rerun the experiment (for avg'ing results)")
    aa("--device", type=str, default="cuda:0")

    aa("--iter", type=int, default=1000, help="total iterations")
    aa("--bs", type=int, default=1, help="batch size")
    aa("--lr", type=float, default=1.0, help="learning rate for GD")

    aa("--initalg", type=str, default="random", choices=["random", "kmeans", "lev", "gs", "lev_cluster", "load"])
    aa("--load_file", type=str, default="", help="if initalg=load, provide filepath for sketches")

    aa("--num_A_sample", type=int, default=1, help="number of training samples to average over in evaluation; more is slower")
    aa("--num_gs_samples", type=int, default=10, help="number of coefficients to sample and evaluate in [-2, 2], for each bin")
    aa("--num_bins_sample", type=int, help="out of m, number of bins to consider and evaluate")
    aa("--row_order", type=str, default="random", choices=["random", "forwards", "backwards", "dec_row_norm", "lev_score"], help="visit order for columns of S")
    aa("--n_early_factor", type=float, default=1.0, help="fraction of columns of S to visit")

    aa("--S_init_method", type=str, default="pm1", choices=["pm1", "gaussian","gaussian_pm1"])

    aa("--k_sparse", type=int, default=1, help="number of values in a column of S, sketching mat")
    aa("--d", type=int, default=5000, help="For synthetic type dataset only")

    aa("--save_fldr", type=str,
       help="folder to save experiment results into; if None, then general folder")  # default: None
    aa("--save_file", type=str, help="append to runtype, if not None")

    aa("--lev_ridge", dest='lev_ridge', default=False, action='store_true',
       help="use ridge regression version with lambda?")
    aa("--lev_cutoff", type=int, help="how many top k to isolate; must be <=m? if m, then not isolate, but share")
    aa("--lev_count", default=False, action="store_true", help="use counting method to compute top k over A_train?")

    aa("--bw", dest='bw', default=False, action='store_true', help="input images to black and white")
    aa("--dwnsmp", type=int, default=1, help="how much to downsample input images")
    aa("--raw", dest='raw', default=False, action='store_true', help="generate raw?")
    return parser

if __name__ == '__main__':
    task = "regression"
    important_keys = ["bs", "lr", "iter", "size"]
    sparse_algs = ["gd", "greedy_gd", "random", "col_sampling"]
    initalg_name2fn_dict = {"kmeans": init_w_kmeans, "lev": init_w_lev, "gs": init_w_gramschmidt,
                            "lev_cluster": init_w_lev_cluster, "load": init_w_load}

    parser = make_parser()
    args = parser.parse_args()
    print(args)
    defaults = parser.parse_args([])

    if args.data == "video" and args.lr == 1.0:
        # Video datasets have a different default lr
        args.lr = 10.0

    m=args.m

    save_dir = form_save_fldrs(task, args, defaults, important_keys)

    train_data, test_data = load_data_regression(args)
    A_train = train_data[0]
    B_train = train_data[1]
    A_test = test_data[0]
    B_test = test_data[1]

    n = A_train[0].shape[0]
    d = A_train[0].shape[1]
    N_train=len(A_train)
    N_test=len(A_test)
    print("Dim= ", n,d, B_train[0].shape[1])
    print("N train=", N_train, "N test=", N_test)

    best_train, best_test = get_best_error(task, save_dir, args, train_data, test_data)

    print_freq = 50  # Also evaluation frequency

    # enable adding trials to folder instead of starting from 0
    exp_num_offset = 0
    if not args.overwrite:
        subdir_list = [o for o in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, o))]
        if subdir_list: # may be first time for this exp
            subdir_list.sort()
            exp_num_offset = int(subdir_list[-1][-1]) + 1

    for exp_num in range(args.num_exp):
        exp_index = exp_num + exp_num_offset
        it_save_dir = os.path.join(save_dir,"exp_%d" % exp_index)
        it_print_freq = print_freq
        it_lr = args.lr

        if not os.path.exists(it_save_dir):
            os.makedirs(it_save_dir)

        # save args
        args_save_fpath = os.path.join(it_save_dir, "args_it_0.pkl")
        f = open(args_save_fpath, "wb")
        pickle.dump(vars(args), f)
        f.close()

        # initialize logging data structures
        test_errs = []
        train_errs = []
        fp_times = []
        bp_times = []
        timing_dict = {"total": 0, "pre_gd": 0, "gd": 0, "eval_all_train_and_test": 0}
        train_start = time.perf_counter()

        # Initialize S
        if args.alg in sparse_algs:
            # sketch_vector
            if args.alg in ["random", "gd"]:
                sketch_vector = torch.randint(m, [args.k_sparse, n]).int()
                if args.S_init_method == "pm1":
                    sketch_value = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2)
                elif args.S_init_method == "gaussian":
                    sketch_value = torch.from_numpy(np.random.normal(size=[args.k_sparse, n]).astype("float32"))
                elif args.S_init_method == "gaussian_pm1":
                    sketch_value = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2)
                    sketch_value = sketch_value + torch.from_numpy(
                        np.random.normal(size=[args.k_sparse, n]).astype("float32"))
            elif args.alg == "greedy_gd":
                # call helper which saves and return S
                dataname = args.data
                if args.data == 'video':
                    dataname = args.dataname

                exp_path = os.path.join(rltdir, task, dataname, "greedy", "m_%i" % (m), "A_%i_order_%s_values_%i_bins_%i_frac_%f" % (args.num_A_sample, args.row_order, args.num_gs_samples, m if args.num_bins_sample is None else args.num_bins_sample, args.n_early_factor), "exp_%i" % exp_index)

                if args.load_file:
                    sketch_vector, sketch_value, _ = torch.load("%s/exp_%i/saved_tensors_it_%i" % (args.load_file, exp_num, n-m-1))
                else:
                    print("Saving greedy files at %s" % exp_path)
                    print("Saving greedy files at %s" % exp_path)
                    sketch_vector, sketch_value, _ = run_greedy_opt(A_train, B_train, A_test, B_test, exp_path, m, num_A_sample=args.num_A_sample, row_order=args.row_order, num_gs_samples=args.num_gs_samples, num_bins_sample=args.num_bins_sample, n_early_factor=args.n_early_factor, device=args.device)
                    sketch_vector = torch.from_numpy(sketch_vector)
                    sketch_value = torch.from_numpy(sketch_value)

            sketch_value = sketch_value.to(args.device)
            sketch_vector.requires_grad = False
            sketch_value.requires_grad = True

        pre_gd_end = time.perf_counter()

        for bigstep in tqdm(range(args.iter)):
            if (bigstep%1000==0) and it_lr>1:
                it_lr=it_lr*0.95 # TODO
            if bigstep>200:
                it_print_freq=200


            batch_rand_ind = np.random.randint(0, high=N_train, size=args.bs)
            AM = A_train[batch_rand_ind].to(args.device)
            BM = B_train[batch_rand_ind].to(args.device)

            if args.alg in sparse_algs: # otherwise, S formed outside of loop
                S = torch.zeros(m, n).to(args.device)
                S[sketch_vector.type(torch.LongTensor).reshape(-1), torch.arange(n).repeat(args.k_sparse)] = sketch_value.reshape(-1)

            if bigstep % it_print_freq == 0 or bigstep == (args.iter - 1):
                eval_begin = time.perf_counter()
                train_err, test_err = save_iteration_regression(S, A_train, B_train, A_test, B_test, it_save_dir, bigstep, args.device)
                eval_end = time.perf_counter()
                train_errs.append(train_err)
                test_errs.append(test_err)

                if args.alg in ["random"]:
                    # don't train! after evaluating and saving, exit trial
                    break

            fp_start_time = time.perf_counter()
            SA = torch.matmul(S, AM)
            SB = torch.matmul(S, BM)
            U, Sig, V = torch.svd(SA)
            Sig_inv = torch.diag_embed(1.0/Sig)

            with torch.no_grad():
                bool_array = torch.isclose(Sig, torch.zeros_like(Sig), atol=1e-2)
                zero_inds = torch.nonzero(bool_array)
                Sig_inv[zero_inds[:, 0], zero_inds[:, 1]] = 0

            X = V.matmul(Sig_inv).matmul(U.permute(0, 2, 1)).matmul(SB)
            ans = AM.matmul(X)
            loss = torch.mean(torch.norm(ans - BM, dim=(1,2)))

            fp_times.append(time.time() - fp_start_time)
            bp_start_time = time.time()
            loss.backward()
            bp_times.append(time.time() - bp_start_time)

            with torch.no_grad():
                sketch_value -= (it_lr/args.bs)*sketch_value.grad
                sketch_value.grad.zero_()

            del SA, SB, U, Sig, V, X, ans, loss, AM, BM, S
            torch.cuda.empty_cache()

        train_end = time.perf_counter()
        np.save(os.path.join(it_save_dir, "train_errs.npy"), train_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "test_errs.npy"), test_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "fp_times.npy"), fp_times, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "bp_times.npy"), bp_times, allow_pickle=True)

        timing_dict = {"total": 0, "pre_gd": 0, "gd": 0, "eval_all_train_and_test": 0}
        timing_dict["total"] = train_end - train_start
        timing_dict["pre_gd"] = pre_gd_end - train_start
        timing_dict["gd"] = train_end - pre_gd_end
        timing_dict["eval_all_train_and_test"] = eval_end - eval_begin

        f = open(os.path.join(it_save_dir, "timing_dict.pkl"), "wb")
        pickle.dump(timing_dict, f)
        f.close()

