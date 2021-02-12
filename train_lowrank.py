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
from greedy_opt import run_greedy_opt

def make_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--alg", type=str, default="gd", help="method for constructing sketch", choices=["greedy_gd", "gd", "random", "exact_svd", "col_sampling", "load"]) # "best" is automatically computed
    aa("--data", type=str, default="tech", choices=["tech", "video", "hyper", "social_network"])
    aa("--dataname", type=str, default="logo", choices=["eagle", "logo", "friends"])
    aa("--m", type=int, default=10, help="m for S")
    aa("--k", type=int, default=10, help="target: rank k approximation")
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

    # If alg = greedy_gd
    aa("--num_A_sample", type=int, default=1, help="number of training samples to average over in evaluation; more is slower")
    aa("--num_gs_samples", type=int, default=10, help="number of coefficients to sample and evaluate in [-2, 2], for each bin")
    aa("--num_bins_sample", type=int, help="out of m, number of bins to consider and evaluate")
    aa("--row_order", type=str, default="random", choices=["random", "forwards", "backwards", "dec_row_norm", "lev_score"], help="visit order for columns of S")
    aa("--n_early_factor", type=float, default=1.0, help="fraction of columns of S to visit")

    aa("--S_init_method", type=str, default="pm1", choices=["pm1", "gaussian","gaussian_pm1"])

    aa("--n_sample_rows", type=int, default=-1, help="Train with n_sample_rows rows")
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
    task = "lra1"
    important_keys = ["bs", "lr", "iter", "size"]
    sparse_algs = ["gd", "greedy_gd", "random", "col_sampling", "load"]
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
    k=args.k

    # IPython.embed()
    save_dir = form_save_fldrs(task, args, defaults, important_keys)

    train_data, test_data = load_data(args)
    A_train = train_data[0]
    A_test = test_data[0]

    if args.transpose:
        A_train = A_train.permute(0, 2, 1)
        A_test = A_test.permute(0, 2, 1)
        # IPython.embed()

    n = A_train[0].shape[0]
    d = A_train[0].shape[1]
    N_train=len(A_train)
    N_test=len(A_test)
    print("Dim= ", n,d)
    print("N train=", N_train, "N test=", N_test)

    best_train, best_test = get_best_error(task, save_dir, args, train_data, test_data)

    print_freq = 50  # Also evaluation frequency

    if args.n_sample_rows > 0 and args.n_sample_rows <= n:
        #numpy.random.choice(a, size=None, replace=True, p=None)¶
        sample_rows_inds = np.random.choice(np.arange(n), size=args.n_sample_rows, replace=False)

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

        print("Saving at %s" % it_save_dir)
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
            elif args.alg == "load":
                load_path = os.path.join(args.load_file, "saved_tensors_it_1011")
                saved_tensors = torch.load(load_path)
                sketch_vector = saved_tensors[0].to(args.device)
                sketch_value = saved_tensors[1].to(args.device)
                # IPython.embed()
            elif args.alg == "greedy_gd":
                # call helper which saves and return S
                dataname = args.data
                if args.data == 'video':
                    dataname = args.dataname
                if args.transpose:
                    dataname += "_transpose"

                # IPython.embed()
                exp_path = os.path.join(rltdir, task, dataname, "greedy", "k_%i_m_%i" % (k, m), "A_%i_order_%s_values_%i_bins_%i_frac_%f" % (args.num_A_sample, args.row_order, args.num_gs_samples, m if args.num_bins_sample is None else args.num_bins_sample, args.n_early_factor), "exp_%i" % exp_index)
                # print(args.device)
                # print("running greedy opt")
                print("Saving greedy files at %s" % exp_path)
                print("Saving greedy files at %s" % exp_path)
                sketch_vector, sketch_value, _ = run_greedy_opt(A_train, A_test, exp_path, m, k, num_A_sample=args.num_A_sample, row_order=args.row_order, num_gs_samples=args.num_gs_samples, num_bins_sample=args.num_bins_sample, n_early_factor=args.n_early_factor, device=args.device)
                sketch_vector = torch.from_numpy(sketch_vector)
                sketch_value = torch.from_numpy(sketch_value)
            elif args.alg == "col_sampling":
                ind = np.random.randint(0, N_train)
                A_samp = A_train[ind]
                samp_probs = np.linalg.norm(A_samp, axis=1)
                samp_probs = samp_probs / np.sum(samp_probs)
                sel_col_ind = np.random.choice(np.arange(n), size=m, replace=False, p=samp_probs)
                sel_col_ind = np.sort(sel_col_ind)
                sketch_vector = np.zeros(n)
                sketch_vector[sel_col_ind] = np.arange(m)

                sketch_vector = torch.from_numpy(sketch_vector).int()

                sketch_value = np.zeros(n)
                sketch_value[sel_col_ind] = 1.0 / (
                            samp_probs[sel_col_ind])
                sketch_value = torch.from_numpy(sketch_value).float()

            sketch_value = sketch_value.to(args.device)
            sketch_vector.requires_grad = False
            sketch_value.requires_grad = True
        else:
            if args.alg == "exact_svd":
                ind = np.random.randint(0, N_train)
                A_rand = A_train[ind]
                U, Sig, V = torch.svd(A_rand)
                S = U[:, :m].T
                S = S.to(args.device)


        pre_gd_end = time.perf_counter()

        for bigstep in tqdm(range(args.iter)):
            # with autograd.detect_anomaly():
            if (bigstep%1000==0) and it_lr>1:
                it_lr=it_lr*0.3
            if bigstep>200:
                it_print_freq=200

            A = A_train[np.random.randint(0, high=N_train, size=args.bs)]
            AM = A.to(args.device)
            Ad=d
            An=n

            if args.alg in sparse_algs: # otherwise, S formed outside of loop
                S = torch.zeros(m, n).to(args.device)
                if args.n_sample_rows >= m and args.n_sample_rows <= n:
                    zero_ind = sketch_vector.type(torch.LongTensor).reshape(-1)[sample_rows_inds]
                    zero_ind[:m] = torch.arange(m)
                    S[zero_ind, sample_rows_inds] = sketch_value.reshape(-1)[sample_rows_inds]
                else:
                    S[sketch_vector.type(torch.LongTensor).reshape(-1), torch.arange(n).repeat(args.k_sparse)] = sketch_value.reshape(-1)

            if bigstep % it_print_freq == 0 or bigstep == (args.iter - 1):
                eval_begin = time.perf_counter()
                train_err, test_err = save_iteration(S, A_train, A_test, args, it_save_dir, bigstep)
                eval_end = time.perf_counter()
                train_errs.append(train_err)
                test_errs.append(test_err)

                if args.alg in ["random", "exact_svd", "col_sampling"]:
                    # don't train! after evaluating and saving, exit trial
                    break

            fp_start_time = time.time()
            SA = torch.matmul(S, AM)
            U2, Sigma2, V2 = torch.svd(SA) # returns compact SVD
            AU = AM.matmul(V2)
            U3, sigma3, V3 = torch.svd(AU)
            Sigma3 = torch.diag_embed(sigma3[:, :k]).to(args.device)
            ans = U3[:, :, :k].matmul(Sigma3).matmul(
                V3.permute(0, 2, 1)[:, :k]).matmul(V2.permute(0, 2, 1))
            loss = torch.mean(torch.norm(ans - AM, dim=(1, 2)))

            fp_times.append(time.time() - fp_start_time)
            bp_start_time = time.time()
            loss.backward()
            bp_times.append(time.time() - bp_start_time)

            with torch.no_grad():
                sketch_value -= (it_lr/args.bs)*sketch_value.grad
                sketch_value.grad.zero_()

            del SA, U2, Sigma2, V2, AU, U3, Sigma3, V3, ans, loss, AM
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

