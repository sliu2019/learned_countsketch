"""
Like train_rightsketch.py, but with 2 additional projection matrices
"""
import argparse
import os
import torch
from evaluate import evaluate,evaluate_both,getbest,evaluate_dense,compute_4_sketch
from pathlib import Path
import sys
import time
from misc_utils import *
import warnings
from tqdm import tqdm
import numpy as np
from global_variables import *

def make_4sketch_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--alg", type=str, default="gd", help="method for constructing sketch", choices=["greedy_gd", "gd", "random", "exact_svd", "col_sampling"]) # "best" is automatically computed
    aa("--data", type=str, default="tech", help="tech|video|hyper|generate")
    aa("--dataname", type=str, default="logo", help="transformer|logo|friends")
    aa("--m", type=int, default=10, help="m for S")
    aa("--k", type=int, default=10, help="target: rank k approximation")
    aa("--m_r", type=int, default=10, help="m_r for R")
    aa("--m_t", type=int, default=10, help="left dim of T, countsketch")
    aa("--m_w", type=int, default=10, help="right dim of W, countsketch")
    aa("--overwrite", default=False, action='store_true', help="Overwrite all trials?")
    aa("--transpose", default=False, action='store_true', help="Run on transposed dataset; same as computing right sketch for non-transposed A")


    aa("--start_exp", type=int, default=0, help="which trial to begin with?")
    aa("--iter", type=int, default=1000, help="total iterations")
    aa("--size", type=int, default=-1, help="dataset size")
    aa("--scale", type=int, default=100, help="scale")

    aa("--lr", type=float, default=1.0, help="gd learning rate")
    aa("--learn_R", dest='learn_R', default=False, action='store_true', help="is the R sketching matrix also learned?")
    aa("--learn_T", dest='learn_T', default=False, action='store_true', help="is the T sketching matrix also learned?")
    aa("--learn_W", dest='learn_W', default=False, action='store_true', help="is the W sketching matrix also learned?")

    aa("--load_S", type=str, help="Folder path for loading S")
    aa("--load_R", type=str)

    aa("--save_fldr", type=str,
       help="folder to save experiment results into; if None, then general folder")
    aa("--save_file", type=str, help="append to runtype, if not None")
    aa("--k_sparse", type=int, default=1, help="number of values in a column of S, sketching mat")
    aa("--num_exp", type=int, default=1, help="number of times to rerun the experiment (for avg'ing results)")
    aa("--bs", type=int, default=1, help="batch size")
    aa("--bw", dest='bw', default=False, action='store_true', help="input images to black and white")
    aa("--dwnsmp", type=int, default=1, help="how much to downsample input images")
    aa("--single", dest='single', default=False, action='store_true', help="generate raw?")
    aa("--dense", type=int, default=-1, help="calculate dense?")
    # aa("--lr_S", type=float, default=1, help="learning rate scale?")
    aa("--raw", dest='raw', default=False, action='store_true', help="generate raw?")
    aa("--bestonly", dest='bestonly', default=False, action='store_true', help="only compute best?")
    aa("--device", type=str, default="cuda:0")
    return parser

if __name__ == '__main__':
    task = "lra4"
    important_keys = ["bs", "lr", "iter", "size"]

    parser = make_4sketch_parser()
    args = parser.parse_args()
    print(args)
    defaults = parser.parse_args([])

    m=args.m
    k=args.k
    alg = args.alg
    assert(args.m_t > args.m_r, "Choose m_t >> m_r")
    assert(args.m_w > args.m, "Choose m_w >> m_s")

    if args.data == "video" and args.lr == 1.0:
        # Video datasets have a different default lr
        args.lr = 10.0

    save_dir = form_save_fldrs(task, args, defaults, important_keys)

    train_data, test_data = load_data(args)
    A_train = train_data[0]
    A_test = test_data[0]

    n = A_train[0].shape[0]
    d = A_train[0].shape[1]
    N_train=len(A_train)
    N_test=len(A_test)
    print("Dim= ", n,d)
    print("N train=", N_train, "N test=", N_test)

    best_train, best_test = get_best_error(task, save_dir, args, train_data, test_data)

    start=time.time()

    print_freq = 50

    m_r = args.m_r
    m_t = args.m_t
    m_w = args.m_w
    # print(m_r)
    small_const = 1e-06
    for exp_num in range(args.start_exp, args.num_exp):

        it_save_dir = os.path.join(save_dir,"exp_%d" % exp_num)
        print("Saving trial at %s" % it_save_dir)
        it_print_freq = print_freq
        it_lr = args.lr

        if not os.path.exists(it_save_dir):
            os.makedirs(it_save_dir)

        # save args
        args_save_fpath = os.path.join(it_save_dir, "args_it_0.pkl")
        f = open(args_save_fpath, "wb")
        pickle.dump(vars(args), f)
        f.close()

        test_errs = []
        train_errs = []
        fp_times = []
        bp_times = []

        if args.load_S:
            if args.alg in ["exact_svd", "col_sampling"]:
                it = 0
            else:
                it = 999
            load_path = os.path.join(args.load_S, "exp_%i" %  exp_num, "it_%i" % it)
            saved_tensors = torch.load(load_path)
            S = saved_tensors[0][0].to(args.device)
            if args.alg not in ["exact_svd", "col_sampling"]:
                S = saved_tensors[0][0].detach()
                nnz = torch.nonzero(S)
                nnz_np = nnz.cpu().detach().numpy()
                nnz_args = np.argsort(nnz_np[:, 1])
                nnz_sorted = nnz_np[nnz_args]
                sketch_vector = nnz_sorted[:, 0]
                sketch_vector = torch.from_numpy(sketch_vector)
                sketch_value = S[nnz_sorted[:, 0], nnz_sorted[:,1]].to(args.device)
                sketch_vector.requires_grad = False
                sketch_value.requires_grad = True
        else:
            sketch_vector = torch.randint(m, [args.k_sparse, n]).int()
            sketch_value = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).to(args.device)
            sketch_vector.requires_grad = False
            sketch_value.requires_grad = True

        if args.load_R:
            if args.alg in ["exact_svd", "col_sampling"]:
                it = 0
            else:
                it = 999
            load_path = os.path.join(args.load_R, "exp_%i" %  exp_num, "it_%i" % it)
            saved_tensors = torch.load(load_path, map_location=args.device)
            R = saved_tensors[0][0].to(args.device).permute(1, 0)
            if args.alg not in ["exact_svd", "col_sampling"]:
                R = saved_tensors[0][0].detach()
                nnz = torch.nonzero(R)
                nnz_np = nnz.cpu().detach().numpy()
                nnz_args = np.argsort(nnz_np[:, 1])
                nnz_sorted = nnz_np[nnz_args]
                R_sketch_vector = nnz_sorted[:, 0]
                R_sketch_vector = torch.from_numpy(R_sketch_vector)
                R_sketch_value = R[nnz_sorted[:, 0], nnz_sorted[:,1]].to(args.device)
                R_sketch_vector.requires_grad = False
                R_sketch_value.requires_grad = args.learn_R

        else:
            R_sketch_vector = torch.randint(m_r, [d]).int()
            R_sketch_value = ((torch.randint(2, [d]).float() - 0.5) * 2).to(args.device)
            R_sketch_vector.requires_grad = False
            R_sketch_value.requires_grad = args.learn_R

        T_sketch_vector = torch.randint(args.m_t, [n]).int()
        T_sketch_vector.requires_grad = False
        T_sketch_value = ((torch.randint(2, [n]).float() - 0.5) * 2).to(args.device)
        T_sketch_value.requires_grad = args.learn_T

        W_sketch_vector = torch.randint(args.m_w, [d]).int()
        W_sketch_vector.requires_grad = False
        W_sketch_value = ((torch.randint(2, [d]).float() - 0.5) * 2).to(args.device)
        W_sketch_value.requires_grad = args.learn_W

        avg_detailed_timing = np.zeros((5))
        for bigstep in tqdm(range(args.iter)):
            if (bigstep%1000==0) and it_lr>1:
                it_lr=it_lr*0.95
            if bigstep>200:
                it_print_freq=200
            A = A_train[np.random.randint(0, high=N_train, size=args.bs)]

            AM = A.to(args.device)
            Ad=d
            An=n

            if args.alg not in ["exact_svd", "col_sampling"]:
                S = torch.zeros(m, n).to(args.device)
                S[sketch_vector.type(torch.LongTensor).reshape(-1), torch.arange(n).repeat(args.k_sparse)] = sketch_value.reshape(-1)

                R = torch.zeros(d, m_r).to(args.device)
                R[torch.arange(d), R_sketch_vector.type(torch.LongTensor).reshape(-1)] = R_sketch_value.reshape(-1)

            T = torch.zeros(m_t, n).to(args.device)
            T[T_sketch_vector.type(torch.LongTensor), torch.arange(n)] = T_sketch_value

            W = torch.zeros(d, m_w).to(args.device)
            W[torch.arange(d), W_sketch_vector.type(torch.LongTensor)] = W_sketch_value
            if bigstep % it_print_freq == 0 or bigstep == (args.iter - 1):
                train_err, test_err = save_iteration_4sketch(S, R, T, W, A_train, A_test, args, it_save_dir, bigstep)
                train_errs.append(train_err)
                test_errs.append(test_err)

                if alg in ["random", "exact_svd", "col_sampling"]:
                    break

            fp_start_time = time.time()  # forward pass timing

            time_dict = []
            temp_start = time.time()
            AR = AM.matmul(R)
            SA = S.matmul(AM)
            TAR = T.matmul(AR)
            TAW = T.matmul(AM).matmul(W)
            SAW = SA.matmul(W)

            C = TAR
            D = SAW
            G = TAW

            # Removes batch entries with rank-deficient C or D
            U_c, Sig_c, V_c = torch.svd(C)
            U_d, Sig_d, V_d = torch.svd(D.permute(0, 2, 1))

            Sig_c_cpu = Sig_c.cpu()
            bool_array = torch.isclose(Sig_c_cpu, torch.zeros_like(Sig_c_cpu), atol=1e-4)
            zero_inds = torch.nonzero(bool_array)
            unique_c, counts = np.unique(zero_inds[:, 0], return_counts=True)

            Sig_d_cpu = Sig_d.cpu()
            bool_array = torch.isclose(Sig_d_cpu, torch.zeros_like(Sig_d_cpu), atol=1e-4)
            zero_inds = torch.nonzero(bool_array)
            unique_d, counts = np.unique(zero_inds[:, 0], return_counts=True)

            good_ind = np.arange(args.bs)
            good_ind = np.delete(good_ind, unique_c)
            good_ind = np.delete(good_ind, unique_d)

            # Modify batch
            C = C[good_ind]
            D = D[good_ind]
            G = G[good_ind]
            AM = AM[good_ind]
            AR = AR[good_ind]
            SA = SA[good_ind]

            # Start old algorithm
            time_dict.append(time.time() - temp_start)
            temp_start = time.time()
            # Full QR, not truncated
            U_c, R_c = torch.qr(C, some=True)
            U_d, R_d = torch.qr(D.permute(0, 2, 1), some=True)
            time_dict.append(time.time() - temp_start)

            temp_start = time.time()
            G_proj = (U_c.permute(0, 2, 1)).matmul(G).matmul(U_d)
            U1, Sig1, V1 = torch.svd(G_proj)
            X_prime_L = U1[:, :, :k].matmul(torch.diag_embed(Sig1[:, :k]))
            X_prime_R = V1.permute(0, 2, 1)[:, :k]
            time_dict.append(time.time() - temp_start)

            temp_start = time.time()
            rk_c = R_c.shape[1]
            T_c = R_c[:, :, :rk_c]
            try:
                T_c_inv = torch.inverse(T_c)
            except:
                print("Couldn't invert T_c")
                IPython.embed()

            X_L = T_c_inv.matmul(X_prime_L)

            # check that c x_l = u_c x_l'!
            rk_d = R_d.shape[1]
            T_d = R_d[:, :, :rk_d]  # + torch.eye(rk_d).to(args.device)*small_const
            try:
                T_d_inv = torch.inverse(T_d)
            except:
                print("couldn't invert T_d")
                IPython.embed()
            X_R = X_prime_R.matmul(T_d_inv.permute(0, 2, 1))
            time_dict.append(time.time() - temp_start)

            temp_start = time.time()
            X = X_L.matmul(X_R)

            ans = AR.matmul(X).matmul(SA)

            loss = torch.mean(torch.norm(ans - AM, dim=(1, 2)))
            fp_time = time.time() - fp_start_time
            fp_times.append(fp_time)
            bp_start_time = time.time()  # backwards pass timing
            loss.backward()
            bp_time = time.time() - bp_start_time
            bp_times.append(bp_time)
            time_dict.append(time.time() - temp_start)
            keys = ["sparse matrix mult", "x2 qr", "svd", "recover (x2 inverse)", "loss"]

            avg_detailed_timing += np.array(time_dict)
            if bigstep == 10:
                avg_detailed_timing = avg_detailed_timing / 11.0
                avg_fp_time = np.mean(fp_times)
                percen = (avg_detailed_timing * 100) / avg_fp_time
                for i, key in enumerate(keys):
                    pass

            with torch.no_grad():
                sketch_value -= (it_lr / args.bs) * sketch_value.grad
                sketch_value.grad.zero_()
                if args.learn_R:
                    R_sketch_value -= (it_lr / args.bs) * R_sketch_value.grad
                    R_sketch_value.grad.zero_()
                if args.learn_T:
                    T_sketch_value -= (it_lr / args.bs) * T_sketch_value.grad
                    T_sketch_value.grad.zero_()
                if args.learn_W:
                    W_sketch_value -= (it_lr / args.bs) * W_sketch_value.grad
                    W_sketch_value.grad.zero_()

            del C, D, G, U_c, Sig_c, V_c, U_d, Sig_d, V_d, loss, S, R, T, W, AR, SA, TAR, TAW, SAW, A, AM

        np.save(os.path.join(it_save_dir, "train_errs.npy"), train_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "test_errs.npy"), test_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "fp_times.npy"), fp_times, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "bp_times.npy"), bp_times, allow_pickle=True)