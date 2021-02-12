import numpy as np, torch, IPython, os
from data.videos import *
from global_variables import *
from tqdm import tqdm
# import tensorflow as tf
import time
import argparse
from data.hyperspectra import getHyper
from data.synthetic import getSynthetic
# from data.videos import getVideos
from data.tech import getTech
import math
import sys
import pickle


def compute_proj_loss(A, sketch_vector, sketch_value, m):
    n = A.size()[1]

    S = torch.zeros((m, n))
    S[sketch_vector, torch.arange(n)] = sketch_value
    SA = S.matmul(A)
    U, Sig, V = torch.svd(SA)

    proj = A.matmul(V).matmul(V.permute(0, 2, 1))
    loss = torch.mean(torch.norm(A - proj, dim=(1, 2)))
    return loss


def compute_full_loss(A, sketch_vector, sketch_value, m, k):
    n = A.size()[1]

    S = torch.zeros((m, n))
    S[sketch_vector, torch.arange(n)] = sketch_value
    SA = S.matmul(A)
    U, Sig, V = torch.svd(SA)

    AU = A.matmul(V)
    U3, Sigma3, V3 = torch.svd(AU)
    ans = U3[:, :, :k].matmul(torch.diag_embed(Sigma3[:, :k])).matmul(
        V3.permute(0, 2, 1)[:, :k]).matmul(V.permute(0, 2, 1))
    loss = torch.mean(torch.norm(ans - A, dim=(1, 2)))
    return loss


def evaluate(A_train, sketch_vector, sketch_value, m, k):
    N_train = A_train.size()[0]
    n = A_train.size()[1]
    d = A_train.size()[2]

    full_loss, proj_loss = 0, 0
    for i in range(math.ceil(N_train / 50)):
        ind_2 = min(N_train, (i + 1) * 50)
        A_section = A_train[i * 50: ind_2]

        full_loss += compute_full_loss(A_section, sketch_vector, sketch_value, m, k) * ((ind_2 - i * 50) / N_train)
        proj_loss += compute_proj_loss(A_section, sketch_vector, sketch_value, m) * ((ind_2 - i * 50) / N_train)

    return proj_loss, full_loss


def update_sketch_values(A_train, A_test, sketch_vector, old_sketch_value, m, k, active_ind, device, LR=10,
                         num_its=1000):
    """
	Assumptions:
	Proj loss (rather than full loss)

	:param A_train:
	:param A_test:
	:param sketch_vector: cpu
	:param sketch_values: cpu
	:param m:
	:param k:
	:param active_ind:
	:param LR:
	:param num_its:
	:return:
	"""
    N_train = A_train.size()[0]
    n = A_train.size()[1]
    d = A_train.size()[2]

    sketch_value = old_sketch_value.data
    sketch_value.requires_grad = True

    print_freq = 200
    bs = 5
    print("Retraining sketch_values")
    for i in range(num_its):
        if (i % print_freq) == 0:
            print("it %d" % i)

        S = torch.zeros((m, n)).to(device)
        S[sketch_vector, torch.arange(n)] = sketch_value.to(device)
        AM = A_train[np.random.randint(0, N_train, bs)].to(device)
        SA = S.matmul(AM)
        U, Sig, V = torch.svd(SA)

        proj = AM.matmul(V).matmul(V.permute(0, 2, 1))
        loss = torch.mean(torch.norm(AM - proj, dim=(1, 2)))
        loss.backward()
        with torch.no_grad():
            sketch_value[active_ind] -= (LR / bs) * sketch_value.grad[active_ind]
            sketch_value.grad.zero_()
        del S, AM, SA, U, Sig, V, proj, loss
        torch.cuda.empty_cache()

    return sketch_value.data

def fast_rank1_update_svd(U, Sig, V, a, b, device):
    """
	Batched!
	Only need to compute V'
	inputs should all be on cuda/GPU
	:param U: bsxmxm
	:param Sig: bsxmxm
	:param V: bsxdxm
	:param a: (m*num_gs_samples) x m x 1
	:param b: (m*num_gs_samples) x d x 1
	:return: V'
	"""
    m = V.size()[1]
    d = V.size()[0]

    m_tens = U[None].permute(0, 2, 1).matmul(a)
    p = a - U[None].matmul(m_tens)  # a perp U
    R_a = torch.norm(p, dim=1)
    P = p * (1.0 / R_a[:, :, None])

    n = V[None].permute(0, 2, 1).matmul(b)
    q = b - V[None].matmul(n)
    R_b = torch.norm(q, dim=1)
    Q = q * (1.0 / R_b[:, :, None])

    S_ext = torch.zeros(m + 1, m + 1).to(device)
    S_ext[:m, :m] = torch.diag(Sig)
    y = torch.cat((n, R_b[:, :, None]), dim=1)
    K = S_ext + torch.cat((m_tens, R_a[:, :, None]), dim=1).matmul(y.permute(0, 2, 1))

    bs = a.size()[0]

    V_tiled = V[None].repeat(bs, 1, 1)
    V_ext = torch.cat((V_tiled, Q), dim=2)

    u1, s1, v1 = torch.svd(K)
    V_prime = V_ext.matmul(v1)  # anyways, s1[m] is tiny

    del m_tens, p, R_a, P, n, q, R_b, Q, S_ext, y, K, V_tiled, V_ext, u1, s1, v1
    return V_prime

def fast_loss(gs_samples, AM, U0, Sig0, V0, m, n, d, k, use_proj_loss, i, device, num_bins_sample, sampled_bins=None):
    """
	:param gs_samples:
	:param AM: n x d
	:param U0: m x m
	:param Sig0: m x m
	:param V0: d x m
	:param m:
	:param n:
	:param d:
	:param num_bins_sample:
	:return:
	"""
    with torch.no_grad():
        num_gs_samples = gs_samples.size()[0]
        if num_bins_sample == 0:
            num_bins_sample = m
            sampled_bins = np.arange(m)
        if num_bins_sample and sampled_bins is None:
            sampled_bins = np.random.choice(np.arange(m), size=num_bins_sample, replace=False)
        a = torch.zeros((num_bins_sample, m))
        a[np.arange(num_bins_sample), sampled_bins] = 1.0
        a = a[:, :, None]
        a = torch.repeat_interleave(a, num_gs_samples, dim=0)
        a = a.to(device)

        b = (gs_samples[:, None].matmul(AM[i][None]))[:, :, None]
        b = b.repeat(num_bins_sample, 1, 1)
        b = b.to(device)
        V = fast_rank1_update_svd(U0, Sig0, V0, a, b, device)
        V = V[:, :, :m]
        del U0, Sig0, V0, a, b
        if use_proj_loss:
            total = num_bins_sample * num_gs_samples
            gpu_bs = 50
            proj_losses = torch.empty(total)
            for j in range(math.ceil(total / float(gpu_bs))):
                V_batch = V[j * gpu_bs: min(total, (j + 1) * gpu_bs)]
                proj = AM[None].matmul(V_batch).matmul(V_batch.permute(0, 2, 1))
                loss_tensor = torch.norm(AM[None] - proj, dim=(1, 2))
                proj_losses[j * gpu_bs: min(total, (j + 1) * gpu_bs)] = loss_tensor
                del V_batch, proj, loss_tensor
                torch.cuda.empty_cache()
            losses = proj_losses
        else:
            total = num_bins_sample * num_gs_samples
            gpu_bs = 50
            full_losses = torch.empty(total)
            for j in range(math.ceil(total / float(gpu_bs))):
                V_batch = V[j * gpu_bs: min(total, (j + 1) * gpu_bs)]
                AV = AM.matmul(V_batch)
                U3, Sigma3, V3 = torch.svd(AV)
                ans = U3[:, :, :k].matmul(torch.diag_embed(Sigma3[:, :k]).to(device)).matmul(
                    V3.permute(0, 2, 1)[:, :k]).matmul(V_batch.permute(0, 2, 1))
                loss_tensor = torch.norm(ans - AM, dim=(1, 2))
                full_losses[j * gpu_bs: min(total, (j + 1) * gpu_bs)] = loss_tensor
                del V_batch, AV, U3, Sigma3, V3, ans, loss_tensor
                torch.cuda.empty_cache()
            losses = full_losses
    torch.cuda.empty_cache()
    return losses, sampled_bins


def args_to_fldrname_gs(args, parser):
    """
	:param args: from parse_args(), a namespace
	:return: str, foldername
	"""
    ignore_keys = ["save_fldr", "save_file", "device", "data", "dataname"]
    d_args = vars(args)
    exp_fldr = args.save_file
    for key in sorted(d_args.keys()):
        if key not in ignore_keys and d_args[key] != parser.get_default(key):
            exp_fldr += "_" + str(key) + "_" + str(d_args[key])
    if not args.save_file:
        exp_fldr = exp_fldr[1:]
    exp_path = os.path.join("/your/path/here", args.data,
                            args.dataname if args.data == "video" else "", "gs", args.save_fldr, exp_fldr)
    return exp_path


def run_greedy_opt(A_train, A_test, save_path, m, k, num_A_sample=1,
                     retrain_svalues_freq=0, num_gs_samples=10, use_proj_loss=True, n_early_factor=1,
                     device="cuda:0", LR=1.0, switch_objectives=False, num_bins_sample=None,
                     row_order="random"):
    """
    Runs position optimization algorithm
	"""
    print("Device in greedy_opt: ", device)
    N_train = A_train.size()[0]
    n = A_train.size()[1]
    d = A_train.size()[2]

    # early termination option
    end_ind = math.ceil(n * n_early_factor)
    if num_bins_sample is None:
        num_bins_sample = m

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save args
    with open(os.path.join(save_path, "args.pkl"), 'wb') as handle:
        args_dict = {"num_A_sample": num_A_sample,
                     "retrain_svalues_freq": retrain_svalues_freq, "num_gs_samples": num_gs_samples, "use_proj_loss": use_proj_loss, "n_early_factor":n_early_factor,
                     "device": device, "LR":LR, "switch_objectives":switch_objectives, "num_bins_sample":num_bins_sample,
                     "row_order": row_order}
        args_dict["n"] = n
        args_dict["d"] = d
        args_dict["end_ind"] = end_ind
        pickle.dump(args_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # begin run

    # Sample 1 training matrix
    AM = (A_train[np.random.randint(0, N_train, num_A_sample)]).to(device)
    AM_np = AM.data.cpu().numpy()[0]

    print("check out row_order algs and SVD error")
    if row_order == "random":
        shuff_row_ind = np.arange(n)
        np.random.shuffle(shuff_row_ind)
    elif row_order == "forwards":
        shuff_row_ind = np.arange(n)
    elif row_order == "backwards":
        shuff_row_ind = np.arange(n)[::-1]
    elif row_order == "dec_row_norm":
        row_norms = np.linalg.norm(AM_np, axis=1)
        shuff_row_ind = np.argsort(row_norms)[::-1]
    elif row_order == "lev_score":
        U, Sig, VT = np.linalg.svd(AM_np)
        lev_scores = np.linalg.norm(U, axis=1)
        shuff_row_ind = np.argsort(lev_scores)[::-1]
    else:
        print("Invalid row order selection, exiting")
        sys.exit(0)

    sketch_vector = np.zeros(n)
    sketch_values = np.zeros(n).astype("float32")
    active_ind = shuff_row_ind[:m]
    sketch_vector[active_ind] = np.arange(m)
    sketch_values[active_ind] = np.random.normal(size=m)

    count = 0
    print_freq = 100  # Modify

    # init save data structs
    test_errs = np.empty((0, 2))
    train_errs = np.empty((0, 2))
    exp_use_proj_loss = use_proj_loss
    for i in tqdm(shuff_row_ind[m:end_ind]):
        if count > 200:
            print_freq = 500
        if count == int((end_ind - m) // 2) and switch_objectives:
            print("using full loss")
            exp_use_proj_loss = False

        with torch.no_grad():
            gs_samples = torch.linspace(-1.0, 1.0, steps=num_gs_samples).to(
                device)  # can use a diff range besides [-1.0, 1.0]
            S = torch.zeros((m, n)).to(device)
            S[sketch_vector, torch.arange(n)] = torch.from_numpy(sketch_values).to(device)
            SA = S.matmul(AM)
            t0 = time.time()
            U0, Sig0, V0 = torch.svd(SA)

            avg_proj_losses = torch.zeros(num_bins_sample * num_gs_samples)
            sampled_bins = None
            for j in range(num_A_sample):
                j_proj_loss, sampled_bins = fast_loss(gs_samples, AM[j], U0[j], Sig0[j], V0[j], m, n, d, k,
                                                      exp_use_proj_loss, i, device, num_bins_sample, sampled_bins)

                avg_proj_losses += j_proj_loss / float(num_A_sample)

            min_ind_flat = torch.argmin(avg_proj_losses)
            min_ind = [min_ind_flat // num_gs_samples, min_ind_flat % num_gs_samples]

            # update sketch vector/values
            sketch_vector[i] = torch.tensor(sampled_bins[min_ind[0]])
            sketch_values[i] = gs_samples[min_ind[1]]
            active_ind = np.concatenate((active_ind, [i]))

        if retrain_svalues_freq:
            if count % retrain_svalues_freq == 0:
                sketch_values = update_sketch_values(A_train, A_test, sketch_vector, sketch_values, m, k,
                                                     active_ind, device, LR=LR)

        # every so often: evaluate (train and test) and save errors and sketch vector/values
        if count % print_freq == 0 or count == (end_ind - m - 1):
            proj_loss, full_loss = evaluate(A_train, sketch_vector, torch.from_numpy(sketch_values), m, k)
            train_errs = np.concatenate((train_errs, np.array([[proj_loss, full_loss]])), axis=0)
            print("it %d, train errs: %f, %f" % (count, proj_loss, full_loss))
            proj_loss, full_loss = evaluate(A_test, sketch_vector, torch.from_numpy(sketch_values), m, k)
            test_errs = np.concatenate((test_errs, np.array([[proj_loss, full_loss]])), axis=0)
            print("it %d, test errs: %f, %f" % (count, proj_loss, full_loss))
            torch.save(
                [torch.from_numpy(sketch_vector), torch.from_numpy(sketch_values), torch.from_numpy(active_ind)],
                os.path.join(save_path, "saved_tensors_it_%d" % count))
            np.save(os.path.join(save_path, "train_errs.npy"), train_errs)
            np.save(os.path.join(save_path, "test_errs.npy"), test_errs)

        count += 1

    torch.cuda.empty_cache()
    return sketch_vector, sketch_values, active_ind

