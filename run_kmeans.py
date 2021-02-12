import torch, numpy
import time
from datetime import datetime
import IPython
from create_baseline_sketch import *
import sys, os, pickle
from sklearn.cluster import KMeans, MiniBatchKMeans
from global_variables import *
from misc_utils import *

# Note: careful about device override from global_variables

def m_fn(x):
    if x == 3:
        return 20
    elif x == 6:
        return 40
    elif x == 10:
        return 70
    else:
        print("error in m_fn")
        sys.exit(0)


def CS_sketch(A_test, S, n_clusters, exp_save_fldr_pth, i):
    m = S.shape[0]

    sketched = S[None].matmul(A_test)
    U, Sig, V = torch.svd(sketched)
    A_test_sketched = A_test.matmul(V)

    # save sketches, S and G
    torch.save(S, os.path.join(exp_save_fldr_pth, "S_trial_%i" % i))

    return A_test_sketched

def evaluate_cost(n_clusters, A_train, A_test, dataset_spec, sketch_method, save_fldr_pth, num_rand_trials=5, S_list=None):
    """
    If you're doing a timing run, just feed in A_test with size 1. Fix n_clusters, dataset.

    :param n_clusters: number of k-means clusters
    :param A_test: n_test x n x d, torch
    :param sketch_method: one of "learned_CS" (CountSketch), "random_CS", "exact_SVD", ...
    :param n_trials: number of sketch instantiations to average over.
    If deterministic, n_trials = 1.
    :param S_list: if sketch_method contains "learned", use these. Torch tensors
    :return:
    """
    # Setting up saving
    if len(dataset_spec) == 1:
        exp_save_fldr_pth = os.path.join(save_fldr_pth, "k_%i_dataset_%s_sketch_method_%s" % (n_clusters, dataset_spec[0], sketch_method))
    else:
        exp_save_fldr_pth = os.path.join(save_fldr_pth, "k_%i_dataset_%s_%s_sketch_method_%s" % (n_clusters, dataset_spec[0], dataset_spec[1], sketch_method))

    print("Saving in %s" % exp_save_fldr_pth)
    if not os.path.exists(exp_save_fldr_pth):
        os.mkdir(exp_save_fldr_pth)

    # creating vars to be used later
    n_test = A_test.shape[0]
    n = A_test.shape[1]
    d = A_test.shape[2]
    n_train = A_train.shape[0]

    err_list = []

    if "learned" in sketch_method:
        n_trials = len(S_list)
    else:
        n_trials = num_rand_trials

    kmeans_est = KMeans(n_clusters=n_clusters, random_state=random_seed_number)

    for i in range(n_trials):
        # Sketch A_test: produce A_test_sketched (n x anything)
        if sketch_method == "learned_CS":
            S = S_list[i]
            A_test_sketched = CS_sketch(A_test, S, n_clusters, save_fldr_pth, i)
        elif sketch_method == "random_CS":
            m = m_fn(n_clusters)
            sketch_vector = np.random.randint(0, m, size=n)
            sketch_value = ((torch.randint(2, [n]).float() - 0.5) * 2)
            S = torch.zeros((m, n))
            S[sketch_vector, np.arange(n)] = sketch_value
            A_test_sketched = CS_sketch(A_test, S, n_clusters, save_fldr_pth, i)
        elif sketch_method == "learned_CS_oblivious":
            S = S_list[i]
            A_test_sketched = A_test.matmul(S.permute(1, 0)[None])
        elif sketch_method == "random_CS_oblivious":
            m = m_fn(n_clusters)
            sketch_vector = np.random.randint(0, m, size=d)
            sketch_value = ((torch.randint(2, [d]).float() - 0.5) * 2)
            S = torch.zeros((m, d))
            S[sketch_vector, np.arange(d)] = sketch_value
            A_test_sketched = A_test.matmul(S.permute(1, 0)[None])

            torch.save(S, os.path.join(exp_save_fldr_pth, "S_trial_%i" % i))
        elif sketch_method == "exact_SVD":
            ind = np.random.randint(0, n_train)
            m = m_fn(n_clusters)
            U, S, V = torch.svd(A_train[ind])
            V_m = V[:,:m]
            A_test_sketched = A_test.matmul(V_m[None])

            torch.save(V_m, os.path.join(exp_save_fldr_pth, "V_m_trial_%i" % i))
        elif sketch_method == "col_sampling":
            ind = np.random.randint(0, n_train)
            A_samp = A_train[ind]
            m = m_fn(n_clusters)
            samp_probs = np.linalg.norm(A_samp, axis=0)
            samp_probs = samp_probs/np.sum(samp_probs)
            sel_col_ind = np.random.choice(np.arange(d), size=m, replace=False, p=samp_probs)
            sel_col_ind = np.sort(sel_col_ind)

            R = np.zeros((d, m))
            R[sel_col_ind, np.arange(m)] = 1.0/(samp_probs[sel_col_ind]*d)
            R = torch.from_numpy(R).float()

            A_test_sketched = A_test.matmul(R[None])
            torch.save(R, os.path.join(exp_save_fldr_pth, "R_trial_%i" % i))

        A_test_sketched_np = A_test_sketched.data.cpu().numpy()
        A_test_np = A_test.data.cpu().numpy()
        sum_err_over_test = 0
        for j in range(n_test):
            # Run kmeans++ to get clustering
            labels = kmeans_est.fit_predict(A_test_sketched_np[j])

            # Evaluate cluster cost on A_test
            A = A_test_np[j]
            X_c = np.zeros((n_clusters, n))  # k x n

            class_ind, counts = np.unique(labels, return_counts=True)  # RV is sorted
            norm_consts = 1.0 / np.sqrt(counts)
            X_c[labels, np.arange(n)] = norm_consts[labels]

            obj_val = np.mean(np.linalg.norm(A - X_c.T @ X_c @ A, axis=1))
            sum_err_over_test += obj_val

        avg_err = sum_err_over_test/n_test
        err_list.append(avg_err)
        np.save(os.path.join(exp_save_fldr_pth, "errs.npy"), err_list)

        print(avg_err)

    return np.mean(err_list), np.std(err_list)

def get_dataset(dataset_spec):
    raw = False
    size = 500
    bw = False
    dwnsmp = 1

    rawdir = "your/path/here" if get_hostname() == "your-hostname" else "your/path/here"
    dataset = dataset_spec[0]
    if dataset=='tech':
        A_train,A_test,n,d=getTech(raw,rawdir,100)
    elif dataset=='hyper':
        A_train,A_test,n,d=getHyper(raw,size,rawdir,100)
    elif dataset=='video':
        dataname = dataset_spec[1]
        A_train,A_test,n,d=getVideos(dataname,raw,size,rawdir,100, bw, dwnsmp)

    return A_train, A_test, n, d

def load_S(S_path, n):
    S_list = []

    exp_fldr = os.path.join("your/path/here", S_path)
    args = pickle.load(open(os.path.join(exp_fldr, "args_it_0.pkl"), "rb"))
    num_exp = args["num_exp"]
    iter = args["iter"]
    m = args["m"]
    for exp_num in range(num_exp):
        saved_tensors_fpath = os.path.join(exp_fldr, "exp_%d" % exp_num, "it_%d" % (iter - 1))
        saved_tensors = torch.load(saved_tensors_fpath)
        S = saved_tensors[0][0].data.cpu()
        S_list.append(S)
    return S_list

if __name__ == "__main__":
    """
    Sketch methods: exact_SVD, col_sampling, random_CS_oblivious, random_CS, learned_CS_oblivious, learned_CS
    """
    kmeans_folder_pth = "/your/path/here" if get_hostname() == "your-hostname" else "/your/path/here"

    num_rand_trials = 1

    save_folder_nm = "debug_table_1"

    n_cluster_list = [6]
    dataset_spec_list = [["video", "logo"]]
    sketch_method_list = ["learned_CS_oblivious"]


    S_fpth_dict = {"learned_CS_oblivious": {("hyper", ): {3: "rlt/hyper/greedy_ablation/train_direct_grad_half_update_exp_2_S_init_method_pm1_bs_1_data_hyper_dataname_logo_device_cuda:1_initalg_load_iter_1000_k_20_k_sparse_1_m_20_n_sample_rows_-1_num_exp_1_random_False_size_500"}, ("video", "logo"): {6: "rlt/video/logo/debug_kmeans_table_1/train_direct_grad_S_init_method_pm1_bs_1_data_video_dataname_logo_initalg_random_iter_2000_k_40_k_sparse_1_m_40_n_sample_rows_-1_num_exp_1_random_False_size_500"}}}

    # Save params
    save_fldr_pth = os.path.join(kmeans_folder_pth, save_folder_nm)
    if not os.path.isdir(save_fldr_pth):
        os.mkdir(save_fldr_pth)

    args = {"n_cluster_list": n_cluster_list, "dataset_spec_list": dataset_spec_list, "sketch_method_list": sketch_method_list, "S_fpth_dict": S_fpth_dict, "num_rand_trials": num_rand_trials}
    with open(os.path.join(save_fldr_pth, 'args.pkl'), 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # create table
    table_pth = os.path.join(save_fldr_pth, "output_data.npy")
    table = np.empty((len(n_cluster_list), len(dataset_spec_list), len(sketch_method_list)), dtype="object")

    for j, dataset_spec in enumerate(dataset_spec_list):
        # Load data
        A_train, A_test, n, d = get_dataset(dataset_spec)

        A_train = A_train.permute(0, 2, 1)
        A_test = A_test.permute(0, 2, 1)
        for i, n_cluster in enumerate(n_cluster_list):
            for k, sketch_method in enumerate(sketch_method_list):
                # Load S_list if need be
                S_list = None
                if sketch_method in S_fpth_dict:
                    S_fpth = S_fpth_dict[sketch_method][tuple(dataset_spec)][n_cluster]
                    S_list = load_S(S_fpth, n)

                # call evaluate_cost
                result = evaluate_cost(n_cluster, A_train, A_test, dataset_spec, sketch_method, save_fldr_pth, S_list=S_list, num_rand_trials=num_rand_trials)

                # store in table
                table[i, j, k] = result

                # save table
                np.save(table_pth, table, allow_pickle=True)
