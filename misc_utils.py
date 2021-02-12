import numpy as np 
import torch
import sys
import IPython
import os
import pickle
from evaluate import evaluate_to_rule_them_all, evaluate_to_rule_them_all_sparse, evaluate_to_rule_them_all_regression, evaluate_to_rule_them_all_rsketch, evaluate_to_rule_them_all_4sketch, getbest, getbest_regression
import warnings
import matplotlib.pyplot as plt
from global_variables import *
import math
import re
# import numpy_indexed as npi
from collections import Counter
# from sparsity_pattern_init_algs import *
from pathlib import Path
import shutil
from data.hyperspectra import getHyper
# from data.tech import getTech
from data.videos import getVideos
from data.videos_regression import getVideosRegression, getVidTest
from data.hyperspectra_regression import getHyperRegression
from data.social_network import getGraphs
from data.social_network_regression import getGraphsRegression
def mysvd(init_A,k):
    if k>min(init_A.size(0),init_A.size(1)):
        k=min(init_A.size(0),init_A.size(1))
    d=init_A.size(1)
    x=[torch.Tensor(d).uniform_() for i in range(k)]
    for i in range(k):
        x[i]=x[i].to(device)
        x[i].requires_grad=False
    def perStep(x,A):
        x2=A.t().mv(A.mv(x))
        x3=x2.div(torch.norm(x2))
        return x3
    U=[]
    S=[]
    V=[]
    Alist=[init_A]
    for kstep in range(k): #pick top k eigenvalues
        cur_list=[x[kstep]]   #current history
        for j in range(300):  #steps
            cur_list.append(perStep(cur_list[-1],Alist[-1]))  #works on cur_list
        V.append((cur_list[-1]/torch.norm(cur_list[-1])).view(1,cur_list[-1].size(0)))
        S.append((torch.norm(Alist[-1].mv(V[-1].view(-1)))).view(1))
        U.append((Alist[-1].mv(V[-1].view(-1))/S[-1]).view(1,Alist[-1].size(0)))
        Alist.append(Alist[-1]-torch.ger(Alist[-1].mv(cur_list[-1]), cur_list[-1]))
    return torch.cat(U,0).t(),torch.cat(S,0),torch.cat(V,0).t()


def return_data_fldr_pth(fldr_nm):
	hostname = get_hostname()
	if hostname == "your-hostname":
		data_fldr_pth = "your/path/here"
	else:
		data_fldr_pth = "your/path/here"
	return data_fldr_pth


def args_to_fldrname(task, args, defaults, important_keys):
	"""
	:param args: from parse_args(), a namespace
	:return: str, foldername
	"""
	d_args = vars(args)
	d_defaults = vars(defaults)
	important_greedy_keys = ["num_A_sample", "num_gs_samples", "num_bins_sample", "row_order", "n_early_factor"]
	fldrnm = ""
	for key in important_keys:
		if d_args[key] != d_defaults[key]:
			if fldrnm:
				fldrnm += "_"
			fldrnm += "%s_%s" % (key, str(d_args[key]))
	if task != "lra4" and args.alg == "greedy_gd":
		for key in important_greedy_keys:
			if d_args[key] != d_defaults[key]:
				if fldrnm:
					fldrnm += "_"
				fldrnm += "%s_%s" % (key, str(d_args[key]))
	if not fldrnm:
		fldrnm = "default"
	return fldrnm

def form_save_fldrs(task, args, defaults, important_keys):
	"""
	Forms save_fldrpath for experiment
	"""
	assert task in ["lra1", "lra4", "regression", "kmeans"]

	dataname = args.data
	if args.data == 'video':
		dataname = args.dataname
	if args.transpose:
		dataname += "_transpose"

	# sketch_size_other_params
	if task in ["lra1", "kmeans"]:
		sketch_size_other_params = "k_%i_m_%i" % (args.k, args.m)
	elif task == "lra4":
		sketch_size_other_params = "k_%i_m_%i_mr_%i_mt_%i_mw_%i" % (args.k, args.m, args.m_r, args.m_t, args.m_w)
	elif task == "regression":
		sketch_size_other_params = "m_%i" % (args.m)

	fldrnm = args_to_fldrname(task, args, defaults, important_keys)
	save_fldrpath = os.path.join(rltdir, task, dataname, args.alg, sketch_size_other_params, fldrnm)

	# make foldername
	if args.overwrite and os.path.exists(save_fldrpath):
		shutil.rmtree(save_fldrpath)  # Removes all the subdirectories!

	os.makedirs(save_fldrpath, exist_ok=True)
	print("Saving experiment at %s" % save_fldrpath)
	return save_fldrpath

def load_data(args):
	if args.data == 'hyper':
		A_train, A_test, n, d = getHyper(args.raw, args.size, rawdir, 100)
		train_data = [A_train]
		test_data = [A_test]
	elif args.data == 'video':
		A_train, A_test, n, d = getVideos(args.dataname, args.raw, args.size, rawdir, 100, args.bw, args.dwnsmp)
		train_data = [A_train]
		test_data = [A_test]
	elif args.data == "social_network":
		A_train, A_test, n, d = getGraphs(args.raw, rawdir)
		train_data = [A_train]
		test_data = [A_test]
	return train_data, test_data

def load_data_regression(args):
	if args.data == 'video':
		train_data, test_data, n, d_a, d_b = getVidTest(args.dataname, args.raw, args.size, rawdir, 100)
	elif args.data == 'hyper':
		train_data, test_data, n, d_a, d_b = getHyperRegression(args.raw, args.size, rawdir, 100)
	elif 'social_network' in args.data:
		if args.data != 'social_network':
			numB = int(args.data.split("_")[-1])
			train_data, test_data, n, d_a, d_b = getGraphsRegression(args.raw, rawdir, numB)
		else:
			train_data, test_data, n, d_a, d_b = getGraphsRegression(args.raw, rawdir)
	return train_data, test_data

def get_best_error(task, save_dir, args, train_data_list, test_data_list):
	# Compute and save, if doesn't exist
	N_train = len(train_data_list[0])
	N_test = len(test_data_list[0])
	if task in ["lra1", "lra4"]:
		filename = "N_%i_k_%i" % ((N_train + N_test), args.k)
	elif task == "regression":
		filename = "N_%i" % ((N_train + N_test))

	best_fldr_path = os.path.join(Path(save_dir).parents[2], "best")
	os.makedirs(best_fldr_path, exist_ok=True)

	best_file_path = os.path.join(best_fldr_path, filename)
	if not os.path.exists(best_file_path) or args.raw:
		print("computing optimal solution, saving at", best_file_path)
		if task in ["lra1", "lra4"]:
			A_train = train_data_list[0]
			A_test = test_data_list[0]
			getbest(A_train, A_test, args.k, args.data, best_file_path)
		elif task == "regression":
			A_train, B_train = train_data_list
			A_test, B_test = test_data_list
			getbest_regression(A_train, B_train, A_test, B_test, best_file_path)

	best_train, best_test = torch.load(best_file_path)
	print("Best: %f , %f" % (best_train, best_test))

	return best_train, best_test

def save_iteration_4sketch(S, R, T, W, A_train, A_test, args, save_dir, bigstep):
	torch_save_fpath = os.path.join(save_dir, "it_%d" % bigstep)

	test_err = evaluate_to_rule_them_all_4sketch(A_test, S, R, T, W, args.k)
	train_err = 0
	torch.save([[S, R, T, W], [train_err, test_err]], torch_save_fpath)

	print(train_err, test_err)
	print("Saved iteration: %d" % bigstep)
	return train_err, test_err

def save_iteration_rsketch(S, R, A_train, A_test, args, save_dir, bigstep):
	torch_save_fpath = os.path.join(save_dir, "it_%d" % bigstep)

	test_err = evaluate_to_rule_them_all_rsketch(A_test, S, R, args.k)
	train_err = 0
	torch.save([[S, R], [train_err, test_err]], torch_save_fpath)

	print(train_err, test_err)
	print("Saved iteration: %d" % bigstep)
	return train_err, test_err

def save_iteration_regression(S, A_train, B_train, A_test, B_test, save_dir, bigstep, device):
	"""
	Not implemented:
	Mixed matrix evaluation
	"""
	torch_save_fpath = os.path.join(save_dir, "it_%d" % bigstep)

	test_err = evaluate_to_rule_them_all_regression(A_test, B_test, S, device)
	train_err = evaluate_to_rule_them_all_regression(A_train, B_train, S, device)
	torch.save([[S], [train_err, test_err]], torch_save_fpath)

	print(train_err, test_err)
	print("Saved iteration: %d" % bigstep)
	return train_err, test_err

def save_iteration(S, A_train, A_test, args, save_dir, bigstep, type=None, S2=None, sparse=False):
	if sparse:
		eval_fn = evaluate_to_rule_them_all_sparse
	else:
		eval_fn = evaluate_to_rule_them_all

	warnings.warn("Save iteration does not handle 'tech' or sparse type data")
	if type is None:
		torch_save_fpath = os.path.join(save_dir, "it_%d" % bigstep)
	else:
		torch_save_fpath = os.path.join(save_dir, str(type) + "_it_%d" % bigstep)

	if S2 is None:
		test_err = eval_fn(A_test, S, args.k)
		train_err = eval_fn(A_train, S, args.k)
		torch.save([[S], [train_err, test_err]], torch_save_fpath)
	else:
		test_err = eval_fn(A_test, torch.cat([S, S2]), args.k)
		train_err = eval_fn(A_train, torch.cat([S, S2]), args.k)
		torch.save([[S, S2], [train_err, test_err]], torch_save_fpath)

	print(train_err, test_err)
	print("Saved iteration: %d" % bigstep)
	return train_err, test_err

######## KMEANS ########
def initialize_centroids(k, points):
	"""returns k centroids from the initial points"""
	centroids = points.copy()
	np.random.shuffle(centroids)
	return centroids[:k]

def closest_centroid(points, centroids):
	"""returns an array containing the index to the nearest centroid for each point"""
	distances = np.linalg.norm(points - centroids[:, np.newaxis], axis=2)
	# IPython.embed()
	return np.argmin(distances, axis=0)

def update_centroids(points, closest, centroids):
	"""returns the new centroids assigned from the points closest to them"""
	# IPython.embed()
	new_centroids = np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
	return new_centroids

def run_kmeans(data, k_means):
	centroids = initialize_centroids(k_means, data)

	dist = float("inf")
	count = 0
	while dist > 0.05 and count<20: # stopping condition
		closest = closest_centroid(data, centroids)
		new_centroids = update_centroids(data, closest, centroids)
		dist = np.linalg.norm(new_centroids - centroids)
		print(dist)
		centroids = new_centroids
		count +=1
	return centroids

def init_w_kmeans(A_train, m, rk_k):
	"""
	Note: currently only uses the first matrix in set A_train
	"""
	rand_ind = np.random.randint(low=0, high=len(A_train))
	print("sampled matrix %d" % rand_ind)
	A_train_sample = A_train[rand_ind].numpy()
	A_train_sample = (A_train_sample.T/np.linalg.norm(A_train_sample, axis=1)).T
	centroids = run_kmeans(np.copy(A_train_sample), m)
	rv = closest_centroid(np.copy(A_train_sample), np.copy(centroids))
	rv = torch.from_numpy(rv)

	return rv

def visualize_kmeans(A_train_sample, centroids):
	u, s, vt = np.linalg.svd(A_train_sample)
	proj_sample = A_train_sample@(vt[:2].T)
	proj_centroids = centroids@(vt[:2].T)

	plt.scatter(proj_sample[:, 0], proj_sample[:, 1])
	for i in range(proj_centroids.shape[0]):
		plt.plot([0, proj_centroids[i, 0]], [0, proj_centroids[i, 1]])
	plt.savefig("visualize_kmeans.jpg")

def init_w_load(load_file, exp_num, n, m):
	"""
	CAUTION: Should only be used for greedy experiments
	:param load_file:
	:param exp_num:
	:return: Expects everything to be torch tensor!
	"""
	big_lowrank_pth = "/this/path" if get_hostname() == "your-hostname" else "/other/path"

	exp_args = pickle.load(open(os.path.join(big_lowrank_pth, load_file, "args.pkl"), "rb"))
	last_itr = exp_args["end_ind"] - m -1

	full_flpth = os.path.join(big_lowrank_pth, load_file, "exp_%d" % exp_num, "saved_tensors_it_%d" % last_itr)
	if not os.path.exists(full_flpth):
		print(full_flpth, " does not exist")
		sys.exit(0)

	print("Loading pre-initialized sketch from %s" % full_flpth)
	x = torch.load(full_flpth)

	sketch_vector = x[0]
	sketch_value = x[1]
	active_ind = x[2]
	if type(sketch_vector) == np.ndarray:
		sketch_vector = torch.from_numpy(sketch_vector)

	active_ind = torch.arange(len(sketch_vector))
	return sketch_vector, sketch_value, active_ind


