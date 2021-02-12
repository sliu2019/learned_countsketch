import numpy as np
import os
import torch
import h5py
import IPython
import pickle, csv

def processRaw1(rawdir):
    # load all the .edges files from gplus folder
    data_fldr = os.path.join(rawdir, "gplus")
    files = os.listdir(data_fldr)
    files = [x for x in files if x[-5:] == "edges"]

    # print("Number of A_i: %i" % (len(files)))
    # 132 ego networks in this dataset

    A_i_list = []
    raw_file_path = os.path.join(rawdir, "gplus_raw.pkl")
    A_i_sizes = []
    # for each, read line by line
    # IPython.embed()
    for file in files:
        filepath = os.path.join(data_fldr, file)

        # A_i = np.zeros((0, 0))
        node_2_index_dict = {}
        nodes = []
        # IPython.embed()
        n_i = 0
        with open(filepath) as filehandle:
            for row in csv.reader(filehandle,delilogoer="\t"):
                row_list = str.split(row[0])
                v1 = int(row_list[0])
                v2 = int(row_list[1])

                if v1 not in nodes:
                    nodes.append(v1)
                if v2 not in nodes:
                    nodes.append(v2)
            print(len(nodes))
            nodes = np.sort(nodes)
            for i, node in enumerate(nodes):
                node_2_index_dict[node] = i
            A_i = np.zeros((len(nodes), len(nodes)))

            filehandle.seek(0)
            for row in csv.reader(filehandle,delilogoer="\t"):
                row_list = str.split(row[0])
                v1 = int(row_list[0])
                v2 = int(row_list[1])
                A_i[node_2_index_dict[v1], node_2_index_dict[v2]] = 1

                n_i += 1

        print(n_i)
        # print(A_i.shape[0])
        A_i_list.append(A_i)
        A_i_sizes.append(A_i.shape[0])
        # IPython.embed()
        # save by pickle
        pickle.dump(A_i_list, open(raw_file_path, "wb"))

    print(A_i_sizes)
    print(min(A_i_sizes))
    # IPython.embed()

def processRaw2(rawdir, num_B=210):
    A_i_list = pickle.load(open(os.path.join(rawdir, "gplus_raw.pkl"), "rb"))
    sizes = [A.shape[0] for A in A_i_list]
    min_size = 1052

    # IPython.embed()
    n = 0
    # percent_B = 0.2
    # num_B = int(0.2*min_size)
    num_A = min_size - num_B

    print("Num A cols: %i, num B cols: %i" % (num_A, num_B))

    A_all = torch.empty((0, min_size, num_A))
    B_all = torch.empty((0, min_size, num_B))
    for A_i in A_i_list:
        for i in range(A_i.shape[0]//min_size):
            print(n)
            n += 1
            A_torch = torch.from_numpy(A_i[i*min_size:(i+1)*min_size, i*min_size:(i+1)*min_size]).float()
            # U, S, V = A_torch.svd()
            # div=abs(S[0].item())
            # if div<1:
            #     div=1
            #     print("Catch!")
            #     continue
            # div/=scale
            # A_torch = A_torch/div
            A_all = torch.cat((A_all, A_torch[None, :, :num_A]), dim=0)
            B_all = torch.cat((B_all, A_torch[None, :, num_A:]), dim=0)

    # IPython.embed()

    rand_ind = np.random.permutation(A_all.shape[0])
    num_train = int(A_all.shape[0]*0.8)
    train_ind = rand_ind[:num_train]
    test_ind = rand_ind[num_train:]
    A_train = A_all[train_ind]
    B_train = B_all[train_ind]
    A_test = A_all[test_ind]
    B_test = B_all[test_ind]

    torch.save([A_train, B_train], os.path.join(rawdir, "social_network", "gplus_train_%i_numB_%i_regression.dat" % (num_train, num_B)))
    torch.save([A_test, B_test], os.path.join(rawdir, "social_network", "gplus_test_%i_numB_%i_regression.dat" % (A_test.shape[0], num_B)))


def getGraphsRegression(raw,rawdir, num_B=210):
    if raw:
        # TODO: uncomment for polished code v
        # processRaw1(rawdir)
        processRaw2(rawdir, num_B)

    if num_B != 210:
        train_save_fpth = os.path.join(rawdir, "social_network", "gplus_train_147_numB_%i_regression.dat" % num_B)
        test_save_fpth = os.path.join(rawdir, "social_network", "gplus_test_37_numB_%i_regression.dat" % num_B)
    else:
        train_save_fpth = os.path.join(rawdir, "social_network", "gplus_train_147_regression.dat")
        test_save_fpth = os.path.join(rawdir, "social_network", "gplus_test_37_regression.dat")

    print("Loading from %s" % train_save_fpth)
    AB_train= torch.load(train_save_fpth)
    AB_test = torch.load(test_save_fpth)

    n = AB_train[0][0].size()[0]
    d_a = AB_train[0][0].size()[1]
    d_b = AB_train[1][0].size()[1]
    # 327 14 1
    return AB_train, AB_test, n, d_a, d_b


if __name__ == "__main__":
    pass
    # rawdir = "/your/path/here"
    # processRaw2(rawdir)
    # rv = getGraphsRegression(False, rawdir)
    # IPython.embed()