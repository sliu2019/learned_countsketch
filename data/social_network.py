import numpy as np
import csv
import os
import IPython
import pickle
import torch

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

def processRaw2(rawdir):
    scale = 100
    A_i_list = pickle.load(open(os.path.join(rawdir, "gplus_raw.pkl"), "rb"))
    sizes = [A.shape[0] for A in A_i_list]
    print(np.sort(sizes))
    min_size = 1052

    # IPython.embed()
    n = 0
    A_all = torch.empty((0, min_size, min_size))
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
            A_all = torch.cat((A_all, A_torch[None, :, :]), dim=0)

    # IPython.embed()

    rand_ind = np.random.permutation(A_all.shape[0])
    num_train = int(A_all.shape[0]*0.8)
    train_ind = rand_ind[:num_train]
    test_ind = rand_ind[num_train:]
    A_train = A_all[train_ind]
    A_test = A_all[test_ind]

    torch.save(A_train, os.path.join(rawdir, "social_network", "gplus_train_%i.dat" % num_train))
    torch.save(A_test, os.path.join(rawdir, "social_network", "gplus_test_%i.dat" % A_test.shape[0]))

    # IPython.embed()
def getGraphs(raw, rawdir):
    if raw:
        processRaw1(rawdir)
        processRaw2(rawdir)

    A_train=torch.load(os.path.join(rawdir, "social_network", "gplus_train_147.dat"))
    A_test=torch.load(os.path.join(rawdir, "social_network", "gplus_test_37.dat"))
    n=A_train[0].size(0)
    d=A_train[0].size(1)

    return A_train,A_test,n,d

if __name__ == "__main__":
    rawdir = "/your/path/here"
    # processRaw1(rawdir)
    # processRaw2(rawdir)