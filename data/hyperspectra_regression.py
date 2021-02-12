import numpy as np
import os
import torch
import h5py
import IPython

def processRaw(N,rawdir,scale):
    # A_train=[]
    # A_test=[]
    train_fpth = os.path.join(rawdir, "hyper_regression_train_" + str(N) + "_" + str(scale) + ".dat")
    test_fpth = os.path.join(rawdir, "hyper_regression_test_" + str(N) + "_" + str(scale) + ".dat")
    print("Saving processed datasets at %s, %s" % (train_fpth, test_fpth))

    fldr_pth = os.path.join(rawdir, "HS-SOD", "hyperspectral")

    mat_files = [os.path.join(fldr_pth, m) for m in os.listdir(fldr_pth)]
    count = 0
    for mat_file in mat_files:
        f = h5py.File(mat_file, 'r')
        AList=f['hypercube'][:]
        for j in range(AList.shape[0]):
            print(count)

            im = torch.from_numpy(AList[j]).view(AList[j].shape[0], -1).float()
            # print(As.size())
            U, S, V = im.svd()
            div = abs(S[0].item())
            if div < 1e-10:
                div = 1
                print("Catch!")
            div /= scale

            im = (im/div).float()
            # if count == 0:
            #     A_train, A_test = torch.empty((0, As.size()[0], As.size()[1])), torch.empty(
            #         (0, As.size()[0], As.size()[1]))
            # if np.random.random() < 0.8:
            #     A_train = torch.cat((A_train, As[None]), dim=0)
            # else:
            #     A_test = torch.cat((A_test, As[None]), dim=0)

            AM = im[:, :-1]
            BM = im[:, [-1]]
            # print(AM.shape, BM.shape)

            if count == 0:
                A_train, A_test = torch.empty((0, AM.shape[0], AM.shape[1])), torch.empty((0, AM.shape[0], AM.shape[1]))
                B_train, B_test = torch.empty((0, BM.shape[0], BM.shape[1])), torch.empty((0, BM.shape[0], BM.shape[1]))
            if np.random.random() < 0.8:
                # IPython.embed()
                A_train = torch.cat((A_train, AM[None].type(torch.float32)), dim=0)
                B_train = torch.cat((B_train, BM[None].type(torch.float32)), dim=0)
            else:
                # print("adding to test set")
                A_test = torch.cat((A_test, AM[None].type(torch.float32)), dim=0)
                B_test = torch.cat((B_test, BM[None].type(torch.float32)), dim=0)

            count += 1

            if count == N:
                print("Done")
                torch.save([A_train, B_train], train_fpth)
                torch.save([A_test, B_test], test_fpth)
                return

def getHyperRegression(raw,N,rawdir,scale):
    if N<0:
        N=5
    if raw:
        processRaw(N,rawdir,scale)

    AB_train= torch.load(os.path.join(rawdir, "hyper_regression_train_"+str(N)+"_"+str(scale)+".dat"))
    AB_test = torch.load(os.path.join(rawdir, "hyper_regression_test_"+str(N)+"_"+str(scale)+".dat"))

    n = AB_train[0][0].size()[0]
    d_a = AB_train[0][0].size()[1]
    d_b = AB_train[1][0].size()[1]
    # 327 14 1
    return AB_train, AB_test, n, d_a, d_b
