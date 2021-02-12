import numpy as np
import os
import torch
import h5py
import IPython
def processRaw(N,rawdir,scale):
    # A_train=[]
    # A_test=[]
    fldr_pth = os.path.join(rawdir, "HS-SOD", "hyperspectral")

    mat_files = [os.path.join(fldr_pth, m) for m in os.listdir(fldr_pth)]
    # print("N %i" % N)
    # IPython.embed()
    count = 0
    for mat_file in mat_files:
        f = h5py.File(mat_file, 'r')
        AList=f['hypercube'][:]
        for j in range(AList.shape[0]):
            print(count)

            As = torch.from_numpy(AList[j]).view(AList[j].shape[0], -1).float()
            # print(As.size())
            U, S, V = As.svd()
            div = abs(S[0].item())
            if div < 1e-10:
                div = 1
                print("Catch!")
            div /= scale

            As = (As/div).float()
            if count == 0:
                A_train, A_test = torch.empty((0, As.size()[0], As.size()[1])), torch.empty(
                    (0, As.size()[0], As.size()[1]))
            if np.random.random() < 0.8:
                A_train = torch.cat((A_train, As[None]), dim=0)
            else:
                A_test = torch.cat((A_test, As[None]), dim=0)
            count += 1

            if count == N:
                print("Done")
                torch.save(A_train, os.path.join(rawdir, "hyper_train_" + str(N) + "_" + str(scale) + ".dat"))
                torch.save(A_test, os.path.join(rawdir, "hyper_test_" + str(N) + "_" + str(scale) + ".dat"))
                return

def getHyper(raw,N,rawdir,scale):
    if N<0:
        N=5
    if raw:
        processRaw(N,rawdir,scale)
    A_train = torch.load(os.path.join(rawdir, "hyper_train_"+str(N)+"_"+str(scale)+".dat"))
    A_test = torch.load(os.path.join(rawdir, "hyper_test_"+str(N)+"_"+str(scale)+".dat"))
    return A_train,A_test, A_train.size()[1], A_train.size()[2]
