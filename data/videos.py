import os

import cv2
import numpy as np
import torch
from PIL import Image
import IPython
# from global_variables import *
# from misc_utils import *

def get_fname(videoName, rawdir, N, scale, bw, dwnsmp, train_or_test):
    """
    :param videoName:
    :param rawdir:
    :param N:
    :param scale:
    :param bw:
    :param dwnsmp:
    :param type: train or test
    :return:
    """
    fname = os.path.join(rawdir, "videos", videoName + "_" + train_or_test + "_"+ str(N) + "_" + str(scale))
    if bw:
        fname += "_bw"
    if dwnsmp != 1:
        fname += ("_dwnsmp%d" % dwnsmp)
    return fname + ".dat"

def convertTOImage(fname,total=1000):
    rawdir = "your/path/here"

    vid_fl_pth = os.path.join(rawdir, "videos", "%s.mp4" % fname)
    vidcap = cv2.VideoCapture(vid_fl_pth)
    success,image = vidcap.read()
    count = 0
    # path="../big-lowrank/raw/videos/"+fname
    save_fldr_pth = os.path.join(rawdir, "videos", fname)
    if not os.path.exists(save_fldr_pth):
        os.makedirs(save_fldr_pth)
    while success:
        image = np.swapaxes(image, 0, 1)
        save_fl_pth = os.path.join(save_fldr_pth, "frame%d.jpg" % count)
        cv2.imwrite(save_fl_pth, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        print(count)
        if count>total:
            break
    print("Success!")

def computeImage(total=1000):
    # convertTOImage('eagle',total)
    convertTOImage('friends',total)
    convertTOImage('logo',total)

# computeImage(1510)
def processRaw(fname,N,rawdir,scale, bw=False, dwnsmp=1):
    perm=torch.randperm(N)
    for i in range(N):
        print(i)
        if i%100==0:
            print(i)
        image = Image.open(os.path.join(rawdir, "videos", fname, "frame"+str(perm[i].item()+1000)+".jpg"))

        if bw:
            image = np.array(image.convert('1'))
            # cur = torch.from_numpy(bw_np_im).float()
        else:
            image = np.array(image)
            image = image.reshape((image.shape[0]*3, -1))
        # downsample
        image = image[::dwnsmp, ::dwnsmp]

        # to torch
        cur=torch.from_numpy(image).float()
        U, S, V = cur.svd()
        div=abs(S[0].item())
        if div<1:
            div=1
            print("Catch!")
            continue
        div/=scale
        if i == 0:
            A_train, A_test = torch.empty((0, cur.size()[0], cur.size()[1])), torch.empty((0, cur.size()[0], cur.size()[1]))
        if np.random.random()<0.8:
            A_train = torch.cat((A_train, (cur/div)[None, :, :]), dim=0)
        else:
            A_test = torch.cat((A_test, (cur/div)[None, :, :]), dim=0)
        # del cur, U, S, V
        # torch.cuda.empty_cache()
    # IPython.embed()
    torch.save(A_train,get_fname(fname, rawdir, N, scale, bw, dwnsmp, "train"))
    torch.save(A_test,get_fname(fname, rawdir, N, scale, bw, dwnsmp, "test"))

def getSVD(videoName,raw, N,rawdir,scale, bw=False, dwnsmp=1):
    """
    Computes and saves SVDs for every matrix in a pytorch .dat file
    :param videoName:
    :param raw:
    :param N:
    :param rawdir:
    :param scale:
    :param bw:
    :param dwnsmp:
    :return:
    """
    raise(NotImplementedError) # refactor for get_fname
    # fname = get_fname(videoName, rawdir, N, scale, bw, dwnsmp)[:-4] + "_SVD.dat"
    # if os.path.exists(fname):
    #     A_train_SVD, A_test_SVD = torch.load(fname)
    #     return A_train_SVD, A_test_SVD
    # else:
    #     A_train, A_test = torch.load(get_fname(videoName, rawdir, N, scale, bw, dwnsmp))
    #     x = [[], []]
    #     for A in A_train:
    #         U, S, V = A.svd()
    #         x[0].append([U, S, V])
    #     for A in A_test:
    #         U, S, V = A.svd()
    #         x[1].append([U, S, V])
    #     torch.save(x, fname)
    #     return x[0], x[1]

def getVideos(videoName,raw, N,rawdir,scale, bw=False, dwnsmp=1):
    if not videoName in ['logo','eagle','friends']:
        print("Wrong video name!")
        assert(False)
    if N<0:
        N=200

    if raw:
        processRaw(videoName,N,rawdir,scale, bw=bw, dwnsmp=dwnsmp)

    A_train=torch.load(get_fname(videoName, rawdir, N, scale, bw, dwnsmp, "train"))
    A_test=torch.load(get_fname(videoName, rawdir, N, scale, bw, dwnsmp, "test"))
    # IPython.embed()
    n=A_train[0].size(0)
    d=A_train[0].size(1)

    return A_train,A_test,n,d
