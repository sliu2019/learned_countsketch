import os

import cv2
import numpy as np
import torch
from PIL import Image
import IPython
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
    # fname = rawdir + "raw/videos/" + videoName + "_" + train_or_test + "_"+ str(N) + "_" + str(scale)
    fname = os.path.join(rawdir, "videos", videoName + "_regression_" + train_or_test + "_"+ str(N) + "_" + str(scale))
    if bw:
        fname += "_bw"
    if dwnsmp != 1:
        fname += ("_dwnsmp%d" % dwnsmp)
    return fname + ".dat"

def convertTOImage(fname,total=1000):
    big_regression_pth = "your/path/here"
    vidcap = cv2.VideoCapture(os.path.join(big_regression_pth, 'raw/videos', fname+'.mp4'))
    if fname == "eagle":
        for _ in range(960):
            print("discard")
            _ , _ = vidcap.read()

    success,image = vidcap.read()
    count = 0
    path=os.path.join(big_regression_pth, 'raw/videos', fname)
    if not os.path.exists(path):
        os.makedirs(path)

    while success:
        image = np.swapaxes(image, 0, 1)
        cv2.imwrite(path+"/frame%d.jpg" % count, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        print(count)
        if count>total:
            break
    print("Success!")

def computeImage(total=1000):
    convertTOImage('eagle',total)
    # convertTOImage('friends',total)
    # convertTOImage('logo',total)

# computeImage(200)

def processIm(fpath, scale):
    # input: Image.open() output
    # output: ready to concatenate into data
    image = Image.open(fpath)
    image = np.array(image)
    image = image.reshape((image.shape[0] * 3, -1))
    # to torch
    cur = torch.from_numpy(image).float()
    U, S, V = cur.svd()
    div = abs(S[0].item())
    if div < 1:
        div = 1
        print("Catch!")
        return None
    div /= scale
    cur = cur/div
    return cur

        
def processRaw(fname,N,rawdir,scale, bw=False, dwnsmp=1):
    perm=torch.randperm(N)
    for i in range(N):
        # bw, downsample are unused params
        print(i)
        # if i%100==0:
        #     print(i)
        # print(os.path.join(rawdir, "videos", fname, "frame" + str(perm[i].item()+ 1000) + ".jpg"))
        # print(os.path.join(rawdir, "videos", fname, "frame" + str(perm[i].item() + 1005) + ".jpg"))
        AM = processIm(os.path.join(rawdir, "videos", fname, "frame" + str(perm[i].item() + 1000) + ".jpg"), scale)
        BM = processIm(os.path.join(rawdir, "videos", fname, "frame" + str(perm[i].item() + 1005) + ".jpg"), scale)

        if i == 0:
            A_train, A_test = torch.empty((0, AM.shape[0], AM.shape[1])), torch.empty((0, AM.shape[0], AM.shape[1]))
            B_train, B_test = torch.empty((0, BM.shape[0], BM.shape[1])), torch.empty((0, BM.shape[0], BM.shape[1]))
        if i < 0.8*N:
            # IPython.embed()
            A_train = torch.cat((A_train, AM[None].type(torch.float32)), dim=0)
            B_train = torch.cat((B_train, BM[None].type(torch.float32)), dim=0)
        else:
            A_test = torch.cat((A_test, AM[None].type(torch.float32)), dim=0)
            B_test = torch.cat((B_test, BM[None].type(torch.float32)), dim=0)

    torch.save([A_train, B_train], get_fname(fname, rawdir, N, scale, bw, dwnsmp, "train"))
    torch.save([A_test, B_test], get_fname(fname, rawdir, N, scale, bw, dwnsmp, "test"))

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
    #     for AM in A_train:
    #         U, S, V = AM.svd()
    #         x[0].append([U, S, V])
    #     for AM in A_test:
    #         U, S, V = AM.svd()
    #         x[1].append([U, S, V])
    #     torch.save(x, fname)
    #     return x[0], x[1]

def getVideosRegression(videoName,raw, N,rawdir,scale, bw=False, dwnsmp=1):
    if not videoName in ['logo','eagle','friends']:
        print("Wrong video name!")
        assert(False)
    if N<0:
        N=500
    if raw:
        processRaw(videoName,N,rawdir,scale, bw=bw, dwnsmp=dwnsmp)

    AB_train =torch.load(get_fname(videoName, rawdir, N, scale, bw, dwnsmp, "train"))
    AB_test =torch.load(get_fname(videoName, rawdir, N, scale, bw, dwnsmp, "test"))

    n = AB_train[0][0].size()[0]
    d_a = AB_train[0][0].size()[1]
    d_b = AB_train[1][0].size()[1]
    # 327 14 1
    return AB_train, AB_test, n, d_a, d_b


def processRaw_VidTest(videoName, N, rawdir):
    train_fpth = os.path.join(rawdir, "%s_regression_train_N_%i.dat" % (videoName, N))
    test_fpth =  os.path.join(rawdir, "%s_regression_test_N_%i.dat" % (videoName, N))
    print("Saving processed datasets at %s, %s" % (train_fpth, test_fpth))

    perm = torch.randperm(N)
    for i in range(N):
        # bw, downsample are unused params
        print(i)
        # if i%100==0:
        #     print(i)

        im = processIm(os.path.join(rawdir, "videos", videoName, "frame" + str(perm[i].item() + 1000) + ".jpg"), 100)
        AM = im[:, :-1]
        BM = im[:, [-1]]
        # print(im.shape, AM.shape, BM.shape)
        # BM = processIm(os.path.join(rawdir, "raw", "videos", fname, "frame" + str(perm[i].item() + 5) + ".jpg"), scale)

        if i == 0:
            A_train, A_test = torch.empty((0, AM.shape[0], AM.shape[1])), torch.empty((0, AM.shape[0], AM.shape[1]))
            B_train, B_test = torch.empty((0, BM.shape[0], BM.shape[1])), torch.empty((0, BM.shape[0], BM.shape[1]))
        if i < 0.8 * N:
            # IPython.embed()
            A_train = torch.cat((A_train, AM[None].type(torch.float32)), dim=0)
            B_train = torch.cat((B_train, BM[None].type(torch.float32)), dim=0)
        else:
            A_test = torch.cat((A_test, AM[None].type(torch.float32)), dim=0)
            B_test = torch.cat((B_test, BM[None].type(torch.float32)), dim=0)

    torch.save([A_train, B_train], train_fpth)
    torch.save([A_test, B_test],test_fpth)

def getVidTest(videoName, raw, N, rawdir, scale, bw = False, dwnsmp = 1):
    if N<0:
        N=500
    if raw:
        processRaw_VidTest(videoName, N,rawdir)

    AB_train =torch.load(os.path.join(rawdir, "%s_regression_train_N_%i.dat" % (videoName, N)))
    AB_test =torch.load(os.path.join(rawdir, "%s_regression_test_N_%i.dat" % (videoName, N)))

    n = AB_train[0][0].size()[0]
    d_a = AB_train[0][0].size()[1]
    d_b = AB_train[1][0].size()[1]
    # 327 14 1
    return AB_train, AB_test, n, d_a, d_b