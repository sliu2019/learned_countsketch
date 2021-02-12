import torch
import numpy as np
import sys
from global_variables import *
import IPython
import math

def bestPossible(eval_list,k,data):
    totLoss = torch.tensor(0.0)
    for A in eval_list:
        print(".",end="")
        sys.stdout.flush()
        if data=='tech':
            AM=A['M'].to(device)
        else:
            AM=A.to(device)
        U, S, V = AM.svd()
        ans = U[:, :k].mm(torch.diag(S[:k]).to(device)).mm(V.t()[:k])
        # totLoss += torch.norm(ans - AM) ** 2
        totLoss += torch.norm(ans - AM)
    return totLoss

def evaluate(sparse, eval_list,sketch_vector, sketch_value,m,k,n,d):  # evaluate the test/train performance
    totLoss = 0
    count = 0


    for A in eval_list:
        if sparse:
            AM=A['M'].to(device)
            SA = torch.Tensor(m, A['d']).fill_(0).to(device)
            for i in range(A['n']):  # A has this many rows, not mapped yet
                actR = A['Map'][i]  # Actual row in the matrix
                mapR = sketch_vector[actR]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[actR]  # remember: times the weight
        else:
            AM=A.to(device)
            SA = torch.Tensor(m, d).fill_(0).to(device)
            for i in range(n):  # A has this many rows, not mapped yet
                mapR = sketch_vector[i]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[i]  # remember: times the weight

        U2, Sigma2, V2 = SA.svd()
        AU = AM.mm(V2)
        U3, Sigma3, V3 = AU.svd()
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        totLoss += (torch.norm(ans - AM)).item()
        count += 1
        if (count % 10 == 0):
            print(count, end=",")
            sys.stdout.flush()
    return totLoss

def evaluate_dense(sparse, eval_list,sketch, m,k):  # evaluate the test/train performance
    totLoss = 0
    count = 0

    for A in eval_list:
        if sparse:
            AM=A['M'].to(device)
            SA = torch.Tensor(m, A['d']).fill_(0).to(device)
            for i in range(A['n']):  # A has this many rows, not mapped yet
                actR = A['Map'][i]  # Actual row in the matrix
                SA+=torch.ger(sketch[:,actR], AM[i])
        else:
            AM=A.to(device)
            SA=torch.mm(sketch, AM)

        U2, Sigma2, V2 = SA.svd()
        AU = AM.mm(V2)
        U3, Sigma3, V3 = AU.svd()
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        totLoss += (torch.norm(ans - AM)).item()
        count += 1
        if (count % 10 == 0):
            print(count, end=",")
            sys.stdout.flush()
    return totLoss


def evaluate_both(eval_list,sketch_vector, sketch_value,m,k,n,d):  # evaluate the test/train performance
    totLoss = 0
    count = 0

    for A in eval_list:
        if sparse:
            AM=A['M'].to(device)
            SA = torch.Tensor(m, A['d']).fill_(0).to(device)
            for i in range(A['n']):  # A has this many rows, not mapped yet
                actR = A['Map'][i]  # Actual row in the matrix
                mapR = sketch_vector[actR]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[actR]  # remember: times the weight
        else:
            AM=A.to(device)
            SA = torch.Tensor(m, d).fill_(0).to(device)
            for i in range(n):  # A has this many rows, not mapped yet
                mapR = sketch_vector[i]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[i]  # remember: times the weight

        U2, Sigma2, V2 = SA.svd()
        AU = AM.mm(V2)
        U3, Sigma3, V3 = AU.svd()
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        totLoss += (torch.norm(ans - AM)).item()
        count += 1
        if (count % 10 == 0):
            print(count, end=",")
            sys.stdout.flush()
    return totLoss

def evaluate_extra_dense(eval_list,sketch, sketch2, k):
    totLoss = 0
    count = 0
    for A in eval_list:
        AM=A.to(device)
        SA=torch.cat([torch.mm(sketch,AM),torch.mm(sketch2,AM)])

        U2, Sigma2, V2 = SA.svd()
        AU = AM.mm(V2)
        U3, Sigma3, V3 = AU.svd()
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        totLoss += (torch.norm(ans - AM)).item()
        count += 1
    return totLoss

def compute_4_sketch(U_c, Sig_c, V_c, U_d, Sig_d, V_d, G, k):
    """
    Assumes (U_c, Sig_c, V_c, U_d, Sig_d, V_d) are properly truncated; returns solution X
    :param U_c:
    :param Sig_c:
    :param V_c:
    :param U_d:
    :param Sig_d:
    :param V_d:
    :param G:
    :param k:
    :param device:
    :return:
    """
    G_proj = (U_c.permute(0, 2, 1)).matmul(G).matmul(U_d)

    U1, Sig1, V1 = torch.svd(G_proj)
    X_prime_L = U1[:, :, :k].matmul(torch.diag_embed(Sig1[:, :k]))
    X_prime_R = V1.permute(0, 2, 1)[:, :k]

    sig_inv_c = torch.div(1.0, Sig_c)
    sig_inv_d = torch.div(1.0, Sig_d)

    X_L = (V_c).matmul(torch.diag_embed(sig_inv_c)).matmul(X_prime_L)
    X_R = X_prime_R.matmul(torch.diag_embed(sig_inv_d)).matmul(V_d.permute(0, 2, 1))

    X = X_L.matmul(X_R)
    return X

def evaluate_to_rule_them_all_4sketch(A_set, S, R, T, W, k, device="cpu"):
    S = S.to(device)
    R = R.to(device)
    T = T.to(device)
    W = W.to(device)

    n = A_set.shape[0]
    bs = 100
    loss = 0
    for i in range(math.ceil(n / float(bs))):
        AM = A_set[i*bs:min((i+1)*bs, n)].to(device)
        it_bs = min((i+1)*bs, n) - i*bs
        AR = AM.matmul(R)
        SA = S.matmul(AM)
        TAR = T.matmul(AR)
        TAW = T.matmul(AM).matmul(W)
        SAW = SA.matmul(W)


        m_r = R.shape[1]
        m = S.shape[0]
        C = TAR
        D = SAW
        G = TAW

        # Full QR, not truncated
        U_c, Sig_c, V_c = torch.svd(C)
        U_d, Sig_d, V_d = torch.svd(D.permute(0, 2, 1))

        # do fancy indexing to split batch as needed
        table = np.zeros((it_bs, 2))

        # find zeros in sig_c
        bool_array = torch.isclose(Sig_c, torch.zeros_like(Sig_c), atol=1e-2)
        zero_inds = torch.nonzero(bool_array)

        unique, counts = np.unique(zero_inds[:, 0], return_counts=True)
        table[unique, 0] = counts

        # sig_d
        bool_array = torch.isclose(Sig_d, torch.zeros_like(Sig_d), atol=1e-2)
        zero_inds = torch.nonzero(bool_array)

        unique, counts = np.unique(zero_inds[:, 0], return_counts=True)
        table[unique, 1] = counts

        # sort into groups
        unique = np.unique(table, axis=0).astype("int")

        for u in unique:
            batch_indices = np.where((table == u).all(axis=1))[0]

            U_c_batch = U_c[batch_indices]
            Sig_c_batch = Sig_c[batch_indices]
            V_c_batch = V_c[batch_indices]
            
            if u[0] > 0:
                U_c_batch = U_c_batch[:, :, :-u[0]]
                Sig_c_batch = Sig_c_batch[:, :-u[0]]
                V_c_batch = V_c_batch[:, :, :-u[0]]

            U_d_batch = U_d[batch_indices]
            Sig_d_batch = Sig_d[batch_indices]
            V_d_batch = V_d[batch_indices]

            if u[1] > 0:
                U_d_batch = U_d_batch[:, :, :-u[1]]
                Sig_d_batch = Sig_d_batch[:, :-u[1]]
                V_d_batch = V_d_batch[:, :, :-u[1]]

            G_batch= G[batch_indices]
            X = compute_4_sketch(U_c_batch, Sig_c_batch, V_c_batch, U_d_batch, Sig_d_batch, V_d_batch, G_batch, k)
            
            ans = AR[batch_indices].matmul(X).matmul(SA[batch_indices])
            it_loss = torch.sum(torch.norm(ans - AM[batch_indices], dim=(1, 2))) / n

            loss += it_loss.item()

    return loss

def evaluate_to_rule_them_all_rsketch(A_set, S, R, k):
    S = S.cpu()
    R = R.cpu()

    n = A_set.shape[0]
    bs = 100
    loss = 0
    for i in range(math.ceil(n / float(bs))):
        AM = A_set[i*bs:min((i+1)*bs, n)]
        SA = torch.matmul(S, AM)
        AR = torch.matmul(AM, R)
        SAR = S.matmul(AR)
        U1, Sig1, V1 = torch.svd(SAR)
        U2, Sig2, V2 = torch.svd(AR.matmul(V1))
        Y = U2[:, :, :k].matmul(torch.diag_embed(Sig2[:, :k])).matmul(V2.permute(0, 2, 1)[:, :k]).matmul(V1.permute(0,2,1))
        SAR_pinv = V1.matmul(torch.diag_embed(1.0 / Sig1)).matmul(U1.permute(0, 2, 1))
        ans = Y.matmul(SAR_pinv).matmul(SA)
        it_loss = torch.sum(torch.norm(ans - AM, dim=(1, 2)))/n
        loss += it_loss.item()
    return loss

def evaluate_to_rule_them_all_regression(A_set, B_set, S, device):
    """
    BATCHED, but also iterative (i.e. for data=hyper, eval list may be ~3000)
    :param A: list of matrices (3D tensor)
    :param sketch: S or [S, S2] concatenated; assumed matrices
    :param k: low-rank k
    :return: K-rk approx cost, averaged over matrices in eval_list
    """
    n = A_set.size()[0]
    bs = 25
    loss = 0
    S = S.detach()
    for i in range(math.ceil(n/float(bs))):
        AM = A_set[i*bs:min(n, (i+1)*bs)].to(device)
        BM = B_set[i*bs:min(n, (i+1)*bs)].to(device)

        SA = torch.matmul(S, AM)
        SB = torch.matmul(S, BM)
        U, Sig, V = torch.svd(SA)

        Sig_np = Sig.cpu().numpy()
        nontriv = np.logical_not(np.isclose(Sig_np, np.zeros_like(Sig_np), atol=1e-02))
        Sig_inv_np = np.divide(1.0, Sig_np, out=np.zeros_like(Sig_np), where=nontriv)
        Sig_inv = torch.diag_embed(torch.from_numpy(Sig_inv_np).to(device))
        X = V.matmul(Sig_inv).matmul(U.permute(0, 2, 1)).matmul(SB)
        ans = AM.matmul(X)
        it_loss = torch.sum(torch.norm(ans - BM, dim=(1, 2)))/n
        loss += it_loss.item()
        del AM, BM, SA, SB, U, Sig, V, X, ans, it_loss
        torch.cuda.empty_cache()
    return loss

def evaluate_to_rule_them_all(eval_list, sketch, k, device="cpu"):
    """
    BATCHED, but also iterative (i.e. for data=hyper, eval list may be ~3000)
    :param A: list of matrices (3D tensor)
    :param sketch: S or [S, S2] concatenated; assumed matrices
    :param k: low-rank k
    :return: K-rk approx cost, averaged over matrices in eval_list
    """
    n = eval_list.size()[0]
    cpu_bs = 100
    loss = 0
    sketch = sketch.to(device)
    for i in range(math.ceil(n/float(cpu_bs))):
        AM = eval_list[i*cpu_bs:min(n, (i+1)*cpu_bs)].to(device)
        SA = torch.matmul(sketch, AM)
        U2, Sigma2, V2 = torch.svd(SA)
        AU = AM.matmul(V2)
        U3, Sigma3, V3 = torch.svd(AU)
        ans = U3[:, :, :k].matmul(torch.diag_embed(Sigma3[:, :k])).matmul(V3.permute(0, 2, 1)[:, :k]).matmul(
            V2.permute(0, 2, 1))
        it_loss = torch.sum(torch.norm(ans - AM, dim=(1, 2)))/n
        loss += it_loss.item()
    return loss

def evaluate_to_rule_them_all_sparse(eval_list, sketch, k):
    """
    Not batched; uses GPU within iteration
    :param eval_list:
    :param sketch:
    :param k:
    :param device:
    :return:
    """
    device = sketch.device.type + (":%d" % sketch.device.index if sketch.device.index else "")
    loss = 0
    n = len(eval_list)
    for A in eval_list:
        AM = A['M'][None].to(device)
        AMap = A['Map']

        ind = torch.tensor(AMap).type(torch.LongTensor).to(device)
        S = torch.index_select(sketch, dim=1, index=ind)
        SA = S.matmul(AM)
        U2, Sigma2, V2 = torch.svd(SA)
        AU = AM.matmul(V2)
        U3, Sigma3, V3 = torch.svd(AU)
        ans = U3[:, :, :k].matmul(torch.diag_embed(Sigma3[:, :k])).matmul(V3.permute(0, 2, 1)[:, :k]).matmul(
            V2.permute(0, 2, 1))
        loss += torch.norm(ans - AM, dim=(1, 2)).item()/n
    return loss

def evaluate_extra(sparse, eval_list,sketch_vector, sketch_value,sketch_vector2, sketch_value2,m,mextra,k,n,d):
    totLoss = 0
    count = 0
    for A in eval_list:
        if sparse:
            AM=A['M'].to(device)
            SA = torch.Tensor(m+mextra, A['d']).fill_(0).to(device)
            for i in range(A['n']):  # A has this many rows, not mapped yet
                actR = A['Map'][i]  # Actual row in the matrix
                mapR = sketch_vector[actR]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[actR]  # remember: times the weight

                mapR=sketch_vector2[actR]+m
                SA[mapR]+= AM[i] * sketch_value2[actR]
        else:
            AM=A.to(device)
            SA = torch.Tensor(m+mextra, d).fill_(0).to(device)
            for i in range(n):  # A has this many rows, not mapped yet
                mapR = sketch_vector[i]  # row is mapped to this row in the sketch
                SA[mapR] += AM[i] * sketch_value[i]  # remember: times the weight

                mapR=sketch_vector2[i]+m
                SA[mapR] += AM[i] * sketch_value2[i]  # remember: times the weight

        U2, Sigma2, V2 = SA.svd()
        AU = AM.mm(V2)
        U3, Sigma3, V3 = AU.svd()
        ans = U3[:, :k].mm(torch.diag(Sigma3[:k]).to(device)).mm(V3.t()[:k]).mm(V2.t())
        totLoss += (torch.norm(ans - AM)).item()
        count += 1
    return totLoss

def getAvgDim(A_list):
    nL=[]
    dL=[]
    for A in A_list:
        nL.append(A['n'])
        dL.append(A['d'])
    print('Avg height',np.average(nL),'Avg width',np.average(dL))

def getbest(A_train, A_test,k,data,best_file):
    best_train = bestPossible(A_train, k, data).tolist()
    best_test = bestPossible(A_test,k,data).tolist()
    best_errs = [best_train/len(A_train) if len(A_train) != 0 else 0, best_test/len(A_test) if len(A_test) !=0 else 0]
    print(best_errs)
    torch.save(best_errs, best_file)
    return best_train, best_test

def bestPossible_regression(A_set, B_set):
    n = A_set.size()[0]
    bs = 100
    loss = 0
    for i in range(math.ceil(n/float(bs))):
        AM = A_set[i*bs:min(n, (i+1)*bs)]
        BM = B_set[i*bs:min(n, (i+1)*bs)]

        U, Sig, V = torch.svd(AM)
        nontriv = np.logical_not(np.isclose(Sig, np.zeros_like(Sig), atol=1e-02))
        Sig_inv_np = np.divide(1.0, Sig, out=np.zeros_like(Sig), where=nontriv)
        Sig_inv = torch.diag_embed(torch.from_numpy(Sig_inv_np))
        X = V.matmul(Sig_inv).matmul(U.permute(0, 2, 1)).matmul(BM)
        ans = AM.matmul(X)
        it_loss = torch.sum(torch.norm(ans - BM, dim=(1, 2)))/n
        loss += it_loss.item()
    return loss

def getbest_regression(A_train, B_train, A_test, B_test, best_file):
    best_train_err = bestPossible_regression(A_train, B_train)
    best_test_err = bestPossible_regression(A_test, B_test)

    torch.save([best_train_err, best_test_err], best_file)
    return best_train_err, best_test_err