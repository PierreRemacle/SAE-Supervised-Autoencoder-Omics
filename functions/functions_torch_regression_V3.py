# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:23:43 2024

@author: Nolwenn Peyratout
"""

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, scale as scale
import time
import numpy as np
import pandas as pd
import matplotlib as mpl

import torch
from torch import nn
import torch.nn.functional as F
import math

from tqdm import tqdm
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as sc
from sklearn import metrics

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.multiprocessing import Pool
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

import functions.functions_network_pytorch as fnp
import functions.functions_DeepSurv as fds  # DeepSurv
import torchtuples as tt

try:
    import captum
    import shap
except ImportError:
    print(
        "Use '!pip install captum' to install captum; '!pip install shap' to install shap"
    )

from captum.attr import (
    GradientShap,
    DeepLift,
    IntegratedGradients,
)

# ===========================================================================
# Basic functions
# ===========================================================================


def proj_l1ball(w0, eta, device="cpu"):
    # To help you understand, this function will perform as follow:
    #    a1 = torch.cumsum(torch.sort(torch.abs(y),dim = 0,descending=True)[0],dim=0)
    #    a2 = (a1 - eta)/(torch.arange(start=1,end=y.shape[0]+1))
    #    a3 = torch.abs(y)- torch.max(torch.cat((a2,torch.tensor([0.0]))))
    #    a4 = torch.max(a3,torch.zeros_like(y))
    #    a5 = a4*torch.sign(y)
    #    return a5

    w = torch.as_tensor(w0, dtype=torch.get_default_dtype(), device=device)

    init_shape = w.size()

    if w.dim() > 1:
        init_shape = w.size()
        w = w.reshape(-1)

    Res = torch.sign(w) * torch.max(
        torch.abs(w)
        - torch.max(
            torch.cat(
                (
                    (
                        torch.cumsum(
                            torch.sort(torch.abs(w), dim=0,
                                       descending=True)[0],
                            dim=0,
                            dtype=torch.get_default_dtype(),
                        )
                        - eta
                    )
                    / torch.arange(
                        start=1,
                        end=w.numel() + 1,
                        device=device,
                        dtype=torch.get_default_dtype(),
                    ),
                    torch.tensor(
                        [0.0], dtype=torch.get_default_dtype(), device=device),
                )
            )
        ),
        torch.zeros_like(w),
    )

    Q = Res.reshape(init_shape).clone().detach()

    if not torch.is_tensor(w0):
        Q = Q.data.numpy()
    return Q


def proj_l21ball(w2, eta, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        W = torch.tensor(
            [torch.sum(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(ncol):
            Res[:, i] = proj_l2ball(w[:, i], PW[i].data.item(), device=device)

        Q = Res.clone().detach().requires_grad_(True)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    return Q


def proj_l2ball(w0, eta, device="cpu"):
    w = torch.as_tensor(w0, dtype=torch.get_default_dtype(), device=device)

    n = torch.linalg.norm(w, ord=2)
    if n <= eta:
        return w
    return torch.mul(eta / n, w)


# fold in ["local","full",partial"]
def proj_nuclear(w0, eta_star, fold="local", device="cpu"):

    w1 = torch.as_tensor(w0, dtype=torch.get_default_dtype(), device=device)
    init_shape = w1.size()

    if fold == "full":
        w = full_fold_conv(w0)
    elif fold == "partial":
        w = partial_fold_conv(w0)
    else:
        w = w1

    if w.dim() == 1:
        v = proj_l1ball(w, eta_star, device=device)
    elif w.dim() == 2:
        L, S0, R = torch.svd(w, some=True)  # 'economy-size decomposition'
        # norm_nuclear = S0.sum().item() # Note that the S will be a vector but not a diagonal matrix
        v_star = proj_l1ball(S0, eta_star, device=S0.device)
        v = torch.matmul(L, torch.matmul(v_star.diag(), R.t()))
    elif w.dim() > 2:  # occurs only in the case of local folding
        L, S0, R = np.linalg.svd(w.data.numpy(), full_matrices=False)
        # norm_nuclear = S0.sum()
        v_star = proj_l1ball(S0.reshape((-1,)), eta_star, device=device)
        S1 = v_star.reshape(S0.shape)
        v_temp = np.matmul(L, S1[..., None] * R)
        v = torch.as_tensor(v_temp, device=device)

    if fold == "full":
        v = full_unfold_conv(v, init_shape)
    elif fold == "partial":
        v = partial_unfold_conv(v, init_shape)

    Q = v.reshape(init_shape).clone().detach().requires_grad_(True)

    if not torch.is_tensor(w0):
        Q = Q.data.numpy()

    return Q


def proj_l11ball(w2, eta, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        W = torch.tensor(
            [torch.sum(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(ncol):
            Res[:, i] = proj_l1ball(w[:, i], PW[i].data.item(), device=device)

        Q = Res.clone().detach().requires_grad_(True)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    return Q


def proj_l11ball_line(w2, eta, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        W = torch.tensor(
            [torch.sum(torch.abs(w[i, :])).data.item() for i in range(nrow)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(nrow):
            Res[i, :] = proj_l1ball(w[i, :], PW[i].data.item(), device=device)

        Q = Res.clone().detach().requires_grad_(True)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy()

    return Q


def bilevel_proj_l1Inftyball(w2, eta, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        W = torch.tensor(
            [torch.max(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(ncol):
            Res[:, i] = torch.clamp(torch.abs(w[:, i]), max=PW[i].data.item())
            Res[:, i] = Res[:, i].to(device) * torch.sign(w[:, i]).to(device)

        Q = Res.clone().detach().requires_grad_(True)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy()

    return Q


def f1(i, w):
    return torch.max(torch.abs(w[:, i])).data.item()


def f2(i, w, PW):
    return torch.clamp(torch.abs(w[:, i]), max=PW[i].data.item()) * torch.sign(w[:, i])


def proj_l1Inftyball_line(w2, C, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)
    # w = torch.as_tensor(w2, device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, C, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        X = torch.abs(w)
        X = torch.sort(X, 0, True).values
        S = torch.cumsum(X, 0)

        k = [0 for _ in range(ncol)]
        a = [j for j in range(ncol)]
        theta_num = sum([X[0, i] for i in range(ncol)])
        theta_den = ncol
        theta = (theta_num - C) / theta_den
        changed = True
        while changed:
            for j in a:
                i = k[j]
                while i < nrow - 1 and (S[i, j] - theta) / (i + 1) < X[i + 1, j]:
                    i += 1
                theta_num -= S[k[j], j] / (k[j] + 1)
                theta_den -= 1.0 / (k[j] + 1)
                k[j] = i
                if i == nrow - 1 and S[i, j] < theta:
                    a.remove(j)
                    continue
                theta_num += S[k[j], j] / (k[j] + 1)
                theta_den += 1.0 / (k[j] + 1)
            theta_prime = (theta_num - C) / theta_den
            changed = theta_prime != theta
            theta = theta_prime

        Q = w.clone()
        for j in range(ncol):
            if S[-1, j] < theta:
                Q[:, j] = 0
            else:
                mu = (S[k[j], j] - theta) / (k[j] + 1)
                Q[:, j] = torch.min(mu, abs(Q[:, j]))
        Q = Q * torch.sign(w)
        Q = Q.clone().detach().requires_grad_(True)

        # print("Theta = " + str(theta))
    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    return Q


def proj_l1Inftyball_line_unbounded(w2, C, device="cpu"):
    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)
    Q = proj_l1Inftyball_line(w2, C, device)
    if w.dim() == 1:
        return Q
    else:
        return torch.abs(w) * torch.sign(Q)


def proj_l1Inftyball_unbounded(w2, C, device="cpu"):
    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)
    Q = proj_l1Inftyball(w2, C, device)
    if w.dim() == 1:
        return Q
    else:
        return torch.abs(w) * torch.sign(Q)


def proj_l12ball(V, eta, axis=1, threshold=0.001, device="cpu"):

    V = torch.as_tensor(V, dtype=torch.get_default_dtype(), device=device)

    tol = 0.001
    lst_f = []
    test = eta * eta

    if V.dim() == 1:
        return proj_l1ball(V, eta, device=device)

    if axis == 0:
        V = V.T
    Vshape = V.shape
    # m,d = Vshape
    lmbda = 0.0
    # to change in case of tensor
    p = np.ones(Vshape[0], dtype=int) * (Vshape[1] - 1)
    delta = np.zeros(Vshape[0])
    V_abs = np.abs(V)  # maybe transposed if change the value of axis
    sgn = np.sign(V)
    #    V0 = np.sort(V_abs,axis=1)[:,::-1]
    #    V_sum = np.cumsum(V0,axis=1)
    V_sum = np.cumsum(np.sort(V_abs, axis=1)[:, ::-1], axis=1)

    q = np.arange(0, Vshape[1])
    sum_q = np.power(np.array([V_sum[:, qi] for qi in q]), 2)
    sum_q = np.sqrt(sum_q.sum(axis=1))
    lmbda_init = np.max((sum_q / eta - 1) / (q + 1))
    lmbda = lmbda_init
    # lmbda=0
    p = np.argmax(V_sum / (1 + lmbda * np.arange(1, Vshape[1] + 1)), axis=1)

    while np.abs(test) > tol:
        # update lambda
        sum0 = np.array(list(map(lambda x, y: y[x], p, V_sum)))
        sum1 = np.sum(np.power(sum0 / (1 + lmbda * p), 2))
        sum2 = np.sum(p * (np.power(sum0, 2) / np.power(1 + lmbda * p, 3)))
        test = sum1 - eta * eta
        lmbda = lmbda + test / (2 * sum2)
        lst_f.append(test)
        # update p
        p = np.argmax(
            V_sum / (1 + lmbda * np.arange(1, Vshape[1] + 1)), axis=1)

    delta = lmbda * \
        (np.array(list(map(lambda x, y: y[x], p, V_sum))) / (1 + lmbda * p))
    W = V_abs - delta.reshape((-1, 1))
    W[W < 0] = 0
    W = W * sgn
    W[np.where(np.abs(W) < threshold)] = 0
    if axis == 0:
        W = W.T

    return W.float()


def proj_l1inf_numpy(Y, c, tol=1e-5, direction="row"):
    """
    {X : sum_n max_m |X(n,m)| <= c}
    for some given c>0

        Author: Laurent Condat
        Version: 1.0, Sept. 1, 2017

    This algorithm is new, to the author's knowledge. It is based
    on the same ideas as for projection onto the l1 ball, see
    L. Condat, "Fast projection onto the simplex and the l1 ball",
    Mathematical Programming, vol. 158, no. 1, pp. 575-585, 2016.

    The algorithm is exact and terminates in finite time*. Its
    average complexity, for Y of size N x M, is O(NM.log(M)).
    Its worst case complexity, never found in practice, is
    O(NM.log(M) + N^2.M).

    Note : This is a numpy transcription of the original MATLAB code
    *Due to floating point errors, the actual implementation of the algorithm
    uses a tolerance parameter to guarantee halting of the program
    """
    added_dimension = False

    if direction == "col":
        Y = np.transpose(Y)

    if Y.ndim == 1:
        # for vectors
        Y = np.expand_dims(Y, axis=0)
        added_dimension = True

    X = np.flip(np.sort(np.abs(Y), axis=1), axis=1)
    v = np.sum(X[:, 0])
    if v <= c:
        # inside the ball
        X = Y
    else:
        N, M = Y.shape
        S = np.cumsum(X, axis=1)
        idx = np.ones((N, 1), dtype=int)
        theta = (v - c) / N
        mu = np.zeros((N, 1))
        active = np.ones((N, 1))
        theta_old = 0
        while np.abs(theta_old - theta) > tol:
            for n in range(N):
                if active[n]:
                    j = idx[n]
                    while (j < M) and ((S[n, j - 1] - theta) / j) < X[n, j]:
                        j += 1
                    idx[n] = j
                    mu[n] = S[n, j - 1] / j
                    if j == M and (mu[n] - (theta / j)) <= 0:
                        active[n] = 0
                        mu[n] = 0
            theta_old = theta
            theta = (np.sum(mu) - c) / (np.sum(active / idx))
        X = np.minimum(np.abs(Y), (mu - theta / idx) * active)
        X = X * np.sign(Y)

    if added_dimension:
        X = np.squeeze(X)

    if direction == "col":
        X = np.transpose(X)
    return X


def bilevel_proj_l11ball(w2, eta, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        W = torch.tensor(
            [torch.sum(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(ncol):
            Res[:, i] = proj_l1ball(w[:, i], PW[i].data.item(), device=device)

        Q = Res.clone().detach().requires_grad_(True)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    return Q


def proj_l1infball(w0, eta, AXIS=1, device="cpu", tol=1e-5):
    """See the documentation of proj_l1inf_numpy for details
    Note: Due to
    1. numpy's C implementation and
    2. the non-parallelizable nature of the algorithm,
    it is faster to do this projection on the cpu with numpy arrays
    than on the gpu with torch tensors
    """
    w = w0.detach().cpu().numpy()
    res = proj_l1inf_numpy(w, eta, direction="col" if AXIS else "row", tol=tol)
    Q = torch.as_tensor(res, dtype=torch.get_default_dtype(), device=device)
    return Q


def full_fold_conv(M):

    if M.dim() > 2:
        M2 = M.clone().detach()
        init_shape = M2.shape

        row, col = init_shape[0:2]
        N = list(M2.reshape(-1).size())[0]

        Q = torch.transpose(torch.transpose(
            M2, 0, 1).reshape(N).reshape(col, -1), 0, 1)
    else:
        Q = M

    return Q


def full_unfold_conv(M, original_shape):

    if len(list(original_shape)) > 2:
        M2 = M.clone().detach()
        init_shape = original_shape

        inverse_shape = [init_shape[1], init_shape[0]]

        if len(list(init_shape)) > 2:
            last_shape = list(init_shape[2:])
            inverse_shape = inverse_shape + last_shape

        inverse_shape = tuple(inverse_shape)

        row, col = init_shape[0:2]
        N = list(M2.reshape(-1).size())[0]

        Q = torch.transpose(
            torch.transpose(M2, 0, 1).reshape(N).reshape(inverse_shape), 0, 1
        )
    else:
        Q = M

    return Q


def partial_fold_conv(M):

    if M.dim() > 2:
        M2 = M.clone().detach()
        init_shape = list(M2.shape)

        L = len(init_shape)

        Q = torch.cat(
            tuple(
                [
                    torch.cat(tuple([M2[i, j]
                              for j in range(init_shape[1])]), 1)
                    for i in range(init_shape[0])
                ]
            ),
            0,
        )
    else:
        Q = M

    return Q


def partial_unfold_conv(M, original_shape):

    if len(list(original_shape)) > 2:
        M2 = M.clone().detach()
        init_shape = list(original_shape)

        Z = torch.empty(original_shape)

        for i in range(init_shape[0]):
            for j in range(init_shape[1]):
                di = init_shape[2]
                dj = init_shape[3]
                Z[i, j] = M2[
                    i * di: (i * di + init_shape[2]), j * dj: (j * dj + init_shape[3])
                ]
            # print('row: {}-{}, col: {}-{}'.format(i*di,(i*di+init_shape[2]),j*dj,(j*dj+init_shape[3])))
    else:
        Z = M
    return Z


def sort_weighted_projection(y, eta, w, n=None, device="cpu"):
    if type(y) is not torch.Tensor:
        y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    if type(w) is not torch.Tensor:
        w = torch.as_tensor(w, dtype=torch.get_default_dtype())
    if y.dim() > 1:
        y = y.view(-1)
    if w.dim() > 1:
        w = w.view(-1)
    if device is not None and "cuda" in device:
        y = y.cuda()
        w = w.cuda()
    elif y.is_cuda:
        w = w.cuda()
    elif w.is_cuda:
        y = y.cuda()
    if any(w < 0):
        raise ValueError(
            "sort_weighted_projection: The weight should be positive")
    y0 = y * torch.sign(y)
    w = w.type(dtype=y.dtype)
    y0 = y0.type(dtype=y.dtype)
    x = torch.zeros_like(y)
    if n is None:
        n = len(x)
    z = torch.div(y0, w)
    p = torch.argsort(z, descending=True)
    WYs = 0.0
    Ws = 0.0
    for j in p:
        WYs += w[j] * y0[j]
        Ws += w[j] * w[j]
        if ((WYs - eta) / Ws) > z[j]:
            break
    WYs -= w[j] * y0[j]
    Ws -= w[j] * w[j]
    L = (WYs - eta) / Ws
    if n == len(x):
        x = torch.max(torch.zeros_like(y), y0 - w * L)
    else:
        for i in range(n):
            x[i] = max(torch.zeros_like(y), y0[i] - w[i] * L)
    x *= torch.sign(y)
    return x


def sparsity(M, tol=1.0e-3, device="cpu"):
    """
    Return the sparsity for the input matrix M
    ----- INPUT
        M               : (Tensor) the matrix
        tol             : (Scalar,optional) the threshold to select zeros
    ----- OUTPUT
        sparsity         : (Scalar) the spacity of the matrix
    """
    if type(M) is not torch.Tensor:
        M = torch.as_tensor(M, device=device)
    M1 = torch.where(torch.abs(M) < tol, torch.zeros_like(M), M)
    nb_nonzero = len(M1.nonzero())
    return 1.0 - nb_nonzero / M1.numel()


class LoadDataset(torch.utils.data.Dataset):
    """Load data in Pytorch

    Attributes:
        X: numpy array - input data.
        Y: numpy array - labels.
        ind: string - patient id
    """

    def __init__(self, X, Y, ind):
        super().__init__()
        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)
        self.ind = ind

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y.T[i], self.ind[i]


def CrossVal(X, Y, patient_name, BATCH_SIZE=32, nfold=0, seed=1):
    kf = KFold(n_splits=4, shuffle=True, random_state=seed)
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.array(Y)[train_index.astype(int)], np.array(Y)[
            test_index.astype(int)]
        ind_train, ind_test = patient_name[train_index], patient_name[test_index]
        dtrain = LoadDataset(X_train, y_train, ind_train)
        # train_set, _ = torch.utils.data.random_split(dtrain, [1])
        train_dl = torch.utils.data.DataLoader(
            dtrain, batch_size=BATCH_SIZE, shuffle=True
        )
        dtest = LoadDataset(X_test, y_test, ind_test)
        # _, test_set = torch.utils.data.random_split(dtest, [0])
        test_dl = torch.utils.data.DataLoader(dtest, batch_size=1)
        if i == nfold:
            # for i, batch in enumerate(test_dl):
            #     print(batch[0])
            return train_dl, test_dl, len(dtrain), len(dtest), y_test
        i += 1


def CrossValSurv(X, Y, patient_name, BATCH_SIZE=32, nfold=0, seed=1):
    kf = KFold(n_splits=4, shuffle=True, random_state=seed)
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        # Split Y into time and event components
        # Time and event for training
        y_train_time, y_train_event = Y[1][train_index], Y[0][train_index]
        # Time and event for testing
        y_test_time, y_test_event = Y[1][test_index], Y[0][test_index]

        # Get the patient names for train/test sets
        ind_train, ind_test = patient_name[train_index], patient_name[test_index]

        # Assuming LoadDataset takes both time and event as inputs
        dtrain = LoadDataset(X_train, (y_train_event, y_train_time), ind_train)
        train_dl = torch.utils.data.DataLoader(
            dtrain, batch_size=BATCH_SIZE, shuffle=True)

        dtest = LoadDataset(X_test, (y_test_event, y_test_time), ind_test)
        test_dl = torch.utils.data.DataLoader(dtest, batch_size=1)

        if i == nfold:
            return train_dl, test_dl, len(dtrain), len(dtest), (y_test_event, y_test_time)

        i += 1


def TestSet(X, Y, patient_name, BATCH_SIZE=32):
    dtest = LoadDataset(X, Y, patient_name)
    test_dl = torch.utils.data.DataLoader(dtest, batch_size=1)
    return test_dl, len(dtest)


def Activation(activation=None):
    if activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'gelu':
        return torch.nn.GELU()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise NotImplementedError(
            "hidden activation '{}' is not implemented".format(activation))


class FairAutoEncoder(torch.nn.Module):
    """AutoEncoder Net structure, return encode, decode

    Attributes:
        n_inputs: int - number of features.
        n_clusters: int - number of classes.

    Returns:
        encode: tensor - encoded data
    """

    def __init__(self, n_inputs, n_clusters, n_hidden=512,  activation="relu", norm=False):
        super(FairAutoEncoder, self).__init__()
        n_inputs = n_inputs
        hidden1_size = n_hidden
        hidden2_size = n_hidden
        hidden3_size = n_hidden
        hidden4_size = n_hidden
        #        code_size = 2
        code_size = n_clusters
        if norm:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(n_inputs, hidden1_size),
                Activation(activation),
                torch.nn.LayerNorm(hidden1_size),
                torch.nn.Linear(hidden1_size, hidden2_size),
                Activation(activation),
                torch.nn.LayerNorm(hidden2_size),
                torch.nn.Linear(hidden2_size, hidden3_size),
                Activation(activation),
                torch.nn.LayerNorm(hidden3_size),
                torch.nn.Linear(hidden3_size, hidden4_size),
                Activation(activation),
                torch.nn.LayerNorm(hidden4_size),
                torch.nn.Linear(hidden4_size, code_size),
            )
        else:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(n_inputs, hidden1_size),
                Activation(activation),
                torch.nn.Linear(hidden1_size, hidden2_size),
                Activation(activation),
                torch.nn.Linear(hidden2_size, hidden3_size),
                Activation(activation),
                torch.nn.Linear(hidden3_size, hidden4_size),
                Activation(activation),
                torch.nn.Linear(hidden4_size, code_size),
            )

    def forward(self, x):
        encode = self.encoder(x)
        return encode


class LeNet_300_100(nn.Module):
    def __init__(self, n_inputs, n_outputs=2, activation="relu", norm=False):

        super(LeNet_300_100, self).__init__()

        if norm:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(n_inputs, 300),
                Activation(activation),
                torch.nn.LayerNorm(300),
                torch.nn.Linear(300, 100),
                Activation(activation),
                torch.nn.LayerNorm(100),
                torch.nn.Linear(100, n_outputs),
            )
        else:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(n_inputs, 300),
                Activation(activation),
                torch.nn.Linear(300, 100),
                Activation(activation),
                torch.nn.Linear(100, n_outputs),
            )

    def forward(self, x):
        encode = self.encoder(x)
        return encode


class DNN(nn.Module):

    def __init__(
            self,
            in_features,
            out1,
            dropout, ratio=False):

        super(DNN, self).__init__()

        if ratio:
            out2 = int((out1*(out1-1))/2) // 2

            ratio_size = int((out1 * (out1 - 1)) / 2)
            # self.norm = nn.LayerNorm(in_features)
            self.encoder = nn.Sequential(
                nn.Linear(in_features, out1, bias=False),
                nn.ReLU(),
                nn.Lambda(self.ratio),
                nn.LayerNorm(ratio_size),
                nn.Linear(ratio_size, out2, bias=False),
                nn.ReLU(),
                nn.LayerNorm(normalized_shape=out2),
                nn.Dropout(p=dropout),
                nn.Linear(out2, out2, bias=False),
                nn.ReLU(),
                nn.LayerNorm(normalized_shape=out2),
                nn.Dropout(p=dropout),
                nn.Linear(out2, 1)
            )
        else:
            out2 = int((out1*(out1-1))/2) // 2
            self.encoder = nn.Sequential(
                nn.Linear(in_features, out1, bias=False),
                nn.ReLU(),
                nn.Linear(out1, out2, bias=False),
                nn.ReLU(),
                nn.LayerNorm(normalized_shape=out2),
                nn.Dropout(p=dropout),
                nn.Linear(out2, out2, bias=False),
                nn.ReLU(),
                nn.LayerNorm(normalized_shape=out2),
                nn.Dropout(p=dropout),
                nn.Linear(out2, 1)
            )

    def ratio(self, x):
        for i in range(x.shape[1]-1):
            m = (x[:, i+1:]+1)/(x[:, i].view(-1, 1)+1)
            if i == 0:
                new = m
            else:
                new = torch.cat((new, m), 1)
        return (new)

    def forward(self, x):
        encode = self.encoder(x)
        return encode


class netBio(nn.Module):
    def __init__(self, n_inputs, n_outputs=2, n_hidden=300, activation="relu", norm=False):
        super(netBio, self).__init__()

        if norm:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(n_inputs, n_hidden),
                Activation(activation),
                torch.nn.LayerNorm(n_hidden),
                torch.nn.Linear(n_hidden, n_outputs),
            )
        else:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(n_inputs, n_hidden),
                Activation(activation),
                torch.nn.Linear(n_hidden, n_outputs),
            )

    def forward(self, x):
        encode = self.encoder(x)
        return encode


# define the model
class dnn(nn.Module):

    def __init__(
            self,
            in_features,
            out1,
            dropout):

        super(dnn, self).__init__()

        # self.norm = nn.LayerNorm(in_features)
        out2 = int((out1*(out1-1))/2) // 2
        self.linear1 = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=out1,
                bias=False
            )
        )

        self.norm = nn.LayerNorm(int((out1*(out1-1))/2))

        self.linear2 = nn.Sequential(
            nn.Linear(
                in_features=int((out1*(out1-1))/2),
                out_features=out2,
                bias=False
            ),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=out2),
            nn.Dropout(p=dropout)  # 0.1
        )

        self.linear3 = nn.Sequential(
            nn.Linear(
                in_features=out2,
                out_features=out2,
                bias=False
            ),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=out2),
            nn.Dropout(p=dropout)  # 0.1
        )

        self.output = nn.Sequential(
            nn.Linear(
                in_features=out2,
                out_features=1
            )
        )

    def ratio(self, x):
        for i in range(x.shape[1]-1):
            m = (x[:, i+1:]+1)/(x[:, i].view(-1, 1)+1)
            if i == 0:
                new = m
            else:
                new = torch.cat((new, m), 1)
        return (new)

    def forward(self, x):
        x = self.linear1(x)
        x = self.ratio(x)
        x = self.norm(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.output(x)
        return x


def plotGraph(y_data, x_data, y_label, x_label, title):
    # Plot the data
    plt.plot(x_data, y_data)
    # Customize the plot (optional)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()


def RunFCNNNoProj(
    net: nn.Module,
    criterion_regression,
    train_dl,
    train_len: int,
    gaussianKDE,
    test_dl,
    test_len: int,
    optimizer,
    outputPath: str,
    seed,
    SEEDS,
    fold_idx,
    nfolds,
    lr_scheduler=None,
    N_EPOCHS=30,
    AXIS=0,
    typeEpoch=None,

):
    """Full Autoencoder training loop

    Parameters
    ----------
    net : nn.Module
        The neural network to train and evaluate
    criterion_regression : loss module
        The classification loss component
    train_dl : DataLoader
        Training DataLoader
    train_len : int
        Number of samples in the training set
    test_dl : DataLoader
        Testing/Evaluation DataLoader
    test_len : int
        Number of samples in the testing set
    optimizer : Optimizer
        PyTorch optimizer of the model's parameters
    outputPath : str
        Where to save the results of the run (if SAVE_FILE)
    lr_scheduler : LRScheduler, optional
        Learning rate scheduler (NOT IMPLEMENTED), by default None
    N_EPOCHS : int, optional
        Number of epochs for training, by default 30
    AXIS : int, optional
        The projection axis, by default 0

    Returns
    -------
    data_encoder, epoch_loss, best_test, net
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_loss,  train_time = (
        [],
        []
    )
    (
        epoch_val_loss
    ) = ([])
    best_test = np.inf

    for epoch_idx in range(N_EPOCHS):
        t1 = time.perf_counter()
        running_loss = 0
        net.train()
        for i, batch in enumerate(train_dl):
            x = batch[0]
            labels = batch[1]
            # print(labels)

            if torch.cuda.is_available():
                x = x.cuda()
                labels = labels.cuda()

            encoder_out = net(x)

            loss = criterion_regression(encoder_out.flatten(), labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()

            if epoch_idx == N_EPOCHS - 1:
                #                labels = encoder_out.max(1)[1].float()
                if i == 0:
                    data_encoder = torch.cat(
                        (encoder_out, labels.view(-1, 1)), dim=1)
                else:

                    tmp2 = torch.cat((encoder_out, labels.view(-1, 1)), dim=1)
                    data_encoder = torch.cat((data_encoder, tmp2), dim=0)

        t2 = time.perf_counter()
        train_time.append(t2 - t1)
        """
        print(
            "Total loss:",
            running_loss / float(train_len),
            "loss_recon: ",
            running_reconstruction / train_len,
            "loss_classif: ",
            running_classification / train_len,
        )"""
        epoch_loss.append(running_loss / train_len)

        # testing our model
        running_loss = 0
        net.eval()

        for i, batch in enumerate(test_dl):
            with torch.no_grad():
                x = batch[0]
                labels = batch[1]
                if torch.cuda.is_available():
                    x = x.cuda()
                    labels = labels.cuda()

                encoder_out_test = net(x)
                # Compute the loss
                # loss_classification = criterion_classification(torch.nn.functional.log_softmax(encoder_out,dim=1), labels)
                loss = criterion_regression(encoder_out_test.flatten(), labels)
                running_loss += loss.item()

        """
        print(
            "test accuracy : ",
            running_accuracy / test_len,
            "Total test loss:",
            running_loss / float(test_len),
            "test loss_recon: ",
            running_reconstruction / test_len,
            "test loss_classif: ",
            running_classification / test_len,
        )
        """

        if running_loss < best_test:
            best_net_it = epoch_idx
            best_test = running_loss
            torch.save(net.state_dict(), str(outputPath) + "best_net")

        epoch_val_loss.append(running_loss / test_len)

        """
    #print(f"\nFOR EVERY EPOCH {epoch_val_loss}\n")
    title = f"MSE vs Epoch ({'Proj' if run_model == 'MaskGrad' else 'Initial'}, Training, Seed: {
                            seed} in {SEEDS}, Fold {fold_idx+1}/{nfolds})"
    y_data = epoch_classification
    y_label = "MSE"
    x_data = range(0, len(y_data))
    x_label = "Epoch"
    plotGraph(y_data, x_data, y_label, x_label, title)

    #print(f"\nFOR EVERY EPOCH {epoch_val_loss}\n")
    title = f"MSE vs Epoch ({'Proj' if run_model == 'MaskGrad' else 'Initial'}, Validation, Seed: {
                            seed} in {SEEDS}, Fold {fold_idx+1}/{nfolds})"
    y_data = epoch_val_classification
    y_label = "MSE"
    x_data = range(0, len(y_data))
    x_label = "Epoch"
    plotGraph(y_data, x_data, y_label, x_label, title)

    """

    title = f"MSE vs Epoch ( Training, Seed: {seed} "
    title = title + f"Fold {fold_idx+1} in {nfolds})"
    y_data = epoch_loss
    y_label = "MSE"
    x_data = range(0, len(y_data))
    x_label = "Epoch"
    plotGraph(y_data, x_data, y_label, x_label, title)

    print(f"Best net epoch for {typeEpoch} = ", best_net_it)

    # Y_predit=np.array(Y_predit)
    # gaussianKDEPred=sc.gaussian_kde(Y_predit, bw_method=0.2)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # #ax.plot(arr, np.zeros(arr.shape), 'b+', ms=20)  # rug plot
    # x_eval = np.linspace(0, 1, num=len(Y_predit))
    # ax.plot(x_eval, gaussianKDEPred(x_eval), label="Pred",color="tab:blue" )
    # ax.plot(x_eval, gaussianKDE(x_eval), label="True", color="tab:green")
    # plt.xlabel('Weight')
    # plt.ylabel('Density')
    # plt.title('Weight Distribution between predicted and the truth during TESTING')
    # plt.legend(["Predicted","Truth"])
    # plt.show()
    # plt.figure()
    # sns.kdeplot(data=Y_predit, fill=True, bw_adjust=0.6, color="tab:blue")
    # sns.kdeplot(data=Y_true, fill=True, bw_adjust=0.6, color="tab:green")
    # plt.xlabel('Weight')
    # plt.ylabel('Density')
    # plt.title('Weight Distribution between predicted and the truth for TESTING')
    # plt.legend(["Predicted","Truth"])
    # plt.show()

    # if str(run_model) != "ProjectionLastEpoch":
    # plt.figure()
    # plt.plot( epoch_acc )
    # plt.plot( epoch_val_acc )
    # plt.title('Total accuracy classification')
    # plt.show()
    # print(
    #    "{} epochs trained for  {}s , {} s/epoch".format(
    #        N_EPOCHS, sum(train_time), np.mean(train_time)
    #    )
    # )
    return data_encoder, epoch_loss, best_test, net


def RunAutoEncoder(
    net: nn.Module,
    criterion_classification,
    train_dl,
    train_len: int,
    gaussianKDE,
    test_dl,
    test_len: int,
    optimizer,
    outputPath: str,
    TYPE_PROJ,
    seed,
    SEEDS,
    fold_idx,
    nfolds,
    lr_scheduler=None,
    N_EPOCHS=30,
    run_model="No_Proj",
    DO_PROJ_MIDDLE=False,
    ETA=100,
    ETA_STAR=100,
    TOL=1e-3,
    AXIS=0,
    typeEpoch=None,

):
    """Full Autoencoder training loop

    Parameters
    ----------
    net : nn.Module
        The neural network to train and evaluate
    criterion_reconstruction : loss module
        The reconstruction loss component
    criterion_classification : loss module
        The classification loss component
    train_dl : DataLoader
        Training DataLoader
    train_len : int
        Number of samples in the training set
    test_dl : DataLoader
        Testing/Evaluation DataLoader
    test_len : int
        Number of samples in the testing set
    optimizer : Optimizer
        PyTorch optimizer of the model's parameters
    outputPath : str
        Where to save the results of the run (if SAVE_FILE)
    TYPE_PROJ : Projection
        The projection function to use
    lr_scheduler : LRScheduler, optional
        Learning rate scheduler (NOT IMPLEMENTED), by default None
    N_EPOCHS : int, optional
        Number of epochs for training, by default 30
    run_model : str, optional
        The type of model run ("No_Proj" or "MaskGrad" or "ProjectionLastEpoch"), by default "No_Proj"
    DO_PROJ_MIDDLE : bool or list if run_model=="MaskGrad", optional
        Whether to project the middle layer, by default False
    ETA : int, optional
        The projection radius, by default 100
    ETA_STAR : int, optional
        The projection radius for proj_nuclear, by default 100
    TOL : float, optional
        The tolerance for the proj_l1inf algorithm, by default 1e-5
    AXIS : int, optional
        The projection axis, by default 0

    Returns
    -------
    data_encoder, epoch_loss, best_test, net
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_loss,  train_time = (
        [],
        []
    )
    (
        epoch_val_loss
    ) = ([])
    best_test = np.inf
    for epoch_idx in range(N_EPOCHS):
        t1 = time.perf_counter()
        running_loss = 0
        net.train()
        for i, batch in enumerate(train_dl):
            x = batch[0]
            labels = batch[1]
            # print(labels)

            if torch.cuda.is_available():
                x = x.cuda()
                labels = labels.cuda()

            encoder_out = net(x)

            loss = criterion_classification(encoder_out.flatten(), labels)
            optimizer.zero_grad()
            loss.backward()

            # Set the gradient as 0
            if run_model == "MaskGrad":
                net_parameters = list(net.parameters())
                for index, param in enumerate(net_parameters):
                    is_middle = index == (len(net_parameters) / 2) - 1
                    if (
                        not DO_PROJ_MIDDLE
                    ) and is_middle:  # Do no gradient masking at middle layer
                        pass

                    elif index % 2 == 0:
                        param.grad = torch.where(
                            param.data.abs() < 1e-4,
                            torch.zeros_like(param.grad),
                            param.grad,
                        )
            optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()

            if epoch_idx == N_EPOCHS - 1:
                #                labels = encoder_out.max(1)[1].float()
                if i == 0:
                    data_encoder = torch.cat(
                        (encoder_out, labels.view(-1, 1)), dim=1)
                else:

                    tmp2 = torch.cat((encoder_out, labels.view(-1, 1)), dim=1)
                    data_encoder = torch.cat((data_encoder, tmp2), dim=0)

        t2 = time.perf_counter()
        train_time.append(t2 - t1)
        """
        print(
            "Total loss:",
            running_loss / float(train_len),
            "loss_recon: ",
            running_reconstruction / train_len,
            "loss_classif: ",
            running_classification / train_len,
        )"""
        epoch_loss.append(running_loss / train_len)

        # Do projection at last epoch (GRADIENT_MASK)
        if run_model == "ProjectionLastEpoch" and epoch_idx == (N_EPOCHS - 1):
            net_parameters = list(net.parameters())
            for index, param in enumerate(net_parameters):
                is_middle = index == len(net_parameters) / 2 - 1
                # if (
                #     DO_PROJ_MIDDLE == False and is_middle
                # ):  # Do no projection at middle layer
                #     print(
                #         f"Did not project layer {index} ({param.shape}) because: middle"
                #     )
                # elif (
                #     is_decoder_layer and not DO_PROJ_DECODER
                # ):  # Do no projection on the decoder layers
                #     print(
                #           f"Did not project layer {index} ({param.shape}) because: decoder"
                #       )
                if (DO_PROJ_MIDDLE == True or not is_middle):
                    param.data = Projection(
                        param.data, TYPE_PROJ, ETA,  AXIS=AXIS, ETA_STAR=ETA_STAR, device=device, TOL=TOL,).to(device)

# =============================================================================
#                     param.data = Projection(
#                         param.data, TYPE_PROJ, ETA, AXIS=AXIS, ETA_STAR=ETA_STAR, device=device, TOL=TOL,).to(device)
# =============================================================================

        # testing our model
        running_loss = 0
        net.eval()

        for i, batch in enumerate(test_dl):
            with torch.no_grad():
                x = batch[0]
                labels = batch[1]
                if torch.cuda.is_available():
                    x = x.cuda()
                    labels = labels.cuda()

                encoder_out_test = net(x)
                # Compute the loss
                # loss_classification = criterion_classification(torch.nn.functional.log_softmax(encoder_out,dim=1), labels)
                loss = criterion_classification(
                    encoder_out_test.flatten(), labels)
                running_loss += loss.item()

        """
        print(
            "test accuracy : ",
            running_accuracy / test_len,
            "Total test loss:",
            running_loss / float(test_len),
            "test loss_recon: ",
            running_reconstruction / test_len,
            "test loss_classif: ",
            running_classification / test_len,
        )
        """

        if running_loss < best_test:
            best_net_it = epoch_idx
            best_test = running_loss
            torch.save(net.state_dict(), str(outputPath) + "best_net")

        epoch_val_loss.append(running_loss / test_len)

        """
    #print(f"\nFOR EVERY EPOCH {epoch_val_loss}\n")
    title = f"MSE vs Epoch ({'Proj' if run_model == 'MaskGrad' else 'Initial'}, Training, Seed: {
                            seed} in {SEEDS}, Fold {fold_idx+1}/{nfolds})"
    y_data = epoch_classification
    y_label = "MSE"
    x_data = range(0, len(y_data))
    x_label = "Epoch"
    plotGraph(y_data, x_data, y_label, x_label, title)

    #print(f"\nFOR EVERY EPOCH {epoch_val_loss}\n")
    title = f"MSE vs Epoch ({'Proj' if run_model == 'MaskGrad' else 'Initial'}, Validation, Seed: {
                            seed} in {SEEDS}, Fold {fold_idx+1}/{nfolds})"
    y_data = epoch_val_classification
    y_label = "MSE"
    x_data = range(0, len(y_data))
    x_label = "Epoch"
    plotGraph(y_data, x_data, y_label, x_label, title)

    """

    title = f"MSE vs Epoch ({'Proj' if run_model == 'MaskGrad' else 'Initial'}"
    title = title + \
        f"Training, Seed: {seed} in {SEEDS}, Fold {fold_idx+1}in{nfolds})"
    y_data = epoch_loss
    y_label = "MSE"
    x_data = range(0, len(y_data))
    x_label = "Epoch"
    plotGraph(y_data, x_data, y_label, x_label, title)

    print(f"Best net epoch for {typeEpoch} = ", best_net_it)

    # Y_predit=np.array(Y_predit)
    # gaussianKDEPred=sc.gaussian_kde(Y_predit, bw_method=0.2)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # #ax.plot(arr, np.zeros(arr.shape), 'b+', ms=20)  # rug plot
    # x_eval = np.linspace(0, 1, num=len(Y_predit))
    # ax.plot(x_eval, gaussianKDEPred(x_eval), label="Pred",color="tab:blue" )
    # ax.plot(x_eval, gaussianKDE(x_eval), label="True", color="tab:green")
    # plt.xlabel('Weight')
    # plt.ylabel('Density')
    # plt.title('Weight Distribution between predicted and the truth during TESTING')
    # plt.legend(["Predicted","Truth"])
    # plt.show()
    # plt.figure()
    # sns.kdeplot(data=Y_predit, fill=True, bw_adjust=0.6, color="tab:blue")
    # sns.kdeplot(data=Y_true, fill=True, bw_adjust=0.6, color="tab:green")
    # plt.xlabel('Weight')
    # plt.ylabel('Density')
    # plt.title('Weight Distribution between predicted and the truth for TESTING')
    # plt.legend(["Predicted","Truth"])
    # plt.show()

    # if str(run_model) != "ProjectionLastEpoch":
    # plt.figure()
    # plt.plot( epoch_acc )
    # plt.plot( epoch_val_acc )
    # plt.title('Total accuracy classification')
    # plt.show()
    # print(
    #    "{} epochs trained for  {}s , {} s/epoch".format(
    #        N_EPOCHS, sum(train_time), np.mean(train_time)
    #    )
    # )
    return data_encoder, epoch_loss, best_test, net


def RunDeepSurv(
    net: nn.Module,
    criterion_survival,  # Appropriate loss function for survival
    train_dl,
    train_len,
    test_dl,
    test_len,
    optimizer,
    outputPath,
    TYPE_PROJ,
    seed,
    N_EPOCHS=30,
    DO_PROJ_MIDDLE=False,
    ETA=100,
    ETA_STAR=100,
    TOL=1e-3,
    AXIS=0,
    run_model="No_Proj",
    fold_idx=0,
    nfolds=4,
    typeEpoch=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_loss,  train_time = (
        [],
        []
    )
    (
        epoch_val_loss
    ) = ([])
    best_test = np.inf

    for epoch_idx in range(N_EPOCHS):

        model = fds.Custom_CoxPH(net, tt.optim.Adam)

        x_global = []
        duration_global = []
        event_global = []
        t1 = time.perf_counter()
        running_loss = 0
        net.train()
        for i, batch in enumerate(train_dl):
            x = batch[0]
            duration = batch[1][:, 1]
            event = batch[1][:, 0]
            # cast x into numpy and append to x_global
            x_global.append(x.cpu().numpy())
            duration_global.append(duration.cpu().numpy())
            event_global.append(event.cpu().numpy())
            if torch.cuda.is_available():
                x = x.cuda()
                duration = duration.cuda()
                event = event.cuda()

            encoder_out = net(x)  # Forward pass
            loss = model.loss(encoder_out, duration, event)  # Compute loss
            optimizer.zero_grad()  # Zero gradients
            loss.backward()  # Backward pass

            if run_model == "MaskGrad":
                net_parameters = list(net.parameters())
                for index, param in enumerate(net_parameters):
                    is_middle = index == len(net_parameters) / 2 - 1
                    if not DO_PROJ_MIDDLE or not is_middle:
                        pass
                    elif index % 2 == 0:
                        param.grad = torch.where(
                            param.data.abs() < TOL,
                            torch.zeros_like(param.grad),
                            param.grad,
                        )

            optimizer.step()  # Update weights
            with torch.no_grad():
                running_loss += loss.item()

                if epoch_idx == N_EPOCHS - 1:
                    # Collect encoder outputs along with duration and event for analysis
                    if i == 0:
                        data_encoder = torch.cat(
                            (encoder_out, duration.view(-1, 1), event.view(-1, 1)), dim=1
                        )
                    else:
                        tmp = torch.cat(
                            (encoder_out, duration.view(-1, 1), event.view(-1, 1)), dim=1)
                        data_encoder = torch.cat((data_encoder, tmp), dim=0)

            # Apply projection at last epoch (if required)
            if epoch_idx == (N_EPOCHS - 1):
                net_parameters = list(net.parameters())
                for index, param in enumerate(net_parameters):
                    is_middle = index == len(net_parameters) / 2 - 1
                    if not DO_PROJ_MIDDLE or not is_middle:
                        param.data = Projection(
                            param.data, TYPE_PROJ, ETA, AXIS=AXIS, ETA_STAR=ETA_STAR, device=device, TOL=TOL
                        ).to(device)

            running_loss += loss.item()

        t2 = time.perf_counter()
        epoch_loss.append(running_loss / train_len)

        # Model Evaluation on Test Data using integrated Brier Score

        running_loss_test = 0
        net.eval()

        x_global = np.concatenate(x_global)
        duration_global = np.concatenate(duration_global)
        event_global = np.concatenate(event_global)
        model.compute_baseline_hazards(
            x_global, [duration_global, event_global])
        test_duration = []
        test_event = []
        test_x = []
        for i, batch in enumerate(test_dl):
            test_x.append(batch[0].cpu().numpy())
            test_duration.append(batch[1][:, 1].cpu().numpy())
            test_event.append(batch[1][:, 0].cpu().numpy())
        test_x = np.concatenate(test_x)
        test_duration = np.concatenate(test_duration)
        test_event = np.concatenate(test_event)

        test_pred = model.predict_surv_df(test_x)
        ev = fds.EvalSurv(test_pred, test_duration,
                          test_event, censor_surv='km')
        time_grid = np.linspace(test_duration.min(),
                                test_duration.max(), 100)
        running_loss_test = ev.integrated_brier_score(time_grid)

        if running_loss_test < best_test:
            best_net_it = epoch_idx
            best_test = running_loss_test
            torch.save(net.state_dict(), str(outputPath) + "best_net")
            best_model = model

        epoch_val_loss.append(running_loss_test)

    title = f"MSE vs Epoch ( Training, Seed: {seed} "
    title = title + f"Fold {fold_idx+1} in {nfolds})"
    y_data = epoch_loss
    y_label = "MSE"
    x_data = range(0, len(y_data))
    x_label = "Epoch"
    plotGraph(y_data, x_data, y_label, x_label, title)

    print(f"Best net epoch for {typeEpoch} = ", best_net_it)

    return data_encoder, epoch_loss, best_test, net, best_model


def training(seed, feature_len, TYPE_ACTIVATION, DEVICE, n_hidden, norm, feature_names,
             GRADIENT_MASK, net_name, LR, criterion_regression, train_dl, train_len,
             gaussianKDE, test_dl, test_len, outputPath, TYPE_PROJ, SEEDS, fold_idx,
             nfolds, N_EPOCHS, N_EPOCHS_MASKGRAD, DO_PROJ_MIDDLE, ETA, AXIS, TOL):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    net = buildNet(feature_len, TYPE_ACTIVATION,
                   net_name, DEVICE, n_hidden, norm)

    weights_entry, spasity_w_entry = fnp.weights_and_sparsity(net, TOL)

    run_model = "No_proj"
    if GRADIENT_MASK:
        run_model = "ProjectionLastEpoch"

    # Choose projection function
    if not GRADIENT_MASK:
        TYPE_PROJ = "No_proj"
        TYPE_PROJ_NAME = "No_proj"

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=150, gamma=0.1
    )

    data_encoder, epoch_loss, best_test, trained_net, best_model = RunDeepSurv(  # RunDeepSurv
        net,
        criterion_regression,
        train_dl,
        train_len,
        test_dl,
        test_len,
        optimizer,
        outputPath,
        TYPE_PROJ,
        seed,
        N_EPOCHS,
        DO_PROJ_MIDDLE,
        ETA,
        AXIS,
        TOL,
        fold_idx,
        nfolds,
        typeEpoch="Adam",
    )

    weights_interim_enc, _ = fnp.weights_and_sparsity(trained_net, TOL)

    # Do masked gradient
    if GRADIENT_MASK:

        prev_data = [param.data for param in list(trained_net.parameters())]

        # Get initial network and set zeros
        # Recall the SEED to get the initial parameters
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # run AutoEncoder
        net = buildNet(feature_len, TYPE_ACTIVATION,
                       net_name, DEVICE, n_hidden, norm)

        optimizer = torch.optim.Adam(trained_net.parameters(), lr=LR)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 150, gamma=0.1
        )  # unused in the paper

        net_parameters = list(trained_net.parameters())
        for index, param in enumerate(net_parameters):
            is_middle = index == (len(net_parameters) / 2) - 1
            if (
                not DO_PROJ_MIDDLE
            ) and is_middle:  # Do no gradient masking at middle layer
                pass
            elif index % 2 == 0:
                param.data = torch.where(
                    prev_data[index].abs() < TOL,
                    torch.zeros_like(param.data),
                    param.data,
                )

        run_model = "MaskGrad"
        (
            data_encoder,
            epoch_loss,
            best_test,
            net,
            best_model,
        ) = RunDeepSurv(
            trained_net,
            criterion_regression,
            train_dl,
            train_len,
            test_dl,
            test_len,
            optimizer,
            outputPath,
            TYPE_PROJ,
            seed,
            N_EPOCHS_MASKGRAD,
            DO_PROJ_MIDDLE,
            ETA,
            AXIS=AXIS,
            TOL=TOL,
            fold_idx=fold_idx,
            nfolds=nfolds,
            typeEpoch="MaskGrad",
        )

    return data_encoder, net, best_model


def buildNet(feature_len, TYPE_ACTIVATION, net_name, DEVICE, n_hidden, norm):
    if net_name == "LeNet":
        net = LeNet_300_100(n_inputs=feature_len, n_outputs=1, activation=TYPE_ACTIVATION).to(
            DEVICE
        )  # LeNet
    if net_name == "netBio":
        net = netBio(feature_len, 1, n_hidden,
                     # netBio
                     activation=TYPE_ACTIVATION, norm=norm).to(DEVICE)

    if net_name == "FAIR":
        net = FairAutoEncoder(
            # netBio
            feature_len, 1, n_hidden, activation=TYPE_ACTIVATION, norm=norm).to(DEVICE)

    if net_name == "dnn":
        net = DNN(feature_len, 1, 0.1, ratio=False).to(DEVICE)  # netBio

    if net_name == 'DeepSurv':
        net = fds.MLP(feature_len, n_hidden, 1, batch_norm=True,
                      dropout=0.1, output_bias=False).to(DEVICE)
    return net


def selectf(x, feature_name):
    x = x.cpu()
    _, d = x.shape
    mat = []
    lenmax = min(len(feature_name), d)
    for i in range(lenmax):
        mat.append([feature_name[i] + "", np.linalg.norm(x[:, i])])
    mat = sorted(mat, key=lambda norm: norm[1], reverse=True)
    columns = ["Genes", "Weights"]
    res = pd.DataFrame(mat)

    res = res.sort_values(1, axis=0, ascending=False)
    res.columns = columns
    # res.to_csv('{}topGenesCol.csv'.format(outputPath) , sep =';')
    return mat


def runBestNet(
    test_dl,
    outputPath,
    nfold,
    net,
    feature_name,
    test_len,
    SnormGenes=True
):
    """ Load the best net and test it on your test set
    Attributes:
        train_dl, test_dl: train(test) sets
        outputPath: patch to load the net weights
    Return:

        class_test: accuracy of each class for testing
    """
    Y_predit = []
    Y_true = []
    index_pred_probs = []
    net.load_state_dict(torch.load(
        str(outputPath) + "best_net", weights_only=True))
    net.eval()
    # for i, batch in enumerate(train_dl):
    #     x = batch[0]
    #     labels = batch[1]
    #     if torch.cuda.is_available():
    #         x = x.cuda()
    #         labels = labels.cuda()
    #     encoder_out, decoder_out = net(x)
    #     loss_classification = nn.MSELoss(encoder_out.flatten(), labels)

    first = True
    for i, batch in enumerate(test_dl):
        with torch.no_grad():
            x = batch[0]
            labels = batch[1]
            index = batch[2]
            if torch.cuda.is_available():
                x = x.cuda()
                labels = labels.cuda()
            encoder_out = net(x)
            index_pred_probs.append(
                [index[0], labels.item()]
                + encoder_out.detach().cpu().numpy().tolist()[0]
            )

            Y_predit.append(encoder_out.flatten().item())
            Y_true.append(labels.item())

            if first:
                data_encoder = torch.cat(
                    (encoder_out, labels.view(-1, 1)), dim=1)

                first = False
            else:
                tmp2 = torch.cat((encoder_out, labels.view(-1, 1)), dim=1)
                data_encoder = torch.cat((data_encoder, tmp2), dim=0)

    try:
        if nfold != 0:
            df = pd.read_csv(
                "{}Labelspred_value.csv".format(outputPath), sep=";", header=0
            )

            soft = pd.DataFrame(index_pred_probs)
            soft = pd.DataFrame(
                np.concatenate((df.values, soft.values[:, :])),
                columns=["Name", "Initial Value"]
                + ["Predicted Value"],
            )
            soft.to_csv("{}Labelspred_value.csv".format(
                outputPath), sep=";", index=0, float_format='%g')
        else:
            soft = pd.DataFrame(
                index_pred_probs,
                columns=["Name", "Initial Value"]
                + ["Predicted Value"],
            )
            soft.to_csv("{}Labelspred_value.csv".format(
                outputPath), sep=";", index=0, float_format='%g')
    except FileNotFoundError:
        soft = pd.DataFrame(
            index_pred_probs,
            columns=["Name", "Initial Value"]
            + ["Predicted Value"],
        )
        soft.to_csv("{}Labelspred_value.csv".format(outputPath),
                    sep=";", index=0, float_format='%g')

    return (
        data_encoder,
        Y_true,
        Y_predit,
    )


def runBestNet_survie(
    test_dl,
    outputPath,
    nfold,
    net,
    feature_name,
    test_len,
    model,
    SnormGenes=True
):
    """ Test the best DeepSurv model on the test set.
    Attributes:
        test_dl: test dataloader
        outputPath: path to load the model weights
        nfold: fold index for cross-validation (if applicable)
        net: trained DeepSurv network
        feature_name: feature names
        test_len: size of the test dataset
        SnormGenes: indicates gene normalization (if applicable)
    Returns:
        data_encoder: tensor containing predictions and true durations/events
        concordance_index: C-index for model evaluation
    """

    index_pred_probs = []
    net.load_state_dict(torch.load(
        str(outputPath) + "best_net", weights_only=True))
    net.eval()

    durations = []
    events = []
    hazards = []
    xs = []

    first = True

    for i, batch in enumerate(test_dl):
        with torch.no_grad():
            x = batch[0]  # Features
            duration = batch[1][:, 1]  # Observed times
            event = batch[1][:, 0]  # Event indicators
            index = batch[2]  # Sample indices

            if torch.cuda.is_available():
                x = x.cuda()
                duration = duration.cuda()
                event = event.cuda()

            # Predicted hazard (DeepSurv outputs negative log hazard ratios)
            hazard = net(x)

            # Collect predictions
            xs.append(x.cpu().numpy())
            durations.append(duration.cpu().numpy())
            events.append(event.cpu().numpy())

            index_pred_probs.append(
                [index[0], duration.item(), event.item()]
                + x.cpu().numpy().tolist()
            )

            # Build data encoder
            if first:
                data_encoder = torch.cat(
                    (hazard, duration.view(-1, 1), event.view(-1, 1)), dim=1
                )
                first = False
            else:
                tmp2 = torch.cat(
                    (hazard, duration.view(-1, 1), event.view(-1, 1)), dim=1
                )
                data_encoder = torch.cat((data_encoder, tmp2), dim=0)

    # Save predictions to a CSV file
    columns = ["Name", "duration", "event", "Hazard"]
    results_df = pd.DataFrame(index_pred_probs, columns=columns)
    results_df.to_csv(f"{outputPath}Labelspred_value.csv",
                      sep=";", index=False, float_format='%g')

    # # Calculate concordance index (C-index)
    # concordance_index_value = concordance_index(
    #     durations, -1 * np.array(hazards), events
    # )

    # Calculate integrated Brier score
    events = np.concatenate(events)
    durations = np.concatenate(durations)
    xs = np.concatenate(xs)

    # model.compute_baseline_hazards(  # Compute baseline hazards
    #     input=xs
    # )
    test_pred = model.predict_surv_df(np.array(xs))
    ev = fds.EvalSurv(test_pred, np.array(durations),
                      np.array(events), censor_surv='km')
    time_grid = np.linspace(np.array(durations).min(),
                            np.array(durations).max(), 100)
    concordance_index_value = ev.integrated_brier_score(time_grid)

    print(f"integrated_brier_score: {concordance_index_value:.4f}")
    return data_encoder, concordance_index_value, ev


def valueGap(true, predicted, divided):
    """

    Parameters
    ----------
    true : tensor true
    predicted : tensor predicted

    Returns
    -------
    the negative and positive gap

    """
    resultNeg = 0
    resultPos = 0
    true = true.tolist()
    pred = predicted.tolist()
    for i in range(len(true)):
        dif = true[i]-pred[i]
        if dif < 0:
            resultNeg += dif
        if dif > 0:
            resultPos += dif

    return resultNeg*divided/len(true), resultPos*divided/len(true)


def PSNR(true, predicted):
    MSE = mean_squared_error(
        true, predicted
    )

    denom = sum([element for element in true.tolist()]) / \
        len(predicted.tolist())**2

    return MSE/denom


def packClassResult(accuracy_train, accuracy_test, fold_nb, label_name):
    """ Transform the accuracy of each class in different fold to DataFrame
    Attributes:
        accuracy_train: List, class_train in different fold
        accuracy_test: List, class_test in different fold 
        fold_nb: number of fold  
        label_name: name of different classes(Ex: Class 1, Class 2)
    Return:
        df_accTrain: dataframe, training accuracy per Class in different fold 
        df_acctest: dataframe, testing accuracy per Class in different fold     
    """
    columns = ["Global"] + ["Class " + str(x) for x in label_name]
    ind_df = ["Fold " + str(x + 1) for x in range(fold_nb)]
    df_accTrain = pd.DataFrame(accuracy_train, index=ind_df, columns=columns)
    df_accTrain.loc["Mean"] = df_accTrain.apply(lambda x: x.mean())
    df_accTrain.loc["Std"] = df_accTrain.apply(lambda x: x.std())

    df_acctest = pd.DataFrame(accuracy_test, index=ind_df, columns=columns)
    df_acctest.loc["Mean"] = df_acctest.apply(lambda x: x.mean())
    df_acctest.loc["Std"] = df_acctest.apply(lambda x: x.std())
    return df_accTrain, df_acctest


def packMetric(data, fold_nb):
    columns = (
        ["MSE"] + ["RMSE"] + ["MAE"]+["Negative gap"] + ["Positive gap"]+["WD"]
    )
    ind_df = ["Fold " + str(x + 1) for x in range(fold_nb)]

    df = pd.DataFrame(data, index=ind_df, columns=columns)
    df.loc["Mean"] = df.apply(lambda x: x.mean())
    df.loc["Std"] = df.apply(lambda x: x.std())

    return df


def packMetricsResult(data_train, data_test, fold_nb):
    """ Transform the accuracy of each class in different fold to DataFrame
    Attributes:
        accuracy_train: List, class_train in different fold
        accuracy_test: List, class_test in different fold 
        fold_nb: number of fold  
        label_name: name of different classes(Ex: Class 1， Class 2)
    Return:
        df_accTrain: dataframe, training accuracy per Class in different fold 
        df_acctest: dataframe, testing accuracy per Class in different fold     
    """
    df_metricsTrain = packMetric(data_train, fold_nb)
    df_metricsTest = packMetric(data_test, fold_nb)

    return df_metricsTrain, df_metricsTest


def Projection(
    W, TYPE_PROJ=proj_l11ball, ETA=100,  AXIS=0, ETA_STAR=100, device="cpu", TOL=1e-3
):
    """ For different projection, give the correct args and do projection
    Args:
        W: tensor - net weight matrix
        TYPE_PROJ: string and funciont- use which projection
        ETA: int - only for Proximal_PGL1 or Proximal_PGL11 projection
        ETA_STAR: int - only for Proximal_PGNuclear or Proximal_PGL1_Nuclear projection
        AXIS: int 0,1 - only for Proximal_PGNuclear or Proximal_PGL1_Nuclear projection
        device: parameters of projection
    Return:
        W_new: tensor - W after projection
    """
    if TYPE_PROJ == "No_proj":
        W_new = W
    if (
        TYPE_PROJ == proj_l1ball
        or TYPE_PROJ == proj_l11ball
        or TYPE_PROJ == proj_l11ball_line
        or TYPE_PROJ == proj_l21ball
    ):
        W_new = TYPE_PROJ(W, ETA, device)
    if TYPE_PROJ == proj_l12ball:
        W_new = TYPE_PROJ(W, ETA, AXIS, device=device)
    if TYPE_PROJ == proj_l1Inftyball_line_unbounded:
        W_new = TYPE_PROJ(W, ETA, device=device)
    if TYPE_PROJ == proj_l1infball:

        W_new = TYPE_PROJ(W, ETA, AXIS, device=device, tol=TOL)
    if TYPE_PROJ == bilevel_proj_l1Inftyball:
        W_new = TYPE_PROJ(W, ETA, device)

    if TYPE_PROJ == proj_nuclear:
        W_new = TYPE_PROJ(W, ETA_STAR, device=device)
    return W_new


def ShowPcaTsne(X, Y, data_encoder, center_distance, class_len, tit):
    """ Visualization with PCA and Tsne
    Args:
        X: numpy - original imput matrix
        Y: numpy - label matrix
        data_encoder: tensor  - latent sapce output, encoded data
        center_distance: numpy - center_distance matrix
        class_len: int - number of class
    Return:
        Non, just show results in 2d space
    """

    # Define the color list for plot
    color = [
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#BCBD22",
        "#17BECF",
        "#40004B",
        "#762A83",
        "#9970AB",
        "#C2A5CF",
        "#E7D4E8",
        "#F7F7F7",
        "#D9F0D3",
        "#A6DBA0",
        "#5AAE61",
        "#1B7837",
        "#00441B",
        "#8DD3C7",
        "#FFFFB3",
        "#BEBADA",
        "#FB8072",
        "#80B1D3",
        "#FDB462",
        "#B3DE69",
        "#FCCDE5",
        "#D9D9D9",
        "#BC80BD",
        "#CCEBC5",
        "#FFED6F",
    ]
    color_original = [color[i] for i in Y]

    # Do pca for original data
    pca = PCA(n_components=2)
    X_pca = X if class_len == 2 else pca.fit(X).transform(X)
    X_tsne = X if class_len == 2 else TSNE(n_components=2).fit_transform(X)

    # Do pca for encoder data if cluster>2
    if data_encoder.shape[1] != 3:  # layer code_size >2  (3= 2+1 data+labels)
        data_encoder_pca = data_encoder[:, :-1]
        X_encoder_pca = pca.fit(data_encoder_pca).transform(data_encoder_pca)
        X_encoder_tsne = TSNE(n_components=2).fit_transform(data_encoder_pca)
        Y_encoder_pca = data_encoder[:, -1].astype(int)
    else:
        X_encoder_pca = data_encoder[:, :-1]
        X_encoder_tsne = X_encoder_pca
        Y_encoder_pca = data_encoder[:, -1].astype(int)
    color_encoder = [color[i] for i in Y_encoder_pca]

    # Do pca for center_distance
    labels = np.unique(Y)
    center_distance_pca = pca.fit(center_distance).transform(center_distance)
    color_center_distance = [color[i] for i in labels]

    # Plot
    title2 = tit

    plt.figure()
    plt.title(title2)
    plt.scatter(X_encoder_pca[:, 0], X_encoder_pca[:, 1], c=color_encoder)

    plt.show()


def CalculateDistance(x):
    """ calculate columns pairwise distance
    Args:
         x: matrix - with shape [m, d]
    Returns:
         dist: matrix - with shape [d, d]
    """
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist


def Covariance(m, bias=False, rowvar=True, inplace=False):
    """ Estimate a covariance matrix given data(tensor).
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: numpy array - A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: bool - If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """

    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1) if not bias else 1.0 / (m.size(1))
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def Reconstruction(INTERPELLATION_LAMBDA, data_encoder, net, class_len):
    """ Reconstruction the images by using the centers in laten space and data after interpellation
    Args:
         INTERPELLATION_LAMBDA: float - [0,1], interpolated_data = (1-λ)*x + λ*y
         data_encoder: tensor - data in laten space (output of encoder)
         net: autoencoder net

    Returns:
         center_mean: numpy - with shape[class_len, class_len], center of each cluster
         interpellation_latent: numpy - with shape[class_len, class_len], interpolated data

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # For interpellation
    interpellation_latent = np.zeros((class_len, class_len))
    # center of encoder data
    center_mean = np.zeros((class_len, class_len))
    center_latent = np.zeros((class_len, class_len))
    center_Y = np.unique(data_encoder[:, -1])
    for i in range(class_len):
        # For interpellation
        data_i = (data_encoder[data_encoder[:, -1] == center_Y[i]])[:, :-1]
        index_x, index_y = np.random.randint(0, data_i.shape[0], 2)
        interpellation_latent[i] = (
            INTERPELLATION_LAMBDA * data_i[index_x, :]
            + (1 - INTERPELLATION_LAMBDA) * data_i[index_y, :]
        )
        # center of encoder data
        center_mean[i] = data_i.mean(axis=0)

    #    # Decode interpellation data
    #    interpellation_decoded = net.decoder(torch.from_numpy(interpellation_latent).float().to(device))

    # Decode center data
    center_decoded = net.decoder(
        torch.from_numpy(center_mean).float().to(device))

    # Distance of each center
    center_distance = CalculateDistance(center_mean)

    # Prediction center data
    for target in range(class_len):
        logits = net.encoder(center_decoded[target])
        prediction = np.argmax(logits.detach().cpu().numpy())
        center_latent[target, :] = logits.cpu().detach().numpy()
        print("Center class: ", target, "Prediction: ", prediction)
    return center_mean, center_distance


def topGenes(
    X, Y, feature_name, feature_len, method, nb_samples, device, net, tol=1e-3
):
    """ Get the rank of features for each class, depends on it's contribution
    Attributes:
        X,Y,feature_name, feature_len,  device : data
        method: 'Shap' is very slow; 'Captum_ig', 'Captum_dl', Captum_gs' give almost the same results
        nb_samples: only for 'Shap', we used a part of the original data, other methods used all original data
    Return:
        res: dataframe, ranked features (a kind of interpretation of neural networks)
    """

    input_x = torch.from_numpy(X).float().to(device)
    if method == "Shap":
        print("Running Shap Model... (It may take a long time)")
        nb_samples = nb_samples
        rand_index = np.random.choice(
            input_x.shape[0], nb_samples, replace=True)
        background = input_x[rand_index]
        Y_rand = Y[rand_index].reshape(-1, 1)
        Y_unique, Y_counts = np.unique(Y_rand, return_counts=True)
        # Create object that can calculate shap values and explain predictions of the model
        explainer = shap.DeepExplainer(net, background)
        # Calculate Shap values, with dimension (y*N*x) y:number of labels, N number of background samples, x number of features
        shap_values = explainer.shap_values(background)
    if method == "Captum_ig":
        baseline = torch.zeros((X.shape)).to(device)
        ig = IntegratedGradients(net)
        attributions, delta = ig.attribute(
            input_x, baseline, target=0, return_convergence_delta=True
        )
    if method == "Captum_dl":
        baseline = torch.zeros((X.shape)).to(device)
        dl = DeepLift(net)
        attributions, delta = dl.attribute(
            input_x, baseline, target=0, return_convergence_delta=True
        )
    if method == "Captum_gs":
        baseline_dist = (torch.randn((X.shape)) * 0.001).to(device)
        gs = GradientShap(net)
        attributions, delta = gs.attribute(
            input_x,
            stdevs=0.09,
            n_samples=10,
            baselines=baseline_dist,
            target=0,
            return_convergence_delta=True,
        )

    feature_rank = np.empty(
        (feature_len, 2), dtype=object
    )  # save ranked features and weights

    x_axis_data = np.arange(input_x.shape[1])
    x_axis_data_labels = list(map(lambda idx: feature_name[idx], x_axis_data))
    dl_attr_test_sum = attributions.cpu().detach().numpy().sum(0)
    dl_attr_test_norm_sum = dl_attr_test_sum / \
        np.linalg.norm(dl_attr_test_sum, ord=1)

    data = list(zip(x_axis_data_labels, dl_attr_test_norm_sum))
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    sorted_label = [x[0] for x in sorted_data]
    sorted_value = [x[1] for x in sorted_data]

    feature_rank[:, 0] = sorted_label
    feature_rank[:, 1] = sorted_value

    # Save results as DAtaFrame
    mat_head = np.array(
        ["topGenes", "Weights"]
    )
    mat_head = mat_head.reshape(1, -1)
    mat = np.r_[mat_head, feature_rank]
    mat[1:, 1] = mat[1:, 1] / float(mat[1, 1])
    # columns = ["Class", "class"]
    ind_df = ["Attributes"] + [str(x) for x in range(feature_len)]
    res = pd.DataFrame(mat, index=ind_df)
    # print(len(indices))
    return res


def show_img(x_list, file_name):
    """Visualization of Matrix, color map

    Attributes:
        x_list: list - list of matrix to be shown.
        titile: list - list of figure title.

    Returns:
        non
    """

    # En valeur absolue
    x = x_list[0]
    d = np.zeros((x.shape[0] + 1, x.shape[1]))
    d[: x.shape[0], : x.shape[1]] = x
    d = np.where(d > 0, d, abs(d))
    d[-1, :] = np.linalg.norm(x, axis=0)

    x = np.array(sorted(d.T, key=lambda d: d[-1], reverse=True))

    x = x[:, :-1].T

    x = (x - x.min()) / (x.max() - x.min())

    plt.figure()
    plt.plot()
    plt.title(file_name[:-4] + " Features sorted")
    jet = plt.cm.get_cmap('jet', 256)
    newcolors = jet(np.linspace(0, 1, 256))
    black = np.array([0/256, 0/256, 0/256, 1])
    newcolors[:2, :] = black
    newcmp = ListedColormap(newcolors)
    im = plt.imshow(
        x,
        cmap=newcmp,
        norm=mpl.colors.Normalize(vmin=x.min(), vmax=x.max()),
        interpolation="nearest",
        aspect="auto",
    )
    plt.colorbar(im)
    plt.tight_layout()
    plt.xlabel("Features")
    plt.ylabel("Neurons")
    plt.show()


def sparsity_line(M, tol=1.0e-3, device="cpu"):
    """Get the line sparsity(%) of M

    Attributes:
        M: Tensor - the matrix.
        tol: Scalar,optional - the threshold to select zeros.
        device: device, cpu or gpu

    Returns:
        spacity: Scalar (%)- the spacity of the matrix.

    """
    if type(M) is not torch.Tensor:
        M = torch.as_tensor(M, device=device)
    M1 = torch.where(torch.abs(M) < tol, torch.zeros_like(M), M)
    M1_sum = torch.sum(M1, 1)
    nb_nonzero = len(M1_sum.nonzero())
    return (1.0 - nb_nonzero / M1.shape[0]) * 100


def sparsity_col(M, tol=1.0e-3, device="cpu"):
    """Get the line sparsity(%) of M

    Attributes:
        M: Tensor - the matrix.
        tol: Scalar,optional - the threshold to select zeros.
        device: device, cpu or gpu

    Returns:
        spacity: Scalar (%)- the spacity of the matrix.

    """
    if type(M) is not torch.Tensor:
        M = torch.as_tensor(M, device=device)
    M1 = torch.where(torch.abs(M) < tol, torch.zeros_like(M), M)
    M1_sum = torch.sum(M1, 0)
    nb_nonzero = len(M1_sum.nonzero())
    return (1.0 - nb_nonzero / M1.shape[1]) * 100


def intersection_of_columns(df):
    sets = [set(df[col]) for col in df.columns]
    intersection = set.intersection(*sets)
    return list(intersection)


def pad_list_to_length(lst, n):
    while len(lst) < n:
        lst.append(' ')
    return lst


def removedf(dataframe, keepDF):
    liste = keepDF.tolist()
    liste.append('Label')
    liste.append('N')
    newDataframe = dataframe[dataframe['Name'].isin(liste)]

    return newDataframe, liste


def ReadData(
    file_name, doScale=True, doLog=True, doRowNorm=False
):
    try:
        data_pd = pd.read_csv(
            "data/" + str(file_name),
            delimiter=";",
            decimal=",",
            header=0,
            encoding="ISO-8859-1",
            low_memory=False,
        )
    except:
        data_pd = pd.read_csv(
            "datas/" + str(file_name),
            delimiter=";",
            decimal=",",
            header=0,
            encoding="ISO-8859-1",
            low_memory=False,
        )

    X = (data_pd.iloc[1:, 1:].values.astype(float)).T
    Y = data_pd.iloc[0, 1:].values.astype(float).astype(np.int64)
    col = data_pd.columns.to_list()
    if col[0] != "Name":
        col[0] = "Name"
    data_pd.columns = col
    feature_name = data_pd["Name"].values.astype(str)[1:]
    label_name = np.unique(Y)
    patient_name = data_pd.columns[1:]
    # Do standardization
    if doLog:
        X = np.log(abs(X + 1))  # Transformation

    X = X - np.mean(X, axis=0)
    if doScale:
        X = scale(X, axis=0)  # Standardization along rows

    if doRowNorm:
        X = X - np.mean(X, axis=1, keepdims=True)

    newY = [0]*len(Y)
    divided = 0
    for i in range(len(Y)):
        divided = math.pow(10, (1 + math.floor(math.log10(Y[i]))))
        newY[i] = Y[i] / divided
    Y = np.array(newY)

    gaussianKDE = sc.gaussian_kde(Y, bw_method=0.2)
    return X, Y, feature_name, label_name, patient_name, gaussianKDE, divided


def ReadDataCV(
    file_name, test_size=0.2, doScale=True, doLog=True, doRowNorm=False
):
    try:
        data_pd = pd.read_csv(
            "data/" + str(file_name),
            delimiter=";",
            decimal=",",
            header=0,
            encoding="ISO-8859-1",
            low_memory=False,
        )
    except:
        data_pd = pd.read_csv(
            "datas/" + str(file_name),
            delimiter=";",
            decimal=",",
            header=0,
            encoding="ISO-8859-1",
            low_memory=False,
        )

    X = (data_pd.iloc[1:, 1:].values.astype(float)).T
    Y = data_pd.iloc[0, 1:].values.astype(float).astype(np.int64) + 1

    col = data_pd.columns.to_list()
    if col[0] != "Name":
        col[0] = "Name"
    data_pd.columns = col

    # split 80%-20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42)

    feature_name = data_pd["Name"].values.astype(str)[1:]
    label_name_train = np.unique(y_train)
    label_name_test = np.unique(y_test)
    patient_name = data_pd.columns[1:]
    # Do standardization
    if doLog:
        X_train = np.log(abs(X_train + 1))  # Transformation
        X_test = np.log(abs(X_test + 1))  # Transformation

    X_train = X_train - np.mean(X_train, axis=0)
    X_test = X_test - np.mean(X_test, axis=0)
    if doScale:
        X_train = scale(X_train, axis=0)  # Standardization along rows
        X_test = scale(X_test, axis=0)  # Standardization along rows

    if doRowNorm:
        X_train = X_train - np.mean(X_train, axis=1, keepdims=True)
        X_test = X_test - np.mean(X_test, axis=1, keepdims=True)

    # for index, label in enumerate(
    #     label_name
    # ):  # convert string labels to number (0,1,2....)
    #     Y = np.where(Y == label, index, Y)
    # min max scaling
    newY = [0]*len(y_train)
    divided = 0

    for i in range(len(y_train)):
        divided = math.pow(10, (1 + math.floor(math.log10(y_train[i]))))
        newY[i] = y_train[i] / divided
    y_train = np.array(newY)

    newY = [0]*len(y_test)
    divided = 0
    for i in range(len(y_test)):
        divided = math.pow(10, (1 + math.floor(math.log10(y_test[i]))))
        newY[i] = y_test[i] / divided
    y_test = np.array(newY)

    # if not Y.all():
    #     Y += 1  # 0,1,2,3.... labels -> 1,2,3,4... labels

    # Y = [i/100 for i in Y]
    gaussianKDETrain = sc.gaussian_kde(y_train, bw_method=0.2)
    gaussianKDETest = sc.gaussian_kde(y_test, bw_method=0.2)
    return X_train, X_test, y_train, y_test, feature_name, label_name_train, label_name_test, patient_name, gaussianKDETrain, gaussianKDETest, divided


def ReadDataCV_surv(
    file_name, test_size=0.2, doScale=True, doLog=True, doRowNorm=False
):
    try:
        data_pd = pd.read_csv(
            "data/" + str(file_name),
            delimiter=";",
            decimal=",",
            header=0,
            encoding="ISO-8859-1",
            low_memory=False,
        )
    except:
        data_pd = pd.read_csv(
            "datas/" + str(file_name),
            delimiter=";",
            decimal=",",
            header=0,
            encoding="ISO-8859-1",
            low_memory=False,
        )

    to_drop = []
    for i, name in enumerate(data_pd["Name"]):
        print(name, i)
        if name in ["Exitus", "Exitus Date", "Seroteca-1", "Seroteca-2", "Seroteca-3", "Seroteca-1 Date", "Seroteca-2 Date", "Seroteca-3 Date"]:
            to_drop.append(i)
    data_pd = data_pd.drop(to_drop)

    X = (data_pd.iloc[2:, 1:].values.astype(float)).T
    # apply StandardScaler to X
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    Y = data_pd.iloc[:2, 1:].values.astype(float).T

    col = data_pd.columns.to_list()
    if col[0] != "Name":
        col[0] = "Name"
    data_pd.columns = col

    # split 80%-20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42)
    y_train = y_train.T
    y_test = y_test.T
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    feature_name = data_pd["Name"].values.astype(str)[2:]
    label_name_train = np.unique(y_train)
    label_name_test = np.unique(y_test)
    patient_name = data_pd.columns[1:]
    # Do standardization
    if doLog:
        X_train = np.log(abs(X_train + 1))  # Transformation
        X_test = np.log(abs(X_test + 1))  # Transformation

    X_train = X_train - np.mean(X_train, axis=0)
    X_test = X_test - np.mean(X_test, axis=0)
    if doScale:
        X_train = scale(X_train, axis=0)  # Standardization along rows
        X_test = scale(X_test, axis=0)  # Standardization along rows

    if doRowNorm:
        X_train = X_train - np.mean(X_train, axis=1, keepdims=True)
        X_test = X_test - np.mean(X_test, axis=1, keepdims=True)

    # for index, label in enumerate(
    #     label_name
    # ):  # convert string labels to number (0,1,2....)
    #     Y = np.where(Y == label, index, Y)
    # min max scaling

    gaussianKDETrain = sc.gaussian_kde(y_train, bw_method=0.2)
    gaussianKDETest = sc.gaussian_kde(y_test, bw_method=0.2)
    return X_train, X_test, y_train, y_test, feature_name, label_name_train, label_name_test, patient_name, gaussianKDETrain, gaussianKDETest, 1


if __name__ == "__main__":
    print("This is just a file containing functions, so nothing happened.")
