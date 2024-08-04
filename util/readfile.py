from scipy.io import loadmat, savemat
import h5py
import hdf5storage
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


def readfile(filename, paired_rate=0.5, missing_rate=0.5, save=False):
    path = 'D:/WX/dataset/'
    print(path+filename+'.mat')
    X, P, Y, V, N, clusters = None, None, None, None, None, None

    if filename == 'Orl_mtv' or filename == 'yale_mtv' or filename == 'Caltech101-7' or  filename == 'Caltech101-20':
        dataset = loadmat(path+filename+'.mat')
        x = dataset['X']
        Y = dataset['gt']
        Y.shape = (-1)
        V = x.shape[1]
        N = x[0][0].shape[1]
        X = {}
        for i in range(V):
            X[i] = MinMaxScaler([0, 1]).fit_transform((x[0][i] * 1.0).T)

    elif filename == 'handwritten_2views':
        dataset = hdf5storage.loadmat(path+filename+'.mat')
        X = {0: dataset['x1'],
             1: dataset['x2']}
        Y = dataset['gt'].astype(int)
        Y.shape = (-1)
        N = X[0].shape[0]
        V = 2

    elif filename == 'Scene15':
        dataset = loadmat(path+filename+'.mat')
        X = {0: MinMaxScaler((-1, 1)).fit_transform(dataset['X'][0][0]),
             1: MinMaxScaler((-1, 1)).fit_transform(dataset['X'][0][1])}
        Y = dataset['Y'].T[0]
        Y.shape = (-1)
        N = X[0].shape[0]
        V = 2
        
    elif filename == 'HW':
        dataset = hdf5storage.loadmat(path+filename+'.mat')
        V = dataset['X'].shape[1]
        X = {i: dataset['X'][0][i] for i in range(V)}
        print(X[0].shape)
        Y = dataset['Y'].astype(int)
        Y.shape = (-1)
        N = X[0].shape[0]

    elif filename == 'HW2sources':
        dataset = hdf5storage.loadmat(path+filename+'.mat')
        V = dataset['data'].shape[1]
        X = {i: dataset['data'][0][i].T for i in range(V)}
        for i in range(V):
            print(X[i].shape)
        Y = dataset['truelabel'][0][0]
        Y.shape = (-1)
        N = X[0].shape[0]

    elif filename == 'handwritten_2views_better':
        dataset = loadmat(path+filename+'.mat')
        X = {0: dataset['X'][0][0].T,
            1: dataset['X'][0][1].T}
        Y = dataset['gt']
        Y.shape = (-1)
        N = X[0].shape[0]
        V = 2

    elif filename == 'bbcsport-2d-mtv':
        dataset = loadmat(path+filename+'.mat')
        X = {0: MinMaxScaler([0, 1]).fit_transform(dataset['X'][0][0].T),
             1: MinMaxScaler([0, 1]).fit_transform(dataset['X'][0][1].T)}
        Y = dataset['gt']
        Y.shape = (-1)
        N = X[0].shape[0]
        V = 2

    elif filename == 'NH_interval9_mtv':
        dataset = loadmat(path+filename+'.mat')
        X = {0: MinMaxScaler([0, 1]).fit_transform(dataset['X'][0][0].T),
             1: MinMaxScaler([0, 1]).fit_transform(dataset['X'][0][1].T),
             2: MinMaxScaler([0, 1]).fit_transform(dataset['X'][0][2].T)}
        Y = dataset['gt']
        Y.shape = (-1)
        N = X[0].shape[0]
        V = 3
    
    elif filename == 'BDGP':
        dataset = loadmat(path+filename+'.mat')
        X = {0: MinMaxScaler([-1, 1]).fit_transform(dataset['X'][0][0].T),
             1: MinMaxScaler([-1, 1]).fit_transform(dataset['X'][0][1].T)}
        Y = dataset['gt'].T[0]
        Y.shape = (-1)
        N = X[0].shape[0]
        V = 2
    
    elif filename == '3sources':
        dataset = loadmat(path+filename+'.mat')
        X = {0: MinMaxScaler([0, 100]).fit_transform(dataset['X'][0][0].todense()),
             1: MinMaxScaler([0, 100]).fit_transform(dataset['X'][1][0].todense()),
             2: MinMaxScaler([0, 100]).fit_transform(dataset['X'][2][0].todense())}
        Y = dataset['Y']
        Y.shape = (-1)
        N = X[0].shape[0]
        V = 3
    
    elif filename == 'SUNRGBD_fea':
        dataset = loadmat(path+filename+'.mat')
        X = {0: MinMaxScaler([-1, 1]).fit_transform(dataset['X'][0][0]),
             1: MinMaxScaler([-1, 1]).fit_transform(dataset['X'][1][0])}
        Y = dataset['Y']
        Y.shape = (-1)
        N = X[0].shape[0]
        V = 2
    
    elif filename == 'splitmnist_10000' or 'noisymnist_10000':
        dataset = loadmat(path+filename+'.mat')
        X = {0: MinMaxScaler([-1, 1]).fit_transform(dataset['X'][0][0]),
             1: MinMaxScaler([-1, 1]).fit_transform(dataset['X'][0][1])}
        Y = dataset['Y']
        Y.shape = (-1)
        N = X[0].shape[0]
        V = 2

    else:
        print('cannot find this file!   file: {}'.format(path+filename+'.mat'))

    label_max = max(np.unique(Y))
    clusters = len(np.unique(Y))
    if int(label_max) == clusters:
        Y = Y-1


    # idx = np.arange(N)
    # np.random.shuffle(idx)
    # P1 = {}
    # P2 = {}
    # n = int(paired_rate * N)
    # idx_paired = idx[: int(paired_rate * N)]
    # idx_unpaired = idx[int(paired_rate * N):]
    # n = int((N - n) / V)
    # X_nan = {}
    # idx_exist = {}
    # for i in range(V):
    #     # X_all = P1 * X_without_missing
    #     # X_without_missing = P1.T * X_all
    #     # X_all_missing_part = P2 * X_missing
    #     # X_real = X_all + X_all_missing_part
    #     # return X_all
    #     idx_ = idx_unpaired[i*n: i*n+n]
    #     X[i][idx_] = 0.
    #     X_nan[i] = X[i].copy().astype(np.float)
    #     X_nan[i][idx_] = np.nan
    #     idx_exist[i] = list(set(idx)-set(idx_))
    #     idx_exist[i].sort()
    #     idx_exist[i] = np.array(idx_exist[i])
    #     P1[i] = np.delete(np.eye(N), idx_, axis=1)
    #     P2[i] = np.delete(np.eye(N), list(set(idx)-set(idx_)), axis=1)
    #
    # if save is True:
    #     nX = np.zeros((1,V), dtype=object)
    #     nidx_exist = np.zeros((1,V), dtype=object)
    #     nP1 = np.zeros((1,V), dtype=object)
    #     nP2 = np.zeros((1,V), dtype=object)
    #     for i in range(V):
    #         nX[0,i] = X_nan[i]
    #         nP1[0,i] = P1[i]
    #         nP2[0,i] = P2[i]
    #         nidx_exist[0,i] = idx_exist[i]
    #     data = {'X': nX, 'P1': nP1, 'P2': nP2, 'idx_paired': idx_paired, 'idx_exist': nidx_exist, 'Y': Y}
    #     path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/save/'
    #     savemat(path+filename+'_'+str(paired_rate)+'.mat', data)

    idx = np.arange(N)
    np.random.shuffle(idx)
    idx_paired = []
    idx_only = {i:[] for i in range(V)}
    idx_incomplete = []
    idx_left = []
    if paired_rate * N > clusters:
        nums = [np.where(Y==i, 1, 0).sum() for i in range(clusters)]
        temp = [0 for i in range(clusters)]
        for i in idx:
            # ensure that the paired simples include all classes.(Better if the number of samples in each category is average.)
            if temp[Y[i]] < nums[Y[i]] * paired_rate:
                temp[Y[i]] += 1
                idx_paired.append(i)
            # randomly generate missing data
            elif temp[Y[i]] < nums[Y[i]] * paired_rate + (nums[Y[i]] - nums[Y[i]] * paired_rate) * missing_rate:
                temp[Y[i]] += 1
                idx_incomplete.append(i)
                randv = np.arange(V)
                np.random.shuffle(randv)
                for v in range(V // 2):
                    idx_only[randv[v]].append(i)
            else:
                idx_left.append(i)
             
    real_idx = idx_paired + idx_incomplete + idx_left
    Y = Y[real_idx]
    nums_paired = len(idx_paired)
    idx_exist = {i:[] for i in range(V)}
    for i in range(V):
        tmp_idx = idx_paired + idx_only[i] + idx_left
        X[i] = X[i][tmp_idx]
        k = 0
        for j in range(len(real_idx)):
            if tmp_idx[k] == real_idx[j]:
                idx_exist[i].append(j)
                k += 1
    if save:
        nX = np.zeros((1,V), dtype=object)
        nidx_exist = np.zeros((1,V), dtype=object)
        for i in range(V):
            nX[0,i] = X[i]
            nidx_exist[0,i] = idx_exist[i]
        data = {'X': nX, 'paired_rate': paired_rate, 'missing_rate': missing_rate, 'nums_paired': nums_paired, 'idx_exist': nidx_exist, 'Y': Y, 'V': V, 'N': N, 'n_clusters': clusters}
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/save/'
        savemat(path+filename+'_paired_'+str(paired_rate)+'_missing_'+str(paired_rate)+'.mat', data)


    out = 'V: {}(view {} ~ {}), N: {}, nums_paired: {}, clusters: {},  '.format(V, 0, V-1, N, nums_paired, clusters)
    out += 'paired_rate: {}%,  missing_rate: {}%,   shape: '.format(paired_rate*100, missing_rate*100)
    for i in range(V):
        out += 'X[{}]: {}   '.format(i, X[i].shape)
    out += ' Y: {}'.format(Y.shape)
    print(out)

    # return X, P1, P2, idx_paired, idx_exist, Y, V, N, clusters
    return X, nums_paired, idx_exist, Y, V, N, clusters