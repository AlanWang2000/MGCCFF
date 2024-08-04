import numpy as np
import os
import random
from scipy.io import loadmat, savemat
from util.metric import cluster, classify



def seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed(10)


res = loadmat('D:/WX/MetaViewer/res.mat')
Z = res['Z']
Y = res['Y'].reshape(-1)
k = res['k'][0][0]

print(Z.shape, Y.shape, k)
cluster(k, Z, Y)
classify(Z, Y)