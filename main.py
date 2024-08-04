import torch
import numpy as np
import random
import os
import warnings
from util.readfile import readfile
from util.metric import cluster, get_avg_metric

from model.MGCCFF import MGCCFF


def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


warnings.filterwarnings("ignore")
seed_torch(1)
X, nums_paired, idx_exist, Y, V, N, clusters = readfile('handwritten_2views_better', paired_rate=0.7, missing_rate=0.5, save=False)

model = MGCCFF(X, Y, clusters, nums_paired, 20)
s, g, y = model.train(lr=0.001, epochs=[200, 200], log_epoch=100)
for i in range(V):
    print('===============view: {}==================='.format(i))
    cluster(clusters, g[i], Y[idx_exist[i]], method='SC', affinity='precomputed')
    get_avg_metric(Y[idx_exist[i]], [y[i]], count=1)
    cluster(clusters, s[i], Y[idx_exist[i]])
    # cluster(clusters, X[i], Y[idx_exist[i]], method='SC')
    cluster(clusters, X[i], Y[idx_exist[i]])

