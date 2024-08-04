import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.container import ModuleList, ParameterList
from torch.autograd import Variable
from itertools import chain
import timeit
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
import random
import os
from scipy import sparse

from util.dataset import MultiDataset
from util.base import DNN
from util.readfile import readfile
from util.metric import cluster, classify, get_avg_metric


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


class DDC:
    def __init__(self, n_clusters, device):
        super().__init__()
        self.eye = torch.eye(n_clusters, device=device)
        self.n_clusters = n_clusters

    def triu(self, X):
        # Sum of strictly upper triangular part
        return torch.sum(torch.triu(X, diagonal=1))

    def _atleast_epsilon(self, X, eps=1E-9):
        """
        Ensure that all elements are >= `eps`.

        :param X: Input elements
        :type X: torch.Tensor
        :param eps: epsilon
        :type eps: float
        :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
        :rtype: torch.Tensor
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    def hidden_kernel(self, x, rel_sigma=0.15, min_sigma=1E-9):
        """
        Compute a kernel matrix from the rows of a matrix.

        :param x: Input matrix
        :type x: torch.Tensor
        :param rel_sigma: Multiplication factor for the sigma hyperparameter
        :type rel_sigma: float
        :return: Kernel matrix
        :rtype: torch.Tensor
        """
        dist = F.relu(torch.cdist(x, x, p=2).square())
        sigma2 = rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(- dist / (2 * sigma2))
        return k

    def d_cs(self, A, K, n_clusters, eps=1E-9):
        """
        Cauchy-Schwarz divergence.

        :param A: Cluster assignment matrix
        :type A:  torch.Tensor
        :param K: Kernel matrix
        :type K: torch.Tensor
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :return: CS-divergence
        :rtype: torch.Tensor
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)
        nom = self._atleast_epsilon(nom)
        dnom_squared = self._atleast_epsilon(dnom_squared, eps=eps**2)
        d = 2 / (n_clusters * (n_clusters - 1)) * self.triu(nom / torch.sqrt(dnom_squared))
        return d

    def get_loss(self, hidden, output):
        kernel = self.hidden_kernel(hidden)
        n = output.size(0)
        m = torch.exp(-torch.cdist(output, self.eye, p=2).square())
        loss_1 = self.d_cs(output, kernel, self.n_clusters)
        loss_2 = self.d_cs(m, kernel, self.n_clusters)
        loss_3 = 2 / (n * (n - 1)) * self.triu(output @ torch.t(output))
        # loss_3_flipped = 2 / (n_clusters * (n_clusters - 1)) * self.triu(torch.t(output) @ output)
        return loss_1 + loss_2 + loss_3


def k_neighbor(W, k=10, set_one=True):
    import numpy as np
    ind = np.argsort(W)
    N = W.shape[0]
    ind0 = ind[:, :N-k]
    ind1 = ind[:, N-k:]
    for i in range(N):
        W[i, ind0[i]] = 0.
        if set_one:
            W[i, ind1[i]] = 1.
    W = (W + W.T) / 2
    return W


class SelfRepresentation(nn.Module):
    def __init__(self, N):
        super(SelfRepresentation, self).__init__()
        self.C = nn.Parameter(1e-4 * torch.rand((N, N)))
        self.N = N

    def forward(self, x):
        C = (torch.abs(self.C) + torch.abs(self.C.T)) / 2
        coef = C - torch.diag(torch.diag(C))
        output = torch.matmul(coef, x)
        return coef, output

    def get_C(self):
        C = (torch.abs(self.C) + torch.abs(self.C.T)) / 2
        coef = C - torch.diag(torch.diag(C))
        return coef

    def set_C(self, c):
        self.C = nn.Parameter(c)


class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.bias = nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float())
    
    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)


class Discriminator(nn.Module):
    def __init__(self, input_dim, V):
        super(Discriminator, self).__init__()
        self.align = ModuleList()
        for i in range(V):
            self.align.append(DNN([input_dim, 100]))
        self.disc = DNN([200, 2])
        # activation function
        self.act = nn.Sigmoid()

    def forward(self, x1, v1, x2, v2):
        a1 = self.align[v1](x1)
        a2 = self.align[v2](x2)
        d = self.disc(torch.cat([a1, a2], 1))
        d = self.act(d)
        d = F.softmax(d, dim=1)
        return d


class Test:
    def __init__(self, X, gt, clusters, nums_paired, device='cuda:0'):
        self.device = device
        self.V = len(X)
        self.N = Y.shape[0]
        self.gt = torch.from_numpy(gt).view(self.N).type(torch.FloatTensor)
        self.n_cluster = clusters
        self.X = {}
        self.nums = {}
        self.P_paired = {}
        for i in range(self.V):
            self.X[i] = torch.from_numpy(X[i] * 1.0).type(torch.FloatTensor).to(device)
            self.nums[i] = X[i].shape[0]
        self.nums_paired = nums_paired
        self.sr = ModuleList()
        self.en = ModuleList()
        self.de = ModuleList()
        self.eye = torch.eye(nums_paired, nums_paired).to(device)
        dims = [[200, 100], [100, 100], [200, 100]]
        acts = [['relu', 'relu'], ['relu', 'relu'], ['relu', 'relu']]
        self.disc = Discriminator(input_dim=dims[-1][-1], V=self.V).to(device)
        for i in range(self.V):
            dim_en = [X[i].shape[1]] + dims[i]
            dim_de = dim_en.copy()
            dim_de.reverse()
            self.en.append(DNN(dim_en, acts[i][0], last=False).to(device))
            self.de.append(DNN(dim_de, acts[i][1], last=False).to(device))
            self.sr.append(SelfRepresentation(self.nums[i]).to(device))
            self.P_paired[i] = torch.eye(self.nums_paired, self.nums[i]).to(device)

    def train(self, lr=0.001, epochs=[100, 100], log_epoch=20):
        start = timeit.default_timer()
        last = start
        self.dataset = MultiDataset(self.X, self.gt)

        op_pre = torch.optim.Adam(chain(self.en.parameters(), self.de.parameters()), lr=lr)

        op_s = torch.optim.Adam(chain(self.en.parameters(), self.de.parameters()), lr=lr)
        op_sr = torch.optim.Adam(self.sr.parameters(), lr=lr)

        op_disc = torch.optim.Adam(self.disc.parameters(), lr=0.01)


        # for epoch in range(200):
        #     for batch_idx, (x, y, idx) in enumerate(dataloader):
        #         self.gt[idx] = y
        #         for i in range(self.V):
        #             x[i] = x[i].to(self.device)
        #         for i in range(self.V):
        #             q_batch = self.query[i](x[i])
        #             k_batch = self.query[i](x[i])
        #             rec_batch = torch.zeros_like(x[i]).cuda()
        #             reg = torch.zeros([1]).cuda()
        #             for ibatch_idx, (ix, iy, iidx) in enumerate(dataloader):
        #                 ix[i] = ix[i].to(self.device)
        #                 k_block = self.key[i](ix[i])
        #                 c = self.thre(q_batch.mm(k_block.t())) / 1000
        #                 rec_batch += c.mm(ix[i])
        #                 reg += torch.norm(torch.norm(c, p=1, dim=0), p=2)
        #             diag_c = self.thre((q_batch * k_batch).sum(dim=1, keepdim=True)) / 1000
        #             rec_batch = rec_batch - diag_c * x[i]
        #             rec_loss = torch.sum(torch.pow(x[i] - rec_batch, 2))
        #             loss = rec_loss + reg

        #             op_sr.zero_grad()
        #             loss.backward()
        #             op_sr.step()

        #             if epoch % 10 == 0:
        #                 output = 'Epoch : {:2.0f} -- Batch : {:2.0f}  ( time : {:.2f} s )'.format(epoch, batch_idx, timeit.default_timer() - last)
        #                 output += '  ===> Total training loss = {:.4f} : loss_re = {:.4f}, loss_norm = {:.4f}'.format(loss.sum(), rec_loss.sum(), reg.sum())
        #                 print(output)
        #                 last = timeit.default_timer()
        
        # for v in range(self.V):
        #     C = torch.empty([batch_size, N])
        #     val = []
        #     indicies = []
        #     k=50
        #     with torch.no_grad():
        #         for batch_idx, (x, y, idx) in enumerate(dataloader):
        #             x[i] = x[i].to(self.device)
        #             q_batch = self.query[i](x[i])
        #             for ibatch_idx, (ix, iy, iidx) in enumerate(dataloader):
        #                 ix[i] = ix[i].to(self.device)
        #                 k_block = self.key[i](ix[i]) 
        #                 temp = self.thre(q_batch.mm(k_block.t())) / 1000
        #                 C[:, iidx] = temp.cpu()

        #             C[list(range(batch_size)), idx] = 0.0

        #             _, index = torch.topk(torch.abs(C), dim=1, k=k)
                    
        #             val.append(C.gather(1, index).reshape([-1]).cpu().data.numpy())
        #             index = index.reshape([-1]).cpu().data.numpy()
        #             indicies.append(index)
            
        #     val = np.concatenate(val, axis=0)
        #     indicies = np.concatenate(indicies, axis=0)
        #     indptr = [k * i for i in range(N + 1)]
            
        #     C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
        #     cluster(self.n_cluster, C, self.gt.cpu().numpy(), method='SC', affinity='precomputed_nearest_neighbors')
        #     C = C_sparse.A
        #     C = np.abs(C + C.T) / 2
        #     cluster(self.n_cluster, C, self.gt.cpu().numpy(), method='SC', affinity='precomputed')

        realF = [None for i in range(self.V)]
        # pre-train
        for epoch in range(epochs[0]):
            loss_ae = []
            for i in range(self.V):
                op_pre.zero_grad()
                s = self.en[i](self.X[i])
                x_re = self.de[i](s)
                loss_ae.append(F.mse_loss(x_re, self.X[i]))
                loss_ae[-1].backward()
                op_pre.step()

            # log
            if epoch % log_epoch == 0:
                output = 'Pre train Epoch : {:2.0f} ( time : {:.2f} s )  ===>  '.format(epoch, timeit.default_timer() - last)
                for i in range(self.V):
                    output += ' view : {}  :  loss_ae = {:.4f}   ||  '.format(i, loss_ae[i])
                print(output)
            last = timeit.default_timer()

        # train
        for epoch in range(epochs[1]):
            # step1: S
            for i in range(self.V):
                op_s.zero_grad()
                s = self.en[i](self.X[i])
                A, s_re = self.sr[i](s)
                x_re = self.de[i](s_re)
                loss_ae = F.mse_loss(x_re, self.X[i])
                loss_ae.backward()
                op_s.step()
            
            # step2: A
            for e in range(10):
                loss_sr = []
                loss_re = []
                loss_norm = []
                for i in range(self.V):
                    loss_p_s = 0
                    op_sr.zero_grad()
                    s = self.en[i](self.X[i])
                    A, s_re = self.sr[i](s)
                    # D = torch.diag(torch.sum(A, dim=0))
                    # L = torch.matmul(torch.matmul(D.pow(-1/2), D - A), D.pow(-1/2))
                    # L = D - A
                    x_re = self.de[i](s_re)
                    loss_re.append(F.mse_loss(s_re, s))
                    # coef_12 = torch.norm(A, p=1, dim=0)
                    # loss_12norm = torch.norm(coef_12, p=2)
                    loss_norm.append((torch.cdist(s, s) * A).mean())
                    # loss_norm += 0.01 * loss_12norm
                    # loss_norm += torch.norm(A, p='fro')
                    # loss_norm += torch.sum(torch.linalg.svdvals(L)[0:self.n_cluster])
                    # loss_norm += torch.sum(torch.sort(torch.real(torch.linalg.eigvals(L)))[0][0:self.n_cluster])
                    # for j in range(0, self.V):
                    #     if i == 1:
                    #         s_ = self.en[j](self.X[j])
                    #         A_, s_re_ = self.sr[j](s_)
                    #         loss_p_s += F.mse_loss(self.P_paired[i].mm(A).mm(self.P_paired[i].T), self.P_paired[j].mm(A_).mm(self.P_paired[j].T).detach())
                    loss_sr.append(loss_re[-1] + loss_norm[-1])
                    loss_sr[-1].backward()
                    op_sr.step()


            # step3: discriminator
            for e in range(50):
                loss_disc = 0
                op_disc.zero_grad()
                for i in range(self.V):
                    s = self.en[i](self.X[i])
                    for j in range(self.V):
                        if i != j:
                            s_ = self.en[j](self.X[j])
                            pos = self.disc(s[:self.nums_paired], i, s_[:self.nums_paired], j)
                            neg1 = self.disc(s[:self.nums_paired], i, torch.roll(s_[:self.nums_paired], 1, 0), j)
                            neg2 = self.disc(s[:self.nums_paired], i, torch.roll(s_[:self.nums_paired], 2, 0), j)
                            p, n = torch.ones_like(pos), torch.ones_like(pos)
                            p[:,1] = 0.
                            n[:,0] = 0.
                            loss_disc += F.mse_loss(pos, p) + (F.mse_loss(neg1, n) + F.mse_loss(neg2, n)) * 0.6
                loss_disc.backward()
                op_disc.step()

            
            # step4: fusion
            # for v in range(self.V):
            #     k = int(self.nums[v] / 2)
            #     g = self.sr[v].get_C().detach()
            #     so = torch.argsort(g, descending=True)
            #     for i in range(self.nums[v]):
            #         # g[i,so[i,:k]] = 1.
            #         g[i,so[i,k:]] = 0.
            #     self.sr[v].set_C(g)


            # log
            if epoch % log_epoch == 0:
                loss_ae = []
                loss_re = []
                loss_norm = []
                loss_disc = []
                for i in range(self.V):
                    s = self.en[i](self.X[i])
                    A, s_re = self.sr[i](s)
                    x_re = self.de[i](s_re)
                    loss_ae.append(F.mse_loss(x_re, self.X[i]))
                    loss_re.append(F.mse_loss(s_re, s.detach()))
                    # D = torch.diag(torch.sum(A, dim=0))
                    # L = D - A
                    # loss_norm += torch.sum(torch.linalg.svdvals(L)[0:self.n_cluster])
                    # loss_norm += torch.sum(torch.sort(torch.real(torch.linalg.eigvals(L)))[0][0:self.n_cluster])
                    loss_norm.append((torch.cdist(s, s) * A).mean())
                    # coef_12 = torch.norm(A, p=1, dim=0)
                    # loss_12norm = torch.norm(coef_12, p=2)
                    loss_disc.append(0)
                    for j in range(self.V):
                        if i != j:
                            s_ = self.en[j](self.X[j])
                            pos = self.disc(s[:self.nums_paired], i, s_[:self.nums_paired], j)
                            neg = self.disc(s[:self.nums_paired], i, torch.roll(s_[:self.nums_paired], 7, 0), j)
                            p, n = torch.ones_like(pos), torch.ones_like(pos)
                            p[:,1] = 0.
                            n[:,0] = 0.
                            loss_disc[-1] += F.mse_loss(pos, p) + F.mse_loss(neg, n)
                output = 'Epoch : {:2.0f} ( time : {:.2f} s ) ===> '.format(epoch, timeit.default_timer() - last)
                for i in range(self.V):
                    output += ' view : {}  :  loss_ae = {:.4f} , loss_re = {:.4f}, loss_norm = {:.4f}, loss_disc = {:.4f}   ||  '.format(i, loss_ae[i], loss_re[i], loss_norm[i], loss_disc[i])
                print(output)
                

                # for i in range(self.V):
                #     g = self.sr[i].get_C().detach()
                #     so = torch.argsort(g, descending=True)
                #     for j in range(self.nums[i]):
                #         # g[j,so[j,:self.nums[i]]] = 1.
                #         g[j,so[j,self.nums[i]:]] = 0.
                #     g = torch.where(g + g.t() > 0., 1., 0.).cpu().numpy()
                #     cluster(clusters, g, Y[idx_exist[i]], method='SC', affinity='precomputed')


            last = timeit.default_timer()


        for i in range(self.V): 
            s = self.en[i](self.X[i])
            A, s_re = self.sr[i](s)
            x_re = self.de[i](s_re)
            for j in range(self.V):
                if i != j:
                    s_ = self.en[j](self.X[j])
                    pos = self.disc(s[:self.nums_paired], i, s_[:self.nums_paired], j)
                    neg = self.disc(s[:self.nums_paired], i, torch.roll(s_[:self.nums_paired], 7, 0), j)
                    pos_true = pos.min(dim=1)[1].detach().cpu().numpy()
                    pos_false = pos.max(dim=1)[1].detach().cpu().numpy()
                    neg_true = neg.min(dim=1)[1].detach().cpu().numpy()
                    neg_false = neg.max(dim=1)[1].detach().cpu().numpy()
                    from sklearn.metrics import accuracy_score
                    p, n = np.ones_like(pos_true), np.zeros_like(pos_false)
                    acc = []
                    acc.append(accuracy_score(pos_true, p)*100)
                    acc.append(accuracy_score(pos_false, n)*100)
                    acc.append(accuracy_score(neg_true, n)*100)
                    acc.append(accuracy_score(neg_false, p)*100)
                    print(acc)
                


        elapsed = (timeit.default_timer() - start)
        print("Time used: {:.2f} s, model ran on: {}".format(elapsed, self.device))
        A = {}
        for i in range(self.V):
            c = self.sr[i].get_C()
            k = 20
            so = torch.argsort(c, descending=True)
            for j in range(self.nums[i]):
                # g[j,so[j,:k]] = 1.
                c[j,so[j,k:]] = 0.
            c = torch.where((c > 0.) * (c.T > 0.), (c + c.T) / 2, c + c.T)
            A[i] = c.cpu().detach().numpy()
        return A


seed_torch(10000)
X, nums_paired, idx_exist, Y, V, N, clusters = readfile('handwritten_2views_better', paired_rate=0.5, missing_rate=0.5, save=False)

model = Test(X, Y, clusters, nums_paired)
A = model.train()
for i in range(V):
    cluster(clusters, A[i], Y[idx_exist[i]], method='SC', affinity='precomputed')
    fig = plt.figure() #调用figure创建一个绘图对象
    ax = fig.add_subplot(111)
    cax = ax.matshow(A[i], vmin=-1, vmax=1)  #绘制热力图，从-1到1
    fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
    plt.savefig('./A_'+str(i)+'.png', format='png') # 保存为图片
