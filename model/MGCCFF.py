import torch
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from itertools import chain
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import time
import numpy as np

from util.base import DNN, SelfRepresentation
from util.dataset import MultiDataset
from model.DDC import DDC


class MGCCFF:
    def __init__(self, X, gt, clusters, nums_paired, k, device='cuda:0'):
        self.device = device
        self.V = len(X)
        self.N = gt.shape[0]
        self.gt = torch.from_numpy(gt).view(self.N).type(torch.FloatTensor)
        self.n_cluster = clusters
        self.k = k
        self.X = {}
        self.nums = {}
        for i in range(self.V):
            self.X[i] = torch.from_numpy(X[i] * 1.0).type(torch.FloatTensor).to(device)
            self.nums[i] = X[i].shape[0]
        self.nums_paired = nums_paired
        self.sr = ModuleList()
        self.en = ModuleList()
        self.de = ModuleList()
        self.clus = ModuleList()
        dims = [[200, 100], [200, 100], [200, 100], [200, 100], [200, 100], [200, 100]]
        acts = [['relu', 'relu'], ['relu', 'relu'], ['relu', 'relu'], ['relu', 'relu'], ['relu', 'relu'], ['relu', 'relu']]
        self.ddc = DDC(self.n_cluster, device)
        for i in range(self.V):
            dim_en = [X[i].shape[1]] + dims[i]
            dim_de = dim_en.copy()
            dim_de.reverse()
            self.en.append(DNN(dim_en, acts[i][0], last=False).to(device))
            self.de.append(DNN(dim_de, acts[i][1], last=False).to(device))
            self.sr.append(SelfRepresentation(X[i].shape[0]).to(device))
            self.clus.append(DNN([100, self.n_cluster], activation='sigmoid', last=False, softmax=True).to(device))


    def get_s(self):
        p = {}
        y = {}
        c = {}
        s_mu = {}
        s_ae = {}
        s = {}
        for i in range(self.V):
            s_ae[i] = self.en[i](self.X[i])
            y[i] = torch.argmax(self.clus[i](s_ae[i]), dim=1)
            c[i] = torch.sum(F.one_hot(y[i], num_classes=self.n_cluster) * 1.0, dim=0, keepdims=True).T
            s_mu[i] = {}
            for j in range(self.n_cluster):
                s_mu[i][j] = torch.sum(torch.where(y[i].reshape(-1, 1) == j, s_ae[i], 0), dim=0) / (c[i][j] + 1)
            c[i] = c[i]
        for i in range(self.V):
            for j in range(i + 1, self.V):
                w = torch.pow(torch.cdist(c[i], c[j]), 2).cpu().detach().numpy()
                row_ind, col_ind = linear_sum_assignment(w)
                p[i, j] = {}
                p[j, i] = {}
                for k in range(self.n_cluster):
                    p[i, j][row_ind[k]] = col_ind[k]
                    p[i, j][col_ind[k]] = row_ind[k]
                    p[j, i][col_ind[k]] = row_ind[k]
                    p[j, i][row_ind[k]] = col_ind[k]
        for i in range(self.V):
            s[i] = s_ae[i]
            for j in range(s[i].shape[0]):
                for k in range(self.V):
                    if i != k:
                        s[i][j] += s_mu[k][p[i, k][int(y[i][j])]]
            s[i] = (s[i] / self.V).detach()
        return s


    def train(self, lr=0.001, epochs=[200, 200], log_epoch=100):
        self.dataset = MultiDataset(self.X, self.gt)

        op_ae = torch.optim.Adam(chain(self.en.parameters(), self.de.parameters(), self.clus.parameters()), lr=lr)
        op_g = torch.optim.Adam(chain(self.en.parameters(), self.de.parameters(), self.sr.parameters()), lr=lr)
        op_ddc = torch.optim.Adam(chain(self.en.parameters(), self.clus.parameters()), lr=lr)

        s_return = {}
        g_return = {}
        l_return = {}
        y_return = {}

        # pre-train
        for epoch in range(epochs[0]):
            start = time.perf_counter()
            loss_ae = []
            loss_ddc = []
            for i in range(self.V):
                op_ae.zero_grad()
                s_ae = self.en[i](self.X[i])
                x_re = self.de[i](s_ae)
                y = self.clus[i](s_ae)
                loss_ae.append(F.mse_loss(x_re, self.X[i]))
                loss_ddc.append(self.ddc.get_loss(s_ae, y))
                loss_epoch = loss_ae[-1] + loss_ddc[-1]
                # loss_epoch = loss_ae[-1]
                loss_epoch.backward()
                op_ae.step()
            if epoch % log_epoch == 0:
                output = 'Pre1 Epoch : {:2.0f} ( time : {:.2f} s ) ===> '.format(epoch, time.perf_counter() - start)
                for i in range(self.V):
                    output += ' view : {}  :  loss_ae = {:.4f} , loss_ddc = {:.4f}  ||  '.format(i, loss_ae[i], loss_ddc[i])
                print(output)
        for epoch in range(epochs[0]):
            start = time.perf_counter()
            loss_ae = []
            loss_sr = []
            for i in range(self.V):
                op_g.zero_grad()
                s_ae = self.en[i](self.X[i])
                g, s_re = self.sr[i](s_ae)
                x_re = self.de[i](s_re)
                loss_ae.append(F.mse_loss(x_re, self.X[i]))
                loss_sr.append(F.mse_loss(s_re, s_ae))
                loss_epoch = loss_ae[-1] + loss_sr[-1]
                loss_epoch.backward()
                g_return[i] = g
                s_return[i] = s_ae
                op_g.step()
            if epoch % log_epoch == 0:
                output = 'Pre2 Epoch : {:2.0f} ( time : {:.2f} s ) ===> '.format(epoch, time.perf_counter() - start)
                for i in range(self.V):
                    output += ' view : {}  :  loss_ae = {:.4f} , loss_sr = {:.4f}  ||  '.format(i, loss_ae[i], loss_sr[i])
                print(output)

        # train
        for epoch in range(epochs[1]):
            start = time.perf_counter()
            # ddc
            op_ddc.zero_grad()
            loss_ddc = 0
            for i in range(self.V):
                s_ae = self.en[i](self.X[i])
                y = self.clus[i](s_ae)
                loss_ddc += self.ddc.get_loss(s_ae, y)
                y_return[i] = y
                s_return[i] = s_ae
            loss_ddc.backward()
            op_ddc.step()
            # s
            s = self.get_s()
            for e in range(10):
                op_ae.zero_grad()
                loss_ae = 0
                for i in range(self.V):
                    g, s_re = self.sr[i](s[i])
                    x_re = self.de[i](s_re)
                    loss_ae += F.mse_loss(x_re, self.X[i])
                loss_ae.backward()
                op_ae.step()
            # g
            s = self.get_s()
            for e in range(10):
                op_g.zero_grad()
                loss_g = 0
                for i in range(self.V):
                    s_ae = self.en[i](self.X[i])
                    y = self.clus[i](s_ae)
                    g_bar = torch.matmul(y, y.T)
                    g, s_re = self.sr[i](s[i])
                    g_bar = g_bar / torch.sum(g_bar, dim=0)
                    loss_sr = F.mse_loss(s_re, s[i])
                    loss_g_bar = F.mse_loss(g, g_bar)
                    loss_g += loss_sr + loss_g_bar
                    g_return[i] = g
                    y_return[i] = y
                loss_g.backward()
                op_g.step()
                s_return = s

            # log
            if epoch % log_epoch == 0:
                output = 'Epoch : {:2.0f} ( time : {:.2f} s ) ===> '.format(epoch, time.perf_counter() - start)
                # output += 'loss_ddc = {:.4f}'.format(i, loss_ddc)
                output += 'loss_ddc = {:.4f} , loss_ae = {:.4f} , loss_g = {:.4f}'.format(i, loss_ddc, loss_ae, loss_g)
                print(output)


        for i in range(self.V):
            s_return[i] = s_return[i].cpu().detach().numpy()
            g_return[i] = g_return[i].cpu().detach().numpy()
            y_return[i] = y_return[i].cpu().detach().numpy()
            y_return[i] = np.argmax(y_return[i], axis=1)
        return s_return, g_return, y_return

