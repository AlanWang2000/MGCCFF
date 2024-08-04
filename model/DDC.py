import torch
import torch.nn.functional as F


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