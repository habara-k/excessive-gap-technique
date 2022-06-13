import numpy as np
from tqdm import tqdm

from utils import *


class EuclideanDistance:
    def __init__(self, A):
        (n,m) = A.shape
        self.A = A
        self.sigma_1, self.sigma_2 = 1, 1
        self.D_1, self.D_2 = (1-1/n)/2, (1-1/m)/2
        self.A_norm = np.sqrt(max_eigen_value(A @ A.T))
        self.x_0 = np.full(n, 1/n)
        self.u_0 = np.full(m, 1/m)

    def opt_x(self, mu_1, u):
        return euclidean_projection_onto_simplex(
                self.x_0 - self.A @ u / mu_1
                )

    def opt_u(self, mu_2, x):
        return euclidean_projection_onto_simplex(
                self.u_0 + x @ self.A / mu_2
                )

    def prim_grad_map(self, mu_2, x, hint_u=None):
        if hint_u is None:
            hint_u = self.opt_u(mu_2, x)

        return euclidean_projection_onto_simplex(
                x - (self.sigma_2*mu_2 / self.A_norm**2) * self.A @ hint_u
                )

    def dual_grad_map(self, mu_1, u, hint_x=None):
        if hint_x is None:
            hint_x = self.opt_x(mu_1, u)

        return euclidean_projection_onto_simplex(
                u + (self.sigma_1*mu_1 / self.A_norm**2) * hint_x @ self.A 
                )


def gradient_mapping(A, step):
    dist = EuclideanDistance(A)

    scale_1 = dist.A_norm * np.sqrt(dist.D_1 / (dist.sigma_1*dist.sigma_2*dist.D_2))
    scale_2 = dist.A_norm * np.sqrt(dist.D_2 / (dist.sigma_1*dist.sigma_2*dist.D_1))

    mu_2 = scale_2
    u = dist.opt_u(mu_2, dist.x_0)
    x = dist.prim_grad_map(mu_2, dist.x_0, hint_u=u)

    nash_conv = [np.max(x @ A) - np.min(A @ u)]

    for k in tqdm(range(step-1)):
        tau = 2 / (k+3)
        if k % 2 == 0:
            mu_1 = scale_1 * 2 / (k+1)
            mu_2 = scale_2 * 2 / (k+2)
            x_tmp = (1-tau)*x + tau*dist.opt_x(mu_1, u) 
            u_nxt = (1-tau)*u + tau*dist.opt_u(mu_2, x_tmp)
            x_nxt = dist.prim_grad_map(mu_2, x_tmp, hint_u=u_nxt)

            x = x_nxt
            u = u_nxt
        else:
            mu_1 = scale_1 * 2 / (k+2)
            mu_2 = scale_2 * 2 / (k+1)
            u_tmp = (1-tau)*u + tau*dist.opt_u(mu_2, x)
            x_nxt = (1-tau)*x + tau*dist.opt_x(mu_1, u_tmp)
            u_nxt = dist.dual_grad_map(mu_1, u_tmp, hint_x=x_nxt)

            x = x_nxt
            u = u_nxt

        assert LA.norm(x - euclidean_projection_onto_simplex(x)) < 1e-9
        assert LA.norm(u - euclidean_projection_onto_simplex(u)) < 1e-9
        x = euclidean_projection_onto_simplex(x)
        u = euclidean_projection_onto_simplex(u)

        nash_conv.append(np.max(x @ A) - np.min(A @ u))

    return nash_conv

