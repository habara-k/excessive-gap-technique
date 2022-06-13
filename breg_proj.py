import numpy as np
from numpy import linalg as LA
from tqdm import tqdm

from utils import *


class EntropyDistance:
    def __init__(self, A):
        (n,m) = A.shape
        self.A = A
        self.sigma_1, self.sigma_2 = 1, 1
        self.D_1, self.D_2 = np.log(n), np.log(m)
        self.A_norm = np.max(np.abs(A))
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

    def breg_proj(self, z, g):
        grad = z * np.exp(-g)
        grad /= np.sum(grad)
        return grad



def bregman_projection(A, step):
    dist = EntropyDistance(A)

    scale_1 = dist.A_norm * np.sqrt(dist.D_1 / (dist.sigma_1*dist.sigma_2*dist.D_2))
    scale_2 = dist.A_norm * np.sqrt(dist.D_2 / (dist.sigma_1*dist.sigma_2*dist.D_1))

    mu_2 = scale_2
    gamma = dist.sigma_1*dist.sigma_2*mu_2 / dist.A_norm**2
    u = dist.opt_u(mu_2, dist.x_0)
    x = dist.breg_proj(dist.x_0, gamma * A @ u)

    nash_conv = [np.max(x @ A) - np.min(A @ u)]

    for k in tqdm(range(step-1)):
        tau = 2 / (k+3)
        if k % 2 == 0:
            mu_1 = scale_1 * 2 / (k+1)
            mu_2 = scale_2 * 2 / (k+2)

            opt_x = dist.opt_x(mu_1, u)
            x_tmp = (1-tau)*x + tau*opt_x
            opt_u = dist.opt_u(mu_2, x_tmp)
            u_nxt = (1-tau)*u + tau*opt_u
            grad = dist.breg_proj(opt_x, tau / ((1-tau) * mu_1) * A @ opt_u)
            x_nxt = (1-tau)*x + tau*grad

            x = x_nxt
            u = u_nxt
        else:
            mu_1 = scale_1 * 2 / (k+2)
            mu_2 = scale_2 * 2 / (k+1)

            opt_u = dist.opt_u(mu_2, x)
            u_tmp = (1-tau)*u + tau*opt_u
            opt_x = dist.opt_x(mu_1, u_tmp)
            x_nxt = (1-tau)*x + tau*opt_x
            grad = dist.breg_proj(opt_u, -tau / ((1-tau) * mu_2) * opt_x @ A)  # WARNING
            u_nxt = (1-tau)*u + tau*grad

            x = x_nxt
            u = u_nxt

        assert LA.norm(x - euclidean_projection_onto_simplex(x)) < 1e-9, k
        assert LA.norm(u - euclidean_projection_onto_simplex(u)) < 1e-9, k
        x = euclidean_projection_onto_simplex(x)
        u = euclidean_projection_onto_simplex(u)

        nash_conv.append(np.max(x @ A) - np.min(A @ u))

    return nash_conv

