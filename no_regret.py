import numpy as np
from tqdm import tqdm

from utils import *


def no_regret_learning(A, step):
    (n,m) = A.shape
    x = np.full(n, 1/n)
    u = np.full(m, 1/m)
    x_sum = x
    u_sum = u

    nash_conv = [np.max(x_sum @ A) - np.min(A @ u_sum)]

    for t in tqdm(range(1, step)):
        eta = 1 / np.sqrt(t)
        x_nxt = euclidean_projection_onto_simplex(x - eta * A @ u)
        u_nxt = euclidean_projection_onto_simplex(u + eta * x @ A)

        x = x_nxt
        u = u_nxt

        x_sum += x
        u_sum += u

        x_avg = x_sum / (t+1)
        u_avg = u_sum / (t+1)

        assert LA.norm(x_avg - euclidean_projection_onto_simplex(x_avg)) < 1e-9
        assert LA.norm(u_avg - euclidean_projection_onto_simplex(u_avg)) < 1e-9
        x_avg = euclidean_projection_onto_simplex(x_avg)
        u_avg = euclidean_projection_onto_simplex(u_avg)

        nash_conv.append(np.max(x_avg @ A) - np.min(A @ u_avg))

    return nash_conv
