import numpy as np
from tqdm import tqdm

from utils import *


def online_gradient_descent(A, step):
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

        x_avg = x_sum / np.sum(x_sum)
        u_avg = u_sum / np.sum(u_sum)

        nash_conv.append(np.max(x_avg @ A) - np.min(A @ u_avg))

    return nash_conv


def multiplicative_weights_update(A, step):
    (n,m) = A.shape
    x = np.full(n, 1/n)
    u = np.full(m, 1/m)
    x_sum = x
    u_sum = u

    nash_conv = [np.max(x_sum @ A) - np.min(A @ u_sum)]

    k = 1
    j = 0

    for t in tqdm(range(1, step)):
        beta_1 = 1 / (1 + np.sqrt(2 * np.log(n) / k**2))
        beta_2 = 1 / (1 + np.sqrt(2 * np.log(m) / k**2))
        j += 1
        if j == k**2:
            j = 0
            k += 1

        x_nxt = x * np.power(beta_1, A @ u)
        x_nxt /= np.sum(x_nxt)

        u_nxt = u * np.power(beta_2, -x @ A)
        u_nxt /= np.sum(u_nxt)

        x = x_nxt
        u = u_nxt

        x_sum += x
        u_sum += u

        x_avg = x_sum / np.sum(x_sum)
        u_avg = u_sum / np.sum(u_sum)

        nash_conv.append(np.max(x_avg @ A) - np.min(A @ u_avg))

    return nash_conv


def regret_matching(A, step):
    (n,m) = A.shape
    x = np.full(n, 1/n)
    u = np.full(m, 1/m)
    x_sum = x
    u_sum = u

    nash_conv = [np.max(x_sum @ A) - np.min(A @ u_sum)]

    regsum_x = np.zeros(n)
    regsum_u = np.zeros(m)

    for t in tqdm(range(1, step)):
        regsum_x += x @ A @ u - A @ u
        regsum_u += x @ A - x @ A @ u

        x_nxt = np.maximum(regsum_x, 0)
        x_nxt /= np.sum(x_nxt)

        u_nxt = np.maximum(regsum_u, 0)
        u_nxt /= np.sum(u_nxt)

        x = x_nxt
        u = u_nxt

        x_sum += x
        u_sum += u

        x_avg = x_sum / np.sum(x_sum)
        u_avg = u_sum / np.sum(u_sum)

        nash_conv.append(np.max(x_avg @ A) - np.min(A @ u_avg))

    return nash_conv
