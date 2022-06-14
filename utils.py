import numpy as np
from numpy import linalg as LA


def euclidean_projection_onto_simplex(y):
    # Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application." arXiv preprint arXiv:1309.1541 (2013).

    (D,) = y.shape
    u = sorted(y)
    u.reverse()

    s = u[:]
    for j in range(1,D):
        s[j] += s[j-1]

    rho = max([j for j in range(D) if u[j] + (1-s[j]) / (j+1) > 0])

    lamb = (1-s[rho]) / (rho+1)

    return np.maximum(y+lamb, 0)


if __name__ == '__main__':
    test_cases = [
            ([1,1,1], [1/3,1/3,1/3]),
            ([1,0,0], [1,0,0]),
            ([2,0,0], [1,0,0]),
            ([1,1,0], [1/2,1/2,0]),
            ]
    for case, expected in test_cases:
        actual = euclidean_projection_onto_simplex(np.array(case))
        assert LA.norm(actual - expected) < 1e-9, \
            'case: {}, actual: {}, expected: {}'.format(case, actual, expected)


def max_eigen_value(A, step = 100):
    (n,m) = A.shape
    assert n == m

    x = np.random.rand(n)
    x /= np.linalg.norm(x)
    for _ in range(step):
        x = A @ x
        x /= np.linalg.norm(x)

    return np.linalg.norm(A @ x)


if __name__ == '__main__':
    test_cases = [
            ([[1,0],[0,1]], 1),
            ([[2,0],[0,1]], 2),
            ([[2,0,0],[0,1,0],[0,0,1]], 2),
            ]
    for case, expected in test_cases:
        actual = max_eigen_value(np.array(case))
        assert LA.norm(actual - expected) < 1e-9, \
            'case: {}, actual: {}, expected: {}'.format(case, actual, expected)

