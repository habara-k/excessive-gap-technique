import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import os
import json

from utils import *
from no_regret import *
from grad_map import gradient_mapping
from breg_proj import bregman_projection


parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=4)
parser.add_argument('-m', type=int, default=5)
parser.add_argument('--step', '-s', type=int, default=10000)
parser.add_argument('--seed', type=int)
args = parser.parse_args()


def main():
    seed = np.random.randint(0,10000) if args.seed is None else args.seed
    np.random.seed(seed)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('nash conv')

    step = np.arange(1, args.step+1)

    nash_conv = {}

    A = np.random.rand(args.n, args.m)
    nash_conv['MWU'] = multiplicative_weights_update(A, step=args.step)
    plt.plot(step, nash_conv['MWU'], label='no-regret(MWU)')

    nash_conv['RM'] = regret_matching(A, step=args.step)
    plt.plot(step, nash_conv['RM'], label='no-regret(RM)')

    nash_conv['OGD'] = online_gradient_descent(A, step=args.step)
    plt.plot(step, nash_conv['OGD'], label='no-regret(OGD)')

    nash_conv['EGT(Euclid)'] = gradient_mapping(A, step=args.step)
    plt.plot(step, nash_conv['EGT(Euclid)'], label='excessive-gap-technique(Euclid)')

    nash_conv['EGT(Entropy)'] = bregman_projection(A, step=args.step)
    plt.plot(step, nash_conv['EGT(Entropy)'], label='excessive-gap-technique(Entropy)')

    plt.legend()
    plt.grid()

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dir = 'log/{}-n={}-m={}-seed={}-step={}'.format(
            now, args.n, args.m, seed, args.step)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    plt.savefig('{}/nash_conv_per_iter.png'.format(dir))

    result = {
            'A': A.tolist(),
            'nash_conv': nash_conv
            }
    with open('{}/result.json'.format(dir), 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main()

