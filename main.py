import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime

from utils import *
from no_regret import online_gradient_descent, multiplicative_weights_update
from grad_map import gradient_mapping
from breg_proj import bregman_projection


parser = argparse.ArgumentParser()
parser.add_argument('--step', '-s', type=int, default=10000)
parser.add_argument('--output', '-o', default='result.png')
parser.add_argument('--seed', type=int)
args = parser.parse_args()


def main():
    if args.seed is not None:
        np.random.seed(args.seed)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('nash conv')

    step = np.arange(1, args.step+1)
    split = 1000
    mask = [int(split * np.log(t+1) / np.log(args.step)) > int(split * np.log(t) / np.log(args.step)) for t in step]

    A = np.random.rand(4, 5)
    nash_conv = np.array(online_gradient_descent(A, step=args.step))
    plt.plot(step[mask], nash_conv[mask], label='no-regret(OGD)')

    nash_conv = np.array(multiplicative_weights_update(A, step=args.step))
    plt.plot(step[mask], nash_conv[mask], label='no-regret(MWU)')

    nash_conv = np.array(gradient_mapping(A, step=args.step))
    plt.plot(step[mask], nash_conv[mask], label='gradient mapping(euclid distance)')

    nash_conv = np.array(bregman_projection(A, step=args.step))
    plt.plot(step[mask], nash_conv[mask], label='bregman projection(entropy disance)')

    plt.legend()
    plt.grid()
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig('fig/' + now + '-' + args.output)


if __name__ == '__main__':
    main()
