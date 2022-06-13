import numpy as np
import matplotlib.pyplot as plt
import argparse

from utils import *
from no_regret import no_regret_learning
from grad_map import gradient_mapping
from breg_proj import bregman_projection


parser = argparse.ArgumentParser()
parser.add_argument('--step', '-s', type=int, default=10000)
parser.add_argument('--output', '-o', default='result.png')
args = parser.parse_args()


if __name__ == '__main__':

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('nash conv')

    step = np.arange(1, args.step+1)
    split = 1000
    mask = [int(split * np.log(t+1) / np.log(args.step)) > int(split * np.log(t) / np.log(args.step)) for t in step]


    A = np.random.rand(4, 5)
    nash_conv = np.array(no_regret_learning(A, step=args.step))
    plt.plot(step[mask], nash_conv[mask], label='no regret')

    nash_conv = np.array(gradient_mapping(A, step=args.step))
    plt.plot(step[mask], nash_conv[mask], label='grad map(euclid)')

    nash_conv = np.array(bregman_projection(A, step=args.step))
    plt.plot(step[mask], nash_conv[mask], label='breg proj(entropy)')

    plt.legend()
    plt.grid()
    plt.savefig(args.output)
