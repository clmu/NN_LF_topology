'''
@Author: Clemens Müller

This file is to be used to generate a training dataset for the ML model.

'''

import random as r
import LF_3bus.ElkLoadFlow as elkLF
import numpy as np

def gen_data_4bus(upscaling_factor=1.1, learning_samples=10000, verification_samples=2000):
    '''
    Function generates the indata for training a machine learning algorithm for a simple 4 bus network.
    Powers are generated ramdomly from an interval, and the worst case values are selected for total system load at the
    boundary of convergence for the NR solution method. upscale_factor allows for greater values than the max value at
    each individual node.
    :param upscaling_factor: Factor to adjust the extreme scenarios at individual nodes.
    :param learning_samples: amount of learning samples desired.
    :param verification_samples: Amount of verification samples desired.
    :return: None. Writes input data to two files.
    '''
    up = upscaling_factor #for upscaling the crash value to train the network on smaller samples.
    sizes = [learning_samples, verification_samples]
    data = [[], []]

    for c in range(len(sizes)):

        P1 = np.random.uniform(low=0, high=0.368*up, size=sizes[c])
        Q1 = np.random.uniform(low=0, high=0.23*up, size=sizes[c])

        P2 = np.random.uniform(low=0, high=0.736*up, size=sizes[c])
        Q2 = np.random.uniform(low=0, high=0.368*up, size=sizes[c])

        P3 = np.random.uniform(low=0, high=1.472*up, size=sizes[c])
        Q3 = np.random.uniform(low=0, high=0.736*up, size=sizes[c])

        inputs = np.zeros((sizes[c],6), dtype=float)
        thepows = [P1, Q1, P2, Q2, P3, Q3]

        for i in range(sizes[c]):
            for j in range(len(thepows)):
                inputs[i, j] = thepows[j][i]

        data[c] = inputs

    np.savetxt('learning.csv', data[0], delimiter=',')
    np.savetxt('verify.csv', data[1], delimiter=',')


def solve_input_data4bus(learn_filename, verification_filename)



