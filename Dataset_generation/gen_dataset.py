'''
@Author: Clemens Müller

This file is to be used to generate a training dataset for the ML model.

'''

import random as r
import LF_3bus.ElkLoadFlow as elkLF
import LF_3bus.build_sys_from_sheet as build
import numpy as np

def gen_data_4bus(upscaling_factor=1.1, learning_samples=10000, verification_samples=2000,
                  learning_file_name = 'learn', verification_file_name = 'verify'):
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
    data = [[[],[]], [[], []]]
    for c in range(len(sizes)):

        P1 = np.random.uniform(low=0, high=0.368*up, size=sizes[c])
        Q1 = np.random.uniform(low=0, high=0.23*up, size=sizes[c])

        P2 = np.random.uniform(low=0, high=0.736*up, size=sizes[c])
        Q2 = np.random.uniform(low=0, high=0.368*up, size=sizes[c])

        P3 = np.random.uniform(low=0, high=1.472*up, size=sizes[c])
        Q3 = np.random.uniform(low=0, high=0.736*up, size=sizes[c])

        input_P = np.zeros((sizes[c],3), dtype=float)
        input_Q = np.zeros((sizes[c], 3), dtype=float)
        Ps = [P1, P2, P3]
        Qs = [Q1, Q2, Q3]

        for i in range(sizes[c]):
            for j in range(len(Ps)):
                input_P[i, j] = Ps[j][i]
                input_Q[i, j] = Qs[j][i]

        data[c][0] = input_P
        data[c][1] = input_Q

    datalearn = np.array(data[0])
    dataverify = np.array(data[1])


    np.save(learning_file_name, datalearn)
    np.save(verification_file_name, dataverify)


def solve_input_data4bus(learn_filename, verification_filename, obj,  learn_output_filename,
                         verification_output_filename):
    learn_input = np.load(learn_filename)
    verification_input = np.load(verification_filename)

    random, learn_samples, buses = np.shape(learn_input)
    random1, verification_samples, buses = np.shape(verification_input)

    flat_Vs = obj.vomag
    flat_angles = obj.voang

    learning_results = np.zeros((random, learn_samples, buses))
    verification_results = np.zeros((random, verification_samples, buses))


    def calc_loadflow(samples, storage, lf):
        for i in range(samples):
            lf.vomag = flat_Vs
            lf.voang = flat_angles
            lf.ploads = np.append(learn_input[0][i], 0)
            lf.qloads = np.append(learn_input[1][i], 0)
            lf.solve_NR()
            storage[0][i] = lf.vomag[:3:]
            storage[1][i] = lf.voang[:3:]
        return storage
    learning_results = calc_loadflow(learn_samples, learning_results, obj)
    verification_results = calc_loadflow(verification_samples, verification_results, obj)

    np.save('learn_sol.npy', learning_results)
    np.save('verify_sol.npy', verification_results)

def reformat_data_io(l_input, l_solution, v_input, v_solution):
    l_input = np.load(l_input)
    l_solution = np.load(l_solution)
    v_input = np.load(v_input)
    v_solution = np.load(v_solution)

    pq, samples, buses = np.shape(l_input)

    def append_

    for i in range(samples)



BusList, LineList = build.BuildSystem('/home/clemens/PycharmProjects/NN_LF_Topology/LF_3bus/4 bus 1 gen.xls') #import from excel
bus4 = elkLF.decoupled_LF(BusList, LineList) #create LF object

l_filename = 'learn.npy'
v_filename = 'verify.npy'
l_output_filename = 'learn_output.npy'
v_output_filename = 'verify_output.npy'

gen_data_4bus(learning_samples=50, verification_samples=15, learning_file_name=l_filename,
              verification_file_name=v_filename)
solve_input_data4bus(l_filename, v_filename, bus4, l_output_filename, v_output_filename)




