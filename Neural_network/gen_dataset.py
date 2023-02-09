'''
@Author: Clemens Müller

This file is to be used to generate a training dataset for the ML model.

input dataset: PPPQQQ
output dataset: VVVddd
'''

import random as r
import LF_3bus.ElkLoadFlow as elkLF
import LF_3bus.build_sys_from_sheet as build
import numpy as np
import time as t

def gen_data_4bus(upscaling_factor=1.1, learning_samples=10000, verification_samples=2000,
                  learning_file_name = 'learn', verification_file_name = 'verify', path = None):
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
    if path is not None:
        learning_file_name = path + learning_file_name
        verification_file_name = path + verification_file_name

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
                         verification_output_filename, path = None):
    '''
    Function that uses the function solve_nr from LF_3bus.ElkLoadFlow to solve all training and verification samples.
    To conduct all calculations, a loadflow object is imported to the function. The loadflow object is imported once,
    before each sample is solved: .vomag and .voang are formatted for flat start, and the randomly generated Ppowers and
    Qpowers are stores within the object. After the loadflow solution this data is stored in an external NP array.
    The generated data is stored as four .npz files.
    :param learn_filename: filename of the input learning data
    :param verification_filename: filename of the verification data.
    :param obj: the load flow object used access the loadflow solver.
    :param learn_output_filename: name of the output file for learning data.
    :param verification_output_filename: name of the output file for the learning data.
    :param path: path to the temporary folder if using. implemented to avoid mixing different data formats.
    :return: None: stores data as .npz
    '''

    if path is not None:
        learn_filename = path + learn_filename
        verification_filename = path + verification_filename
        learn_output_filename = path + learn_output_filename
        verification_output_filename = path + verification_output_filename

    learn_input = np.load(learn_filename)
    verification_input = np.load(verification_filename)

    random, learn_samples, buses = np.shape(learn_input)
    random1, verification_samples, buses = np.shape(verification_input)

    flat_Vs = obj.vomag
    flat_angles = obj.voang

    learning_results = np.zeros((random, learn_samples, buses))
    verification_results = np.zeros((random, verification_samples, buses))


    def calc_loadflow(samples, storage, lf):
        '''
        Function to calculate the loadflow of each sample.
        :param samples: number of samples to be solved.
        :param storage: predefined np.array to strore data
        :param lf: the loadflow object to perform the calculations
        :return: np.array used to store data.
        '''
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

    np.save(learn_output_filename, learning_results)
    np.save(verification_output_filename, verification_results)

def reformat_data_io(l_input, l_solution, v_input, v_solution, path = None):
    '''
    Function to reformat the data from [2,50,3] to [50,6] for all inputs.
    :param l_input: learning input filename
    :param l_solution: learning solution filename
    :param v_input: verification input filename
    :param v_solution: verification solution filename
    :param path: path to the input data.
    :return: 4 np. arrays: a set of training data and solutions and a set of verification data and solutions
    '''
    if path is not None:
        l_input = path + l_input
        l_solution = path + l_solution
        v_input = path + v_input
        v_solution = path + v_solution

    l_input = np.load(l_input)
    l_solution = np.load(l_solution)
    v_input = np.load(v_input)
    v_solution = np.load(v_solution)

    def append_P_Q(input, solution):
        powers, samples_input, buses = np.shape(input)
        powers, samples_solution, buses = np.shape(solution)
        final_inputs = np.zeros((samples_input, buses*2), dtype=float)# *2 is temporary fix, only applicable for same voltage buses++
        final_solutions = np.zeros((samples_solution, buses*2), dtype=float)
        for i in range(samples_input):
            final_inputs[i] = np.append(input[0][i], input[1][i])
        for i in range(samples_solution):
            final_solutions[i] = np.append(solution[0][i], solution[1][i])
        return final_inputs, final_solutions

    learning_inputs, learning_solutions = append_P_Q(l_input, l_solution)
    verification_inputs, verification_solutions = append_P_Q(v_input, v_solution)

    return learning_inputs, learning_solutions, verification_inputs, verification_solutions



BusList, LineList = build.BuildSystem('/home/clemens/PycharmProjects/NN_LF_Topology/LF_3bus/4 bus 1 gen.xls') #import from excel
bus4 = elkLF.decoupled_LF(BusList, LineList) #create LF object

temp_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/tmp/'

l_filename = 'learn.npy'
v_filename = 'verify.npy'
l_output_filename = 'learn_output.npy'
v_output_filename = 'verify_output.npy'

start = t.perf_counter()

print(f'starting timestamp: {start}')

gen_data_4bus(learning_samples=15000, verification_samples=3000, learning_file_name=l_filename,
              verification_file_name=v_filename, path=temp_path, upscaling_factor=1)
gen_data_time = t.perf_counter()

solve_input_data4bus(l_filename, v_filename, bus4, l_output_filename, v_output_filename, path=temp_path)

solve_input_time = t.perf_counter()

l_i, l_o, v_i, v_o = reformat_data_io(l_filename, l_output_filename, v_filename, v_output_filename, path=temp_path)

np.save('learn_input', l_i)
np.save('learn_output', l_o)
np.save('verification_input', v_i)
np.save('verification_output', v_o)

tot_time = solve_input_time - start
gen_time = gen_data_time - start
solve_time = solve_input_time - gen_data_time


print(f'Tot runtime: {tot_time}s')




