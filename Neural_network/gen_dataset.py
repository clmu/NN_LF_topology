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

def simple_gen_data_4bus(filename='data', upscaling=1.1, samples = 20000, path=None):

    if path is not None:
        filename = path + filename

    up = upscaling #for upscaling the crash value to train the network on smaller samples.
    data = [[], []]
    P1 = np.random.uniform(low=0, high=0.368*up, size=samples)
    Q1 = np.random.uniform(low=0, high=0.23*up, size=samples)

    P2 = np.random.uniform(low=0, high=0.736*up, size=samples)
    Q2 = np.random.uniform(low=0, high=0.368*up, size=samples)

    P3 = np.random.uniform(low=0, high=1.472*up, size=samples)
    Q3 = np.random.uniform(low=0, high=0.736*up, size=samples)

    inputs = np.array([P1, P2, P3, Q1, Q2, Q3], dtype=float)
    inputs = np.transpose(inputs)
    np.save(filename, inputs)
    return inputs

def simple_solve_data(lf_obj, filename='data', path=None, o_filename= 'o_data'):

    if path is not None:
        filename = path + filename
        o_filename = path + o_filename

    input = np.load(filename)
    samples, input_variables = np.shape(input)
    output_variables = input_variables
    output = np.zeros((samples, output_variables), dtype=float)

    def calc_loadflow(nr_samples, data, storage, lf):
        '''
        Function to calculate the loadflow of each sample.
        :param samples: number of samples to be solved.
        :param storage: predefined np.array to strore data
        :param lf: the loadflow object to perform the calculations
        :return: np.array used to store data.
        '''

        flat_Vs = lf.vomag
        flat_angles = lf.voang

        def radToDeg(x):
            return x
            #return x / np.pi * 180
        nr_solution_times = np.zeros((nr_samples,),dtype=float)
        for i in range(nr_samples):
            lf.vomag = flat_Vs
            lf.voang = flat_angles
            lf.ploads = np.append(data[i][:3], 0)
            lf.qloads = np.append(data[i][3:], 0)
            start =t.perf_counter()
            lf.solve_NR()
            nr_solution_times[i] = t.perf_counter() - start
            lf.voang = radToDeg(lf.voang)
            storage[i][:3] = lf.vomag[:3:]
            storage[i][3:] = lf.voang[:3:]
        return storage, nr_solution_times

    learning_results, nr_runtimes = calc_loadflow(samples, input, output, lf_obj)
    np.save(o_filename, learning_results)

    return learning_results, np.average(nr_runtimes)




BusList, LineList = build.BuildSystem('/home/clemens/PycharmProjects/NN_LF_Topology/LF_3bus/4 bus 1 gen.xls') #import from excel
bus4 = elkLF.decoupled_LF(BusList, LineList) #create LF object

temp_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/tmp/'
folder = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'

start = t.perf_counter()

inputs = simple_gen_data_4bus(filename= 'simple data.npy', upscaling=1, samples=60000, path=folder)
outputs, avg_runtime_nr = simple_solve_data(bus4, filename='simple data.npy', path=folder, o_filename='simple o data.npy')

runtime = t.perf_counter() - start

'''
l_filename = 'learn.npy'
v_filename = 'verify.npy'
l_output_filename = 'learn_output.npy'
v_output_filename = 'verify_output.npy'

start = t.perf_counter()

print(f'starting timestamp: {start}')

gen_data_4bus(learning_samples=45000, verification_samples=9000, learning_file_name=l_filename,
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

'''


