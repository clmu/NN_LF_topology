import numpy as np
from numpy.random import default_rng
import time as t
import random

import matplotlib.pyplot as plt
from LF_3bus.ElkLoadFlow import LoadFlow
from LF_3bus.build_sys_from_sheet import BuildSystem
from PyDSAL.DistLoadFlow_v2 import DistLoadFlow3 as dlf
from PyDSAL.BuildSystem_v2 import BuildSystem3
from Neural_network.NN_objects import pickle_store_object as store
from Neural_network.NN_objects import pickle_load_obj as load


def check_similarity(vec_1, vec_2):

    #if len(vec_1) != len(vec_2):
    #    print('Vectors not of same length')
    #else:
    return vec_1 - vec_2

def gen_p_q_tuple(buslist):
    '''
    Function that stores system loading in two tuples.
    :param buslist: buslist of a DLF object.
    :return: two tuples containing the system p and q loading respectively.
    '''
    p = np.zeros(len(buslist), dtype=float)
    q = np.zeros(len(buslist), dtype=float)
    for bus_idx in range(len(buslist)):
        p[bus_idx] = buslist[bus_idx].pload
        q[bus_idx] = buslist[bus_idx].pload
    p_tup = tuple(p)
    q_tup = tuple(p)
    return p_tup, q_tup

def gen_single_set(lf_obj, p_pows, q_pows, accuracy=0.00001, low=0.8, high=1.2):

    '''
    Function takes in two DLF objects, and solves a random FBS for one object before returning the input and output
     data as np.arrays for later use. One of the objects acts as a reference to keep the original values used as a
     baseline, the other to store the randomly generated loads. The objects should be initialized using identical
     procedures.
    :param ref_ob: DLF object containing the original load data.
    :param lf_obj: DLF objet containing the randomly generated load data.
    :param low: low boundary for random generation
    :param high: high boundary for random generation
    :return: np.arrays of input data and solutions.
    '''

    #if len(ref_obj.BusList) != len(lf_obj.BusList):
    #    raise Exception('DLF objects are not identical')


    #low, high = 0.8, 1.2
    input_sample = np.zeros((len(lf_obj.BusList)-1)*2, dtype=float)
    output_sample = np.zeros((len(lf_obj.BusList)-1)*2, dtype=float)

    for bus_idx in range(len(lf_obj.BusList)):
        ''' #old numpy random
        altered_pload = ref_obj.BusList[bus_idx].pload * np.random.uniform(low=low, high=high)
        altered_qload = ref_obj.BusList[bus_idx].qload * np.random.uniform(low=low, high=high)
        '''

        #new numpy random
        generator = default_rng()
        altered_pload = p_pows[bus_idx] * generator.uniform(low=low, high=high)
        altered_qload = q_pows[bus_idx] * generator.uniform(low=low, high=high)
        '''
        seed = random.seed()
        altered_pload = ref_obj.BusList[bus_idx].pload * random.uniform(low, high)
        altered_qload = ref_obj.BusList[bus_idx].qload * random.uniform(low, high)
        '''
        lf_obj.BusList[bus_idx].pload = altered_pload
        lf_obj.BusList[bus_idx].qload = altered_qload
        if bus_idx > 0: #for all samples other than slack bus, store loads in array
            #input_sample[bus_idx - 1], input_sample[(bus_idx - 1) * 2] = altered_pload, altered_qload
            input_sample[bus_idx - 1], input_sample[(bus_idx - 1) + len(lf_obj.BusList)-1] = altered_pload, altered_qload
                                                        #NB!!! cannot simply multiply by two
    lf_obj.vomag = np.ones(len(lf_obj.vomag), dtype=float) #remove NaNs
    lf_obj.voang = np.zeros(len(lf_obj.voang), dtype=float)#remove Nans
    lf_obj.DistLF(accuracy)
    output_sample[:len(lf_obj.BusList)-1] = lf_obj.vomag[1:len(lf_obj.BusList)]
    output_sample[len(lf_obj.BusList)-1:] = lf_obj.voang[1:len(lf_obj.BusList)]

    return input_sample, output_sample

def gen_dataset(lf, nr_of_samples=60000, path_to_storage_folder='NO_PATH_PROVIDED', name_prefix=''):
    p_s, q_s = gen_p_q_tuple(lf.BusList)
    i1, o1 = gen_single_set(lf, p_s, q_s)
    inputs = np.zeros((nr_of_samples, len(i1)), dtype=float)
    outputs = np.zeros((nr_of_samples, len(i1)), dtype=float)

    start = t.perf_counter()
    '''
    for sample in range(nr_of_samples): #change to while loop to generate a fixed number of samples?
        try:
            inputs[sample], outputs[sample] = gen_single_set(ref, lf)
        except StopIteration:
            inputs = np.delete(inputs, sample, axis=0)
            outputs = np.delete(outputs, sample, axis=0)
        if sample % 5000 == 0:
            print(f'{sample} samples finished in a total of {t.perf_counter()-start} seconds')'''
    sample = 0
    tries = 0
    convergence_failures = 0
    while sample < nr_of_samples:  # change to while loop to generate a fixed number of samples?
        try:
            inputs[sample], outputs[sample] = gen_single_set(lf, p_s, q_s, low=0.8, high=1.2)
            sample += 1
        except StopIteration:
            #inputs = np.delete(inputs, sample, axis=0)ll
            #outputs = np.delete(outputs, sample, axis=0)
            convergence_failures += 1
            print(f'total convergence failures: {convergence_failures}')
            sample -= 1 #Go back, try a new randomly generated sample.
        finally:
            tries += 1
            if sample % 5000 == 0:
                print(f'{sample} samples finished in a total of {t.perf_counter() - start} seconds')
            if tries > 3 * nr_of_samples: # Break out of the loop if
                print('Tried more than three times the amount of samples required. Outputs likely not valid.')
                break
    store(inputs, path=path_to_storage_folder, filename=name_prefix + '_inputs')
    store(outputs, path=path_to_storage_folder, filename=name_prefix + '_outputs')
    return convergence_failures

system_description_folder_large_sys= '/home/clemens/Dropbox/EMIL_MIENERG21/Master/IEEE33bus_69bus/'
medium_sys_filename = 'IEEE33BusDSAL.xls'
large_sys_filename = 'IEEE69BusDSAL.xls'
path_storage_folder = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/datasets/'
filename_medium = 'medium_dataset'
filename_large = 'large_dataset'

m_dlf_buses, m_dlf_lines = BuildSystem3(system_description_folder_large_sys + medium_sys_filename)
l_dlf_buses, l_dlf_lines = BuildSystem3(system_description_folder_large_sys + large_sys_filename)


solution_object = dlf(m_dlf_buses, m_dlf_lines)
#solution_object = dlf(l_dlf_buses, l_dlf_lines)

solution_object.initialize(startBus=1)

gen_dataset(solution_object,
            nr_of_samples=60000,
            path_to_storage_folder=path_storage_folder,
            name_prefix='medium')

inputs = load(path=path_storage_folder, filename='medium_inputs.obj')
outputs = load(path=path_storage_folder, filename='medium_outputs.obj')

'''
accuracy = 0.00001
single_sample_gen_start = t.perf_counter()
r_in_1, r_o_1 = gen_single_set(reference_object, solution_object, accuracy=accuracy)
r_in_2, r_o_2 = gen_single_set(reference_object, solution_object, accuracy=accuracy)
single_sample_end = (t.perf_counter()-single_sample_gen_start) / 2

difference_between_solutions = r_o_1 - r_o_2
'''

'''
m_buses, m_lines = BuildSystem(system_description_folder_large_sys + medium_sys_filename)
l_buses, l_lines = BuildSystem(system_description_folder_large_sys + large_sys_filename)
m_dlf_buses, m_dlf_lines = BuildSystem3(system_description_folder_large_sys + medium_sys_filename)
l_dlf_buses, l_dlf_lines = BuildSystem3(system_description_folder_large_sys + large_sys_filename)

check_line_impedances = np.ones((2, len(l_lines)), dtype=int)
#for line_idx in l_lines:


m_lf = LoadFlow(m_buses, m_lines)
l_lf = LoadFlow(l_buses, l_lines)
m_dlf = dlf(m_dlf_buses, m_dlf_lines)
l_dlf = dlf(l_dlf_buses, l_dlf_lines)

for dlf in [m_dlf, l_dlf]:
    dlf.initialize(startBus=1)

epsilon = 0.00001
m_lf.solve_NR(tolerance=epsilon)
l_lf.solve_NR(tolerance=epsilon)
m_dlf.DistLF(epsilon=epsilon)
l_dlf.DistLF(epsilon=epsilon)

m_vomag_similarity = check_similarity(m_dlf.vomag, m_lf.vomag)
m_voang_similarity = check_similarity(m_dlf.voang, m_lf.voang)

l_vomag_similarity = check_similarity(l_dlf.vomag, l_lf.vomag)
l_voang_similarity = check_similarity(l_dlf.voang, l_lf.voang)

l_buses = np.arange(0, len(l_voang_similarity))
m_buses = np.arange(0, len(m_voang_similarity))

plt.plot(l_buses, l_voang_similarity, label='voang_sim')
plt.plot(l_buses, l_vomag_similarity, label='vomag_sim')
plt.plot(m_buses, m_voang_similarity, label='voang_sim')
plt.plot(m_buses, m_vomag_similarity, label='vomag_sim')

plt.legend()
plt.show()

'''






