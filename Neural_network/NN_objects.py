import os
import time

import tensorflow as tf
import numpy as np

architecture = [6, 12, 12, 12, 6]
class NeuralNetwork:
    def __init__(self, structure=None, learning_rate=1e-2):
        self.l_rate = learning_rate
        self.epochs = 150
        self.batch_size = 20
        self.l_data = None
        self.l_sol = None
        self.t_data = None
        self.t_sol = None
        self.norm_input = 2
        if structure is None:
            self.structure = architecture

    def init_data(self, name_data_in, name_data_out, ver_frac, datapath=''):

        '''
        Function to initialize data
        :param name_data_in: name of the input datafile including filename extension
        :param name_data_out: Name of the output datafile, including filename extension
        :param ver_frac:
        :param datapath:
        :return: Pass. Stores data within NN obj.
        '''

        inputdata = np.load(datapath + name_data_in)
        outputdata = np.load(datapath + name_data_out)

        nr_samples, nr_input_var = np.shape(inputdata)
        v_samples = int(nr_samples // (1/ver_frac))
        learn_samples = int(nr_samples - v_samples)
        self.l_data, self.l_sol = inputdata[:learn_samples] / self.norm_input, outputdata[:learn_samples]
        self.t_data, self.t_sol = inputdata[learn_samples:] / self.norm_input, outputdata[learn_samples:]
        pass



