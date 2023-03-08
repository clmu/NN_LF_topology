import os
import time

import tensorflow as tf
import numpy as np

from Neural_network.NN_objects import NeuralNetwork as NN

nn_obj = NN()
path_to_data = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
nn_obj.init_data('simple data.npy', 'simple o data.npy', 0.2, datapath=path_to_data)

loss_function = tf.keras.losses.MeanSquaredError()

nn_obj.init_nn_model(loss_function)