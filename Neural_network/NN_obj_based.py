import os
import time

import tensorflow as tf
import numpy as np

from Neural_network.NN_objects import NeuralNetwork as NN
from Neural_network.NN_objects import print_weights

def yang2020_init(shape, dtype=None):
    variance = 2 / (shape[0]*shape[1])
    std_deviation = np.sqrt(variance)
    return tf.keras.initializers.random_normal(mean=0, stddev=std_deviation)


norm_inputs = 2 #value to ensure all inputs are between 0 and 1.
norm_outputs = 10 #value to make outputs greater to increase performance of meanSquaredError


'''
Loading and sorting the data for model training.
'''



datapath = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
input = np.load(datapath + 'simple data.npy')
output = np.load(datapath + 'simple o data.npy')
nn_obj = NN()
path_to_data = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
nn_obj.init_data('simple data.npy',
                 'simple o data.npy',
                 0.2,
                 datapath=path_to_data,
                 scale_data_out=True)

'''
from Neural_network.nn_functions import import_data
(l_i, l_o), (v_i, v_o) = import_data('simple data.npy', 'simple o data.npy', datapath=datapath, norm_input=norm_inputs,
                                     norm_output=norm_outputs)
nn_obj.l_data = l_i
nn_obj.l_sol = l_o
nn_obj.t_data = v_i
nn_obj.t_sol = v_o

datacheck = [True, True, True, True]
data = [[l_i, l_o, v_i, v_o], [nn_obj.l_data, nn_obj.l_sol, nn_obj.t_data, nn_obj.t_sol]]


for i in range(len(datacheck)):
    
    #   If all datacheck values are true after the check, the object based model utilizes
    #   the same dataset as the function based approach. 
    
    if (data[0][i] != data[1][i]).all():
        datacheck[i] = False

'''


#nn_obj.loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
nn_obj.loss_fn = tf.keras.losses.MeanSquaredError()
#nn_obj.loss_fn = tf.keras.losses.MeanAbsoluteError()
#nn_obj.initializer = yang2020_init
#nn_obj.initializer = tf.keras.initializers.random_normal(mean=0, stddev=2)
nn_obj.initializer = tf.keras.initializers.glorot_uniform(seed=0) #THIS IS THE SAME AS USED IN NON OBJ BASED APPROACH.
#nn_obj.initializer = tf.keras.initializers.random_uniform(minval=-3., maxval=3.)
#nn_obj.initializer = tf.keras.initializers.ones()
nn_obj.epochs = 50
#nn_obj.init_nn_model(architecture=[6, 12, 12, 12, 6], const_l_rate=False)
#nn_obj.init_nn_model_fixed()
nn_obj.init_nn_model_dynamic(architecture=[6, 12, 12, 12, 6], const_l_rate=True)

#print_weights(nn_obj.tf_model)

cp_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/checkpoints/cp_{epoch:04d}'
pred1 = nn_obj.single_prediction([0.02, 0.04, 0.125, 0.007, 0.05, 0.08])
print(pred1)

nn_obj.train_model(checkpoints=False, cp_folder_path=cp_path)

#nn_obj.load_latest_pretrained_model(cp_path)
#nn_obj.tf_model.load_weights(cp_path+'cp_0050.index')

pred2 = nn_obj.single_prediction([0.02, 0.04, 0.125, 0.007, 0.05, 0.08])


'''
Data eval
'''
nn_obj.model_pred()

nn_obj.eval_model_performance()

