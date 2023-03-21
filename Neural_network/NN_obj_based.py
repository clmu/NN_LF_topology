'''

This is the master script for training and evaluating a neural network from the object oriented approach.

Custom loss functions are found in Neural_network.custom_loss_function. if others are to be used, refer to keras doc.


'''

import os
import time

import tensorflow as tf
import numpy as np
import pickle
import math

from Neural_network.NN_objects import NeuralNetwork as NN
from Neural_network.custom_loss_function import loss_acc_for_lineflows,\
    CustomLoss, SquaredLineFlowLoss, LineFlowLossForAngle

#norm_inputs = 2 #value to ensure all inputs are between 0 and 1.
#norm_outputs = 10 #value to make outputs greater to increase performance of meanSquaredError

# PATHS to containers
path_to_data = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
path_to_system_description_file = '/home/clemens/PycharmProjects/NN_LF_Topology/LF_3bus/4 bus 1 gen.xls'
cp_path_custom_loss = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/cp_small_CustomLoss/cp_{epoch:04d}'
cp_path_square_loss = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/cp_small_SquaredLineFlowLoss/cp_{epoch:04d}'
cp_path_angle_loss = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/cp_small_LineFlowLossForAngle/cp_{epoch:04d}'

nn_custom_loss = NN()

# loss functions # NB: depend on output normalizer from NN, do not change this later on.
custom_loss = CustomLoss(path=path_to_system_description_file, o_norm=nn_custom_loss.get_norm_output())
custom_square_loss = SquaredLineFlowLoss(path=path_to_system_description_file, o_norm=nn_custom_loss.get_norm_output())
only_angle_loss = LineFlowLossForAngle(path=path_to_system_description_file, o_norm=nn_custom_loss.get_norm_output())

# NN parameters
nn_custom_loss.epochs = 30
nn_custom_loss.batch_size = 20
nn_custom_loss.initializer = tf.keras.initializers.glorot_uniform(seed=0) #THIS IS THE SAME AS USED IN NON OBJ BASED APPROACH.
nn_custom_loss.init_data('simple data.npy',
                 'simple o data.npy',
                 0.2,
                 datapath=path_to_data,
                 scale_data_out=True)
nn_square_loss = nn_custom_loss
nn_angle_loss = nn_custom_loss
nn_custom_loss.loss_fn = custom_loss
nn_square_loss.loss_fn = custom_square_loss
nn_angle_loss.loss_fn = only_angle_loss

nn_custom_loss.init_nn_model_dynamic(architecture=[6, 12, 12, 12, 6], const_l_rate=True)
nn_square_loss.init_nn_model_dynamic(architecture=[6, 12, 12, 12, 6], const_l_rate=True)


nn_custom_loss.train_model(checkpoints=True,
                           cp_folder_path=cp_path_custom_loss,
                           save_freq=120*nn_custom_loss.batch_size)

nn_square_loss.train_model(checkpoints=True,
                           cp_folder_path=cp_path_square_loss,
                           save_freq=120*nn_custom_loss.batch_size)

nn_square_loss.train_model(checkpoints=True,
                           cp_folder_path=cp_path_angle_loss,
                           save_freq=120*nn_custom_loss.batch_size)

'''
Data eval

nn_custom_loss.model_pred()
thresholds = [20, 10, 5, 2]
for threshold in thresholds:
    nn_custom_loss.generate_performance_data_dict(threshold=threshold)
'''

'''
Store nn_obj_final_perfomance_dict


filename = 'pinn'
f = open('/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/NN_objects/' + filename + '.obj', 'wb')
pickle.dump(nn_obj, f)
f.close()
'''

'''test_sample = np.array([0.2, 0.4, 1.25, 0.07, 0.5, 0.8]) / nn_obj.norm_input
pred1 = nn_obj.single_prediction(test_sample)
print(pred1)'''