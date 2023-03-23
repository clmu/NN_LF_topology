'''

This is the master script for training and evaluating a neural network from the object oriented approach.

Custom loss functions are found in Neural_network.custom_loss_function. if others are to be used, refer to keras doc.


'''

import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #TO SILCENCE INITIAL WARNINGS.
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
path_to_nn_folder = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
path_to_system_description_file = '/home/clemens/PycharmProjects/NN_LF_Topology/LF_3bus/4 bus 1 gen.xls'
cp_path_MSE = 'cp_small_MSE_short/cp_{epoch:04d}'
cp_path_custom_loss = path_to_nn_folder + 'cp_small_CustomLoss/cp_{epoch:04d}'
cp_path_square_loss = path_to_nn_folder + 'cp_small_SquaredLineFlowLoss/cp_{epoch:04d}'
cp_path_angle_loss = path_to_nn_folder + 'cp_small_LineFlowLossForAngle/cp_{epoch:04d}'
nn_regular_mse = NN()
nn_custom_loss = NN()
nn_square_loss = NN()
nn_angle_loss = NN()
list_of_nn_objs = [nn_regular_mse, nn_custom_loss, nn_square_loss, nn_angle_loss]

architecture = [6, 12, 12, 12, 6]

# loss functions # NB: depend on output normalizer from NN, do not change this later on.
custom_loss = CustomLoss(path=path_to_system_description_file, o_norm=nn_custom_loss.get_norm_output())
custom_square_loss = SquaredLineFlowLoss(path=path_to_system_description_file, o_norm=nn_custom_loss.get_norm_output())
only_angle_loss = LineFlowLossForAngle(path=path_to_system_description_file, o_norm=nn_custom_loss.get_norm_output())

nn_regular_mse.loss_fn = tf.keras.losses.MSE
nn_custom_loss.loss_fn = custom_loss
nn_square_loss.loss_fn = custom_square_loss
nn_angle_loss.loss_fn = only_angle_loss

for model in list_of_nn_objs:
    # NN parameters
    model.epochs = 30
    model.batch_size = 20
    model.initializer = tf.keras.initializers.glorot_uniform(seed=0) #THIS IS THE SAME AS USED IN NON OBJ BASED APPROACH.
    model.init_data('simple data.npy',
                     'simple o data.npy',
                     0.2,
                     datapath=path_to_data,
                     scale_data_out=True)
    model.init_nn_model_dynamic(architecture=architecture, const_l_rate=True, custom_loss=False)


nn_regular_mse.train_model(checkpoints=True,  cp_folder_path=cp_path_MSE, save_freq=120*nn_custom_loss.batch_size)


'''
nn_custom_loss.train_model(checkpoints=False,
                           cp_folder_path=cp_path_custom_loss,
                           save_freq=120*nn_custom_loss.batch_size)


nn_square_loss.train_model(checkpoints=True,
                           cp_folder_path=cp_path_square_loss,
                           save_freq=120*nn_custom_loss.batch_size)

nn_angle_loss.train_model(checkpoints=True,
                           cp_folder_path=cp_path_angle_loss,
                           save_freq=120*nn_custom_loss.batch_size)
'''

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