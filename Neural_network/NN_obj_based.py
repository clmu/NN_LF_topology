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
from Neural_network.NN_objects import pickle_store_object as store

def load_loss_function(loss_fun_name, path_to_sys_file=''):
    if loss_fun_name == 'MSE':
        return tf.keras.losses.mean_squared_error
    elif loss_fun_name == 'ME':
        return tf.keras.losses.mean_absolute_error
    elif loss_fun_name == 'CustomLoss':
        return CustomLoss(path=path_to_sys_file)
    elif loss_fun_name == 'SquaredLineFlowLoss':
        return SquaredLineFlowLoss(path=path_to_sys_file)
    elif loss_fun_name == 'LineFlowLossForAngle':
        return LineFlowLossForAngle(path=path_to_sys_file)

def load_architecture(network_name):
    if network_name == 'small_':
        return [6, 12, 12, 12, 6]
    elif network_name == 'medium_':
        return [64, 128, 128, 128, 64]
    elif network_name == 'large_':
        return [136, 272, 272, 272, 136]

def set_params_and_init_nn(model, data_in_name='', data_out_name='', pickle_load=False):
    # NN parameters
    model.epochs = 30
    model.batch_size = 20
    model.initializer = tf.keras.initializers.glorot_uniform(seed=0)
                                            # THIS IS THE SAME AS USED IN NON OBJ BASED APPROACH.
    model.init_data(data_in_name,
                    data_out_name,
                    0.2,
                    datapath=path_to_data,
                    scale_data_out=True,
                    pickle_load=pickle_load)
    model.init_nn_model_dynamic(architecture=model.architecture, const_l_rate=True, custom_loss=True)
    pass

'''# PATHS to containers for small network
path_to_data = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
path_to_nn_folder = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
path_to_system_description_file = '/home/clemens/PycharmProjects/NN_LF_Topology/LF_3bus/4 bus 1 gen.xls'
cp_path_MSE = 'checkpoints/cp_small_MSE_short/cp_{epoch:04d}'
cp_path_custom_loss = path_to_nn_folder + 'checkpoints/cp_small_CustomLoss/cp_{epoch:04d}'
cp_path_square_loss = path_to_nn_folder + 'checkpoints/cp_small_SquaredLineFlowLoss/cp_{epoch:04d}'
cp_path_angle_loss = path_to_nn_folder + 'checkpoints/cp_small_LineFlowLossForAngle/cp_{epoch:04d}'
nn_regular_mse = NN()
nn_custom_loss = NN()
nn_square_loss = NN()
nn_angle_loss = NN()
list_of_nn_objs = [nn_regular_mse, nn_custom_loss, nn_square_loss, nn_angle_loss]

architecture = [6, 12, 12, 12, 6]'''

nn_obj = NN()

path_to_system_description_file = '/home/clemens/PycharmProjects/NN_LF_Topology/LF_3bus/' #'/home/clemens/Dropbox/EMIL_MIENERG21/Master/IEEE33bus_69bus/IEEE33BusDSAL.xls'
path_to_system_description_file += 'IEEE33BusDSAL.xls'
path_to_data = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/datasets/'
network_name = 'medium' + '_'
#network_name = 'large'
network_loss_function = 'LineFlowLossForAngle' #CustomLoss, SquaredLineFlowLoss, LineFlowLossForAngle
input_data_name = network_name + 'inputs.obj'
output_data_name = network_name + 'outputs.obj'
cp_folder_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/checkpoints/cp_' + \
                        network_name + network_loss_function #+ '/next30'
cp_folder_and_name = cp_folder_path + '/cp_{epoch:04d}'
arch = load_architecture(network_name)
loss = load_loss_function(network_loss_function, path_to_sys_file=path_to_system_description_file)
nn_obj.architecture = arch
nn_obj.loss_fn = loss
set_params_and_init_nn(nn_obj, data_in_name=input_data_name, data_out_name=output_data_name, pickle_load=True)


'''nn_regular_mse = NN()
nn_custom_loss = NN()
nn_square_loss = NN()
nn_angle_loss = NN()
list_of_nn_objs = [nn_regular_mse, nn_custom_loss, nn_square_loss, nn_angle_loss]'''


# loss functions # NB: depend on output normalizer from NN, do not change this later on.
'''custom_loss = CustomLoss(path=path_to_system_description_file, o_norm=nn_custom_loss.get_norm_output())
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
    model.init_nn_model_dynamic(architecture=architecture, const_l_rate=True, custom_loss=False)'''

text  = 'This is the testfile.'

with open(cp_folder_path + 'textfile.txt', 'w') as f:
    f.write(text)

nn_obj.train_model(checkpoints=True,  cp_folder_path=cp_folder_and_name, save_freq=120*nn_obj.batch_size)
thresholds = [20, 10, 5, 3]
nn_obj.model_pred()
nn_obj.generate_performance_data_dict_improved(thresholds)
perf_dict_name = 'performance_epoch_' + str(nn_obj.epochs)
store(nn_obj.performance_dict, path=cp_folder_path, filename=perf_dict_name)

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