"""
This script is now obsolete, it is replaced by NN_function.py

This is the master script for training and evaluating a neural network from the object oriented approach.

Custom loss functions are found in Neural_network.custom_loss_function. if others are to be used, refer to keras doc.


"""
import os
import tensorflow as tf
from pathlib import Path
from Neural_network.NN_objects import NeuralNetwork
from Neural_network.custom_loss_function import loss_acc_for_lineflows,\
    CustomLoss, SquaredLineFlowLoss, LineFlowLossForAngle
from Neural_network.NN_objects import pickle_store_object as store
from Neural_network.NN_objects import load_architecture
from Neural_network.data_evaluation import eval_nn_obj_epochs, eval_nn_obj_epochs_list
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #TO SILCENCE INITIAL WARNINGS.

'''
#multithreading CPU
threads = '48' #cores at specific idun node

os.environ['MKL_NUM_THREADS'] = threads
os.environ['GOTO_NUM_THREADS'] = threads
os.environ['OMP_NUM_THREADS'] = threads
os.environ['openmp'] = 'True' '''


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

def set_params_and_init_nn(model, data_in_name='', data_out_name='', path='NONE', pickle_load=True, mse_flag=False):
    # NN parameters
    model.initializer = tf.keras.initializers.glorot_uniform(seed=0)
                                            # THIS IS THE SAME AS USED IN NON OBJ BASED APPROACH.
    model.init_data(data_in_name,
                    data_out_name,
                    0.2,
                    datapath=path,
                    scale_data_out=True,
                    pickle_load=pickle_load)
    model.init_nn_model_dynamic(architecture=model.architecture, const_l_rate=True, custom_loss=not mse_flag)
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



#proj_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) #cwd = current working dir, pardir = parent dir
proj_folder = os.getcwd()
path_sys_file = proj_folder + '/LF_3bus' + '/IEEE69BusDSAL.xls'
path_data = proj_folder + '/datasets/'
path_to_system_description_file = proj_folder + '/LF_3bus/'#'/home/clemens/PycharmProjects/NN_LF_Topology/LF_3bus/' #'/home/clemens/Dropbox/EMIL_MIENERG21/Master/IEEE33bus_69bus/IEEE33BusDSAL.xls'
sys_filename = 'IEEE69BusDSAL.xls'
path_to_data = proj_folder + '/datasets/'#'/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/datasets/'
path_to_cp_folder = proj_folder + '/checkpoints'
folder_hierarchy = {}

'''Selecting network size and Hyperparameters'''
dataset = 'slim' #type slim if slim dataset is desired.
network_name = 'large'
arch = load_architecture(network_name)
remark = 'baseline_slim' # '_learn1e-4_batch10'
l_rate = 1e-4
batch_size = 20
epochs = 150
thresholds = [20, 10, 5, 3]
#network_loss_function = 'MSE' #CustomLoss, SquaredLineFlowLoss, LineFlowLossForAngle
loss_function_list = ['MSE']#, 'CustomLoss']

if dataset != '':
    dataset = '_' + dataset

input_data_name = network_name + dataset + '_inputs.obj'
output_data_name = network_name + dataset +'_outputs.obj'


folder_hierarchy['sys_descriptions'] = {}
folder_hierarchy['checkpoints'] = {}

folder_hierarchy['sys_descriptions']['container'] = path_to_system_description_file
folder_hierarchy['sys_descriptions']['system_path'] = path_to_system_description_file + sys_filename
folder_hierarchy['data_folder'] = path_data
folder_hierarchy['checkpoints']['main_folder'] = path_to_cp_folder


for loss_fun in loss_function_list:

    loss = load_loss_function(loss_fun, path_to_sys_file=folder_hierarchy['sys_descriptions']['system_path'])
    nn_obj = NeuralNetwork()
    nn_obj.l_rate = l_rate
    #arch = [64, 128, 128, 128, 128, 128, 64] # adding additional layer of neurons
    nn_obj.architecture = arch

    #print(f'nn_obj arch: {nn_obj.architecture}')
    nn_obj.loss_fn = loss
    
    folder_hierarchy['checkpoints']['model_folder_path'] = path_to_cp_folder + '/' + network_name + '/' \
                                                      + loss_fun + '_' + remark + '/'
    folder_hierarchy['checkpoints']['model_folder'] = network_name + '/' + loss_fun + '_' + remark + '/'
    folder_hierarchy['checkpoints']['model_storage_path'] = folder_hierarchy['checkpoints']['model_folder_path'] + \
                                                            'cp_{epoch:04d}'

    if loss_fun == 'MSE':
        mse = True
    else:
        mse = False

    set_params_and_init_nn(nn_obj,
                           data_in_name=input_data_name,
                           data_out_name=output_data_name,
                           path=path_to_data,
                           pickle_load=True,
                           mse_flag=mse)
    nn_obj.epochs = epochs
    nn_obj.batch_size = batch_size
    path_to_cp = Path(folder_hierarchy['checkpoints']['model_folder_path'])
    path_to_cp.mkdir(exist_ok=True)

    nn_obj.tf_model.summary()
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=folder_hierarchy['checkpoints']['model_folder'],
        verbose=1,
        save_weights_only=True,
        save_freq='epoch')

    nn_obj.train_model(checkpoints=True,
                       #epochs=epochs, batch_size=batch_size,
                       cp_folder_path=folder_hierarchy['checkpoints']['model_storage_path']) #function does not seem to be able to store cps in correct folder.

    '''
    epoch_performance_dictionaries = eval_nn_obj_epochs(nn_obj,
                                                        thresholds=thresholds,
                                                        folder_structure=folder_hierarchy)
    store(epoch_performance_dictionaries,
          path=folder_hierarchy['checkpoints']['model_folder_path'],
          filename='perf_dict')'''

    untrained_nn_list = []
    for epoch in range(nn_obj.epochs):
        nn_model = NeuralNetwork()
        nn_model.architecture = arch
        set_params_and_init_nn(nn_model,
                               data_in_name=input_data_name,
                               data_out_name=output_data_name,
                               path=path_to_data,
                               pickle_load=True)
        untrained_nn_list.append(nn_model)

    epoch_performance_dictionaries_from_list = eval_nn_obj_epochs_list(untrained_nn_list,
                                                                       epochs,
                                                                       thresholds=thresholds,
                                                                       folder_structure=folder_hierarchy)
    store(epoch_performance_dictionaries_from_list,
          path=folder_hierarchy['checkpoints']['model_folder_path'],
          filename='perf_dict_from_list')

    del untrained_nn_list, epoch_performance_dictionaries_from_list

    print(f'batch_size = {nn_obj.batch_size} \nepochs = {nn_obj.epochs} \nl_rate = {nn_obj.l_rate}')

    test = 2


'''set_params_and_init_nn(nn_obj, data_in_name=input_data_name, data_out_name=output_data_name, pickle_load=True)


nn_obj.train_model(checkpoints=True,
                   epochs=epochs, batch_size=batch_size,
                   cp_folder_path=cp_folder_and_name)#,
                   #save_freq=480*nn_obj.batch_size)

nn_obj.model_pred()
nn_obj.generate_performance_data_dict_improved(thresholds)
perf_dict_name = 'performance_epoch_' + str(nn_obj.epochs)
store(nn_obj.performance_dict, path=cp_folder_path, filename=perf_dict_name)'''








































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