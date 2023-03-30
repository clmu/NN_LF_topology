#import tensorflow as tf
#import numpy as np
import time as t

#from Neural_network.NN_objects import NeuralNetwork as NN
from Neural_network.NN_objects import pickle_store_object as store
from Neural_network.data_evaluation import evaluate_cps_obj_new
from Neural_network.NN_objects import load_architecture


path_to_data = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
training_data_folder = 'datasets/'
cp_containing_folder = 'checkpoints/'
#network = 'small'
network = 'medium'
sq = 'SquaredLineFlowLoss'
cl = 'CustomLoss'
la = 'LineFlowLossForAngle'
mse = 'MSE'


architecture = load_architecture(network)
thresholds = [20, 10, 5, 3]
for err_fn in [mse, sq, cl, la]:
    cp_folder = 'cp_' + network + '_' + err_fn + '/'
    start_eval = t.perf_counter()
    print(f'starting_eval of {err_fn}')
    perf_dicts = evaluate_cps_obj_new(30, path_to_data=path_to_data,
                                  cp_folder=cp_folder,
                                  architecture=architecture,
                                  thresholds=thresholds)
    dict_list_calc_time = t.perf_counter()-start_eval
    print(f'finished eval of {err_fn} in {dict_list_calc_time} seconds')
    print(f'storing eval of {err_fn}')

    store(perf_dicts, path=path_to_data+cp_containing_folder+cp_folder, filename='perf_dict')

print('performance dicts generated and stored')

'''
def gen_filename_ext(number):
    number = str(number)
    while len(number) < 4:
        number = '0' + number
    number = 'cp_' + number# + '.index'
    return number

model_numbers = np.arange(1, 30+1)
file_exts = []
models = []
for model_number in model_numbers:
    models.append(NN())
    file_exts.append(gen_filename_ext(model_number))

for model in models:
    model.epochs = 30
    model.batch_size = 20
    model.initializer = tf.keras.initializers.glorot_uniform(seed=0)
    model.init_data('simple data.npy',
                             'simple o data.npy',
                             0.2,
                             datapath=path_to_data,
                             scale_data_out=True)
    model.loss_fn = tf.keras.losses.MSE
    model.init_nn_model_dynamic(architecture=architecture, const_l_rate=True)

thresholds = [20, 10, 5, 3]
avg_perc_dev = np.zeros(len(models), dtype=float)
performance_dicts = []
for index in range(len(models)):
    models[index].tf_model.load_weights(cp_path_custom_loss + file_exts[index])
    models[index].model_pred()
    for threshold in thresholds:
        models[index].generate_performance_data_dict(threshold)
    performance_dicts.append(models[index].performance_dict)
    avg_perc_dev[index] = models[index].performance_dict['3percent']['average']

store(performance_dicts, path=cp_path_custom_loss, filename='performance_dicts')
'''






