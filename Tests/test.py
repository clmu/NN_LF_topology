import numpy as np
from Neural_network.nn_functions import import_data, verification_predictions, gen_model
from Neural_network.data_evaluation import eval_model_performance

datapath = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
cp_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/checkpoints/'
input = np.load(datapath + 'simple data.npy')
output = np.load(datapath + 'simple o data.npy')

norm_inputs = 2
norm_outputs = 10


(l_i, l_o), (v_i, v_o) = import_data('simple data.npy', 'simple o data.npy', datapath=datapath, norm_input=norm_inputs, norm_output=norm_outputs)

cp15 = gen_model()
cp9 = gen_model()
cp15.load_weights(cp_path + 'cp_0150.ckpt')
cp9.load_weights(cp_path + 'cp_0090.ckpt')

predictions_15, times_15 = verification_predictions(v_i, cp15)
predictions_9, times_9 = verification_predictions(v_i, cp9)

accuracy_cp15 = np.array(eval_model_performance(v_o, predictions_15))
accuracy_cp9 = np.array(eval_model_performance(v_o, predictions_9))