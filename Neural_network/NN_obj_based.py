import os
import time

import tensorflow as tf
import numpy as np

from Neural_network.NN_objects import NeuralNetwork as NN
from Neural_network.NN_objects import print_weights

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


nn_obj.loss_fn = tf.keras.losses.MeanSquaredError() #alt: MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError
nn_obj.initializer = tf.keras.initializers.glorot_uniform(seed=0) #THIS IS THE SAME AS USED IN NON OBJ BASED APPROACH.
nn_obj.epochs = 10
nn_obj.init_nn_model_dynamic(architecture=[6, 12, 12, 12, 6], const_l_rate=True)

cp_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/checkpoints/cp_{epoch:04d}'
test_sample = np.array([0.2, 0.4, 1.25, 0.07, 0.5, 0.8]) / nn_obj.norm_input
pred1 = nn_obj.single_prediction(test_sample)
print(pred1)

nn_obj.train_model(checkpoints=False, cp_folder_path=cp_path)

#nn_obj.load_latest_pretrained_model(cp_path)
#nn_obj.tf_model.load_weights(cp_path+'cp_0050.index')

pred2 = nn_obj.single_prediction(test_sample)
print(pred2)

'''
Data eval
'''
nn_obj.model_pred()
thresholds = [20, 10, 5, 2]
for threshold in thresholds:
    nn_obj.generate_performance_data_dict(threshold=threshold)

