import os
import time

import tensorflow as tf
import numpy as np

from Neural_network.NN_objects import NeuralNetwork as NN

nn_obj = NN()
path_to_data = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
nn_obj.init_data('simple data.npy', 'simple o data.npy', 0.2, datapath=path_to_data)

nn_obj.loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
#nn_obj.loss_fn = tf.keras.losses.MeanSquaredError()
nn_obj.initializer = tf.keras.initializers.random_normal(mean=0.5, stddev=0.01)
nn_obj.init_nn_model()

cp_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/checkpoints/cp_{epoch:04d}'

nn_obj.epochs=7
nn_obj.train_model(checkpoints=True, cp_folder_path=cp_path)

#nn_obj.load_latest_pretrained_model(cp_path)
#nn_obj.tf_model.load_weights(cp_path+'cp_0050.index')

pred = nn_obj.single_prediction([0.02, 0.04, 0.125, 0.007, 0.05, 0.08])

nn_obj.model_pred()

nn_obj.eval_model_performance()

