import os
import time

import tensorflow as tf
import numpy as np
from Neural_network.data_evaluation import eval_model_performance as evaluate
from Neural_network.nn_functions import verification_predictions, import_data, gen_model

#print('TensorFlow version: ', tf.__version__) #to verify correct installation, seems to be OK.

#path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'


norm_inputs = 2 #value to ensure all inputs are between 0 and 1.
norm_outputs = 10 #value to make outputs greater to increase performance of meanSquaredError

small_dataset = False
about = '.ckpt'
if small_dataset:
    small = '_small'
else:
    small = ''

'''
Loading and sorting the data for model training.
'''



datapath = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
input = np.load(datapath + 'simple data.npy')
output = np.load(datapath + 'simple o data.npy')

(l_i, l_o), (v_i, v_o) = import_data('simple data.npy', 'simple o data.npy', datapath=datapath, norm_input=norm_inputs,
                                     norm_output=norm_outputs)


'''
creating, and describing the NN model. 
'''

'''
Training and evaluating the model. Also savinv the progress.

Epoch: number of times the whole training set is worked through
batch: number of samples worked through before the weights are adjusted. 
'''

model = gen_model()

batch_size = 20
epochs = 150

cp_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/checkpoints/cp_{epoch:04d}' + about
cp_dir = os.path.dirname(cp_path)

model.summary()

#create callback to save model weights.
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=1200*batch_size)
model.fit(l_i,
          l_o,
          epochs=epochs,
          batch_size=batch_size,
          callbacks=[cp_callback],
          verbose=1)

'''
loss, acc = model.evaluate(v_i, v_o)

print('model accuracy: {:5.2f}%'.format(100*acc))


test_4_bus_1_gen = np.array([[0.02, 0.04, 0.125, 0.007, 0.05, 0.08]])

result_standard_test = model(test_4_bus_1_gen)

array_result_standard_test = np.array(result_standard_test)
'''

'''
to evaluate results
'''

verification_predictions, avg_model_runtime = verification_predictions(v_i, model)

results_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/NN_results/'
#path, variable

model_predict_name = 'model_predictions.npy'
nr_results_name = 'verification_outputs.npy'

np.save(results_path + model_predict_name, verification_predictions)
np.save(results_path + nr_results_name, v_o)



'''
evaluating model performance by calculating the average accuracy of each prediction.
'''

output_var_percentage_accuracy = evaluate(nr_results_name, model_predict_name, path=results_path)
overall_mean_accuracy = np.average(np.abs(output_var_percentage_accuracy[0]))