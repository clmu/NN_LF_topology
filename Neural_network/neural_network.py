import os
import time

import tensorflow as tf
import numpy as np
from Neural_network.data_evaluation import eval_model_performance as evaluate

#print('TensorFlow version: ', tf.__version__) #to verify correct installation, seems to be OK.

def verification_predictions(verification_input_data):


    v_samples, v_variables = np.shape(verification_input_data)
    model_prediction_time = np.zeros((v_samples,))
    model_predictions = np.zeros((v_samples, v_variables))
    counter = 0
    for verification_row in verification_input_data:
        model_input = np.array([verification_row])
        model_start_time = time.perf_counter()
        model_predictions[counter] = model(model_input).numpy()
        model_prediction_time[counter] = time.perf_counter() - model_start_time
        counter += 1

    avg_model_prediction_time = np.average(model_prediction_time)
    return model_predictions, avg_model_prediction_time

#path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'


norm_inputs = 2 #value to ensure all inputs are between 0 and 1.
norm_outputs = 10 #value to make outputs greater to increase performance of meanSquaredError

small_dataset = False
about = 'large_sq_err.ckpt'
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

samples, input_variables = np.shape(input)
fraction_for_verification = 0.2
verification_samples = int(samples // (1/fraction_for_verification))
learning_samples = int(samples - verification_samples)

(l_i, l_o) = input[:learning_samples]/norm_inputs, output[:learning_samples]*norm_outputs
(v_i, v_o) = input[learning_samples:]/norm_inputs, output[learning_samples:]*norm_outputs


'''
creating, and describing the NN model. 
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(6,)),
    #tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(6)
])

'''
Notes on error function:
mean square error may be a bad alternatrive since outputs often are smaller than 1. 
'''
loss_fn = tf.keras.losses.MeanSquaredError()
#loss_fn = tf.keras.losses.MeanAbsoluteError()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
'''
Training and evaluating the model.

Epoch: number of times the whole training set is worked through
batch: number of samples worked through before the weights are adjusted. 
'''
model.fit(l_i, l_o, epochs=150, batch_size=20)

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

verification_predictions, avg_model_runtime = verification_predictions(v_i)

results_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/NN_results/'
#path, variable

model_predict_name = 'model_predictions.npy'
nr_results_name = 'verification_outputs.npy'

np.save(results_path + model_predict_name, verification_predictions)
np.save(results_path + nr_results_name, v_o)



cp_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/checkpoints/cp_' + about
cp_dir = os.path.dirname(cp_path)



#create callback to save model weights.
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path,
                                                 save_weights_only=True,
                                                 verbose=1)

'''
evaluating model performance by calculating the average accuracy of each prediction.
'''

output_var_percentage_accuracy = evaluate(nr_results_name, model_predict_name, path=results_path)
overall_mean_accuracy = np.average(output_var_percentage_accuracy)