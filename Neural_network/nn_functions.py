import numpy as np
import time
import tensorflow as tf

def verification_predictions(verification_input_data, model):

    '''
    function to calculate predictions for all verification input data.
    :param verification_input_data: the input data for calculating predictions.
    :param model: Model to be used for calculating the predictions.
    :return: matrix  ??  of all predictions and average prediction time.
    '''
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

def import_data(inputfilename, outputfilename, datapath='', verification_fraction=0.2, norm_input=1, norm_output=1):

    '''
    Function to import data for generating neural networks.
    :param inputfilename:
    :param outputfilename:
    :param datapath:
    :param verification_fraction:
    :param norm_input:
    :param norm_output:
    :return:
    '''

    inputdata = np.load(datapath + inputfilename)
    outputdata = np.load(datapath + outputfilename)

    nr_samples, nr_input_var = np.shape(inputdata)
    verification_samples = int(nr_samples // (1/verification_fraction))
    learning_samples = int(nr_samples - verification_samples)
    (l_in, l_out) = inputdata[:learning_samples] / norm_input, outputdata[:learning_samples] * norm_output
    (v_in, v_out) = inputdata[learning_samples:] / norm_input, outputdata[learning_samples:] * norm_output
    return (l_in, l_out), (v_in, v_out)

def gen_model():
    '''
    Function to generate a neural network. Useful to make sure that all networks have the same architecture.
    :return:
    '''
    nn = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(6,)),
        # tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(6)
    ])

    loss_fn = tf.keras.losses.MeanSquaredError()

    nn.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    return nn