import tensorflow as tf
import numpy as np
from Neural_network.nn_functions import import_data, verification_predictions, gen_model
from Neural_network.data_evaluation import eval_model_performance

def gen_list_of_models(num_of_models):
    '''
    Function takes in a number, and makes a list containing this many models.
    :param num_of_models: number of models (int)
    :return: list of models.
    '''

    list_of_models = []
    for i in range(num_of_models):
        list_of_models.append(gen_model())

    return list_of_models

def gen_nr_string(number, digits):

    '''
    function generates a digit string for use in loading checkpoints.
    :param number: number to be converted.
    :param digits: digits in the digit string.
    :return:digit string
    '''
    number = str(number)
    while len(number) < digits:
        number = '0' + number
    return number

def load_respective_checkpoints(list_of_models, path):

    '''
    function to load different checkpoints into the list of models.
    :param list_of_models: list of models to be generated from checkpoints.
    :param path: path to checkpoint files.
    :return: none. Alters the list itself??
    '''
    file_nr = np.arange(10, (len(list_of_models) + 1)*10, step=10)
    for i in range(len(list_of_models)):
        filename = 'cp_' + gen_nr_string(file_nr[i], 4) + '.ckpt'
        list_of_models[i].load_weights(path + filename)
    pass

def predict(list_of_models, input_data):

    '''
    function calculates all NN predictions for the input data.
    :param list_of_models: the list of models to predict.
    :param input_data: the input data to the models.
    :return: list of predictions and average runtimes of all predictions.
    '''
    prediction_list = []
    time_list = []
    for model in list_of_models:
        pred, time = verification_predictions(input_data, model)
        prediction_list.append(pred)
        time_list.append(time)
    return prediction_list, time_list

def evaluate_checkpoints(predictions, actual_solutions):

    '''
    Function to evaluate all predicitons using the actual soltions
    :param predictions: List of prediction matrixes generated from different cps.
    :param actual_solutions: actual solution.
    :return: list of lists containing the accuracies of all datapoints.
    '''
    accuracies = []
    for prediction in predictions:
        accuracies.append(eval_model_performance(prediction, actual_solutions))
    return accuracies


datapath = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
input = np.load(datapath + 'simple data.npy')
output = np.load(datapath + 'simple o data.npy')

norm_inputs = 2
norm_outputs = 10


(l_i, l_o), (v_i, v_o) = import_data('simple data.npy', 'simple o data.npy', datapath=datapath, norm_input=norm_inputs, norm_output=norm_outputs)

number_of_models = 15

models = gen_list_of_models(number_of_models)

cp_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/checkpoints/'

load_respective_checkpoints(models, cp_path)

list_of_predictions, list_of_prediction_times = predict(models, v_i)

evaluate_checkpoints(list_of_predictions, v_o)

print('finito')