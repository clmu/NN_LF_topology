import numpy as np
from Neural_network.nn_functions import verification_predictions, gen_model

def eval_model_performance(nr_results, model_results, path=None):

    '''
    function calculates the model performance by calculating the percentage error of each output.
    note: this function could likely be implemented as error function in the model itself.
    :param nr_results: filename or variable of the results from the nr verification file.
    :param model_results: filename or variable of the file containing the model prediction data.
    :param results_path: path containg the two files above.
    :return: array containing the percentage errors of all model outputs.
    '''

    if isinstance(nr_results, str) and isinstance(model_results, str):
        if path is not None:
            nr_results = path + nr_results
            model_results = path + model_results
            nr_results = np.load(nr_results)
            model_results = np.load(model_results)

    samples, outputs = np.shape(nr_results)

    percentage_error_matrix = np.zeros((samples, outputs), dtype=float)
    for i in range(samples):
        percentage_error_matrix[i] = (model_results[i] - nr_results[i]) / nr_results[i] * 100

    percentage_error_matrix = np.abs(percentage_error_matrix)

    #np.save('percentage_accuracy_matrix.npy', percentage_error_matrix)

    return np.average(percentage_error_matrix, axis=0), np.amax(percentage_error_matrix, axis=0)


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

def load_respective_checkpoints(list_of_models,start_cp, path):

    '''
    function to load different checkpoints into the list of models.
    :param list_of_models: list of models to be generated from checkpoints.
    :param path: path to checkpoint files.
    :return: none. Alters the list itself??
    '''
    file_nr = np.arange(start_cp, (start_cp + len(list_of_models)*10), step=10)
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

def evaluate_checkpoints(actual_solutions, predictions):

    '''
    Function to evaluate all predicitons using the actual solutions
    :param predictions: List of prediction matrices generated from different cps.
    :param actual_solutions: actual solution.
    :return: list of lists containing the accuracies of all datapoints.
    '''
    accuracies = []
    worst_accuracies = []
    for prediction in predictions:
        avg_accuracy, worst_accuracy = eval_model_performance(actual_solutions, prediction)
        accuracies.append(avg_accuracy)
        worst_accuracies.append(worst_accuracy)
    return [accuracies, worst_accuracies]
