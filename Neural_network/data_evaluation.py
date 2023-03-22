import numpy as np
import tensorflow as tf
import time as t
from Neural_network.nn_functions import verification_predictions, gen_model
from Neural_network.NN_objects import NeuralNetwork as NN

def eval_model_performance(nr_results, model_results, path=None, threshold=100):

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
    args_over_threshold = np.argwhere(percentage_error_matrix > threshold)
    np.save('percentage_accuracy_matrix.npy', percentage_error_matrix)
    np.save('args_over_threshold.npy', args_over_threshold)

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
        avg_accuracy, worst_accuracy = eval_model_performance(actual_solutions, prediction, threshold=3)
        accuracies.append(avg_accuracy)
        worst_accuracies.append(worst_accuracy)
    return [accuracies, worst_accuracies]

'''
def evaluate_cps_obj(cps, path_to_data='nopath', cp_folder='nofolder', architecture=None, thresholds=None):

 
    def gen_filename_ext(number):
        number = str(number)
        while len(number) < 4:
            number = '0' + number
        number = 'cp_' + number  # + '.index'
        return number

    cp_path = path_to_data + cp_folder
    model_numbers = np.arange(1, cps + 1)
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

    #thresholds = [20, 10, 5, 3]
    avg_perc_dev = np.zeros(len(models), dtype=float)
    performance_dicts = []
    for index in range(len(models)):
        models[index].tf_model.load_weights(cp_path + file_exts[index])
        models[index].model_pred()
        for threshold in thresholds:
            models[index].generate_performance_data_dict(threshold)
        performance_dicts.append(models[index].performance_dict)
        avg_perc_dev[index] = models[index].performance_dict['3percent']['average']

    return performance_dicts'''

def evaluate_cps_obj_new(cps, path_to_data='nopath', cp_folder='nofolder', architecture=None, thresholds=None):

    '''
    Function creates a performance dictionary for a given number of cps, using the provided thresholds.
    :param cps: number of checkpoints. Could be changed to a list of numbers of all cps to be evaluated later.
    :param path_to_data: path to model training data.
    :param cp_folder: Path to checkpoint folder. It is assumed that this is a subfolder of path_to_data
    :param architecture: NN architecture. Used to generate a NN obj to load data into.
    :param thresholds: list of integers. represent percentages for data evaluation.
    :return: List of performance dictionaries
    '''
    def gen_filename_ext(number):
        number = str(number)
        while len(number) < 4:
            number = '0' + number
        number = 'cp_' + number  # + '.index'
        return number

    cp_path = path_to_data + cp_folder
    model_numbers = np.arange(1, cps + 1)
    file_exts = []
    models = []
    for model_number in model_numbers:
        models.append(NN())
        file_exts.append(gen_filename_ext(model_number))

    for model in models:
        '''
        model.epochs = 30
        model.batch_size = 20
        model.initializer = tf.keras.initializers.glorot_uniform(seed=0)'''
        model.init_data('simple data.npy',
                        'simple o data.npy',
                        0.2,
                        datapath=path_to_data,
                        scale_data_out=True)
        model.loss_fn = tf.keras.losses.MSE
        model.init_nn_model_dynamic(architecture=architecture, const_l_rate=True)

    #avg_perc_dev = np.zeros(len(models), dtype=float)
    performance_dicts = []
    for index in range(len(models)):
        starttime = t.perf_counter()
        models[index].tf_model.load_weights(cp_path + file_exts[index])
        models[index].model_pred()
        models[index].generate_performance_data_dict_improved(thresholds)
        performance_dicts.append(models[index].performance_dict)
        print(f'Finished epoch {index + 1} in {t.perf_counter()-starttime} seconds')
        #avg_perc_dev[index] = models[index].performance_dict['3percent']['average']

    return performance_dicts

