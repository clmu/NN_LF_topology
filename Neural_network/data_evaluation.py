import numpy as np

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

    return np.average(percentage_error_matrix, axis=0)

avg_nr_runtime    = 0.00165041
avg_model_runtime = 0.00166583

results_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/NN_results/'

output_performance = eval_model_performance('verification_outputs.npy', 'model_predictions.npy', path=results_path)
abs_output_performance = np.abs(output_performance)
avg_output_performance = np.average(abs_output_performance)
