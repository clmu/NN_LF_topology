import os
import time

import tensorflow as tf
import numpy as np
import pickle
from keras import layers
from LF_3bus.build_sys_from_sheet import BuildSystem
from LF_3bus.ElkLoadFlow import LoadFlow
from tensorflow import keras

from Neural_network.Custom_model import CustomModel
layers = tf.keras.layers

architecture = [6, 12, 12, 12, 6]

def print_weights(tf_model):

    '''
    Function prints all layer weights except the input layer.
    :param tf_model:
    :return: prints all layer weights.
    '''

    for layer in range(len(tf_model.layers)):
        if layer > 0:
            print(f'weights:\n\t{tf_model.layers[layer].get_weights()[0]}')
            print(f'biases:\n\t{tf_model.layers[layer].get_weights()[1]}')

def pickle_store_object(obj, path=None, filename=None):
    '''
    function to store a (NeuralNetwork) object.
    :param obj: object to be stored
    :param path: path to storage container.
    :param filename: filename without extension of file to be stored.
    :return: pass.
    '''
    f = open(path + filename + '.obj', 'wb')
    pickle.dump(obj, f)
    f.close()
    pass

def pickle_load_obj(path='', filename=''):
    '''
    function to load object
    :param path: path to objec containing folder.
    :param filename: Filename including extension in folde.
    :return: object to be loaded.
    '''
    #f = codecs.open(path + filename, 'r', encoding='utf8')
    #file = f.read()
    #obj = pickle.load(file)
    f = open(path + filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

def load_architecture(network_name):
    if network_name == 'small':
        return [6, 12, 12, 12, 6]
    elif network_name == 'medium':
        return [64, 128, 128, 128, 64]
    elif network_name == 'large':
        return [136, 272, 272, 272, 136]
class NeuralNetwork:
    def __init__(self):
        self.l_rate = None #set in function init_nn_model
        self.epochs = 150
        self.batch_size = 20
        self.l_data = None
        self.l_sol = None
        self.t_data = None
        self.t_sol = None
        self.model_sol = None
        self.tf_model = None
        self.loss_fn = None
        self.initializer = None
        self.avg_model_pred_time = None
        self.abs_percentage_pred_errors = None
        self.architecture = None
        self.load_buses = None
        self.performance_dict = {}
        #self.norm_input = 2
        #self.norm_output = 10
        self._norm_input = 2
        self._norm_output = 10
    '''
    def load_elk_objects(self, path_to_sys_sheet=None):
        
        if path_to_sys_sheet is None:
            path_to_sys_sheet = '/home/clemens/PycharmProjects/NN_LF_Topology/LF_3bus/4 bus 1 gen.xls'
        BusList, LineList = BuildSystem(path_to_sys_sheet)
        self.data_obj = LoadFlow(BusList, LineList)
        self.model_obj = LoadFlow(BusList, LineList)
        pass
    '''
    def set_norm_input(self, integer):
        self._norm_input = integer
        pass

    def set_norm_output(self, integer):
        self._norm_output = integer
        pass

    def get_norm_input(self):
        return self._norm_input

    def get_norm_output(self):
        return self._norm_output

    def init_data(self, name_data_in, name_data_out, ver_frac=None, datapath='', scale_data_out=False, pickle_load=False):

        '''
        Function to initialize data
        :param name_data_in: name of the input datafile including filename extension
        :param name_data_out: Name of the output datafile, including filename extension
        :param ver_frac:
        :param datapath:
        :return: Pass. Stores data within NN obj.
        '''

        if scale_data_out:
            o_scale = self._norm_output
        if pickle_load:
            inputdata = pickle_load_obj(path=datapath, filename=name_data_in)
            outputdata = pickle_load_obj(path=datapath, filename=name_data_out)
        else:
            inputdata = np.load(datapath + name_data_in, allow_pickle=True)
            outputdata = np.load(datapath + name_data_out, allow_pickle=True)

        nr_samples, nr_input_var = np.shape(inputdata)
        v_samples = int(nr_samples // (1/ver_frac))
        learn_samples = int(nr_samples - v_samples)
        self.l_data, self.l_sol = inputdata[:learn_samples] / self._norm_input, outputdata[:learn_samples]*o_scale
        self.t_data, self.t_sol = inputdata[learn_samples:] / self._norm_input, outputdata[learn_samples:]*o_scale
        pass

    def init_nn_model_fixed(self):
        initializer = self.initializer
        loss_fn = self.loss_fn
        self.tf_model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(6,)),
            # tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(12, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(12, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dense(6, kernel_initializer=initializer)
        ])

        self.tf_model.compile(optimizer='adam',
                              loss=loss_fn,
                              metrics=['mean_absolute_percentage_error'])  # , 'accuracy'])
        self.tf_model.summary()
        pass

    def init_nn_model_dynamic(self, architecture=None, const_l_rate=True, print_summary=False, custom_loss=False):

        '''
        Function initializes and compiles a tensorflow NN object within the NeuralNetwork object.
        :param architecture: list containing the amount of neurons in each layer including the input layer.
                                variable is stored in NeuralNetwork.
        :param const_l_rate: bool. variable specify wether or not a decaying learning rate should be used.
        :return:
        '''

        if architecture is None:
            raise Exception('No network architecture provided')
        self.architecture = architecture
        self.tf_model = tf.keras.models.Sequential()
        #self.tf_model = CustomModel() #for more detailed access to control etc.
        self.tf_model.add(layers.Dense(self.architecture[1], activation='relu', input_shape=(self.architecture[0],)))
        self.tf_model.summary

        for layer_idx in range(2, len(self.architecture)):
            neurons = self.architecture[layer_idx]
            if layer_idx == (len(self.architecture) - 1):
                self.tf_model.add(layers.Dense(neurons, kernel_initializer=self.initializer, name='output'))
            else:
                self.tf_model.add(layers.Dense(neurons, activation='relu', kernel_initializer=self.initializer))

        if self.l_rate is None:
            self.set_learning_rate_schedule(const_l_rate=const_l_rate)

        adam = tf.keras.optimizers.Adam(learning_rate=self.l_rate)
        self.tf_model.compile(optimizer=adam,
                              loss=self.loss_fn,
                              metrics=['mean_absolute_percentage_error'], run_eagerly=custom_loss)
                                                                        # Run eagerly reduces performance.

        if self.architecture[-1] % 2 != 0:
            raise Exception('The system does not have equal amounts of active and reactive powers')
        self.load_buses = int(self.architecture[-1] / 2)
        if print_summary:
            self.tf_model.summary()
        pass

    def set_learning_rate_schedule(self, const_l_rate=False, l_rate=1e-3):

        '''
        Function to set the learning rate for the neural network object.
        :param const_l_rate: bool to specify if a constant learning rate should be used.
        :param l_rate: the constant learning rate to be used.
        :return:
        '''

        if const_l_rate:
            self.l_rate = l_rate
        else:
            initial_l_rate = 0.1
            final_l_rate = 0.001
            decay_factor = (final_l_rate / initial_l_rate) ** (1/self.epochs)
            steps_epoch = int(len(self.l_data)/self.batch_size)
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_l_rate,
                decay_steps=steps_epoch,
                decay_rate=decay_factor,
                staircase=True
                )
            self.l_rate = lr_schedule
        pass

    def train_model(self, epochs=None, batch_size=None, checkpoints=False, cp_folder_path=None, save_freq=None):

        '''
        Function to train the model.
        :param epochs: [Optional] if other than specified within obj is to be used.
        :param batch_size: [optional]
        :param checkpoints: Bool to indicate if checkpoints should be stored during training
        :param cp_folder_path: path to container for checkpoint storage
        :param save_freq: storage freqency for checkpoints. Not sure of how to calculate an optimum here.
        :return: Pass. Trains the model stored in NN object.
        '''

        cp_callback=None

        if epochs is None:
            epochs=self.epochs
        if batch_size is None:
            batch_size=self.batch_size

        if checkpoints is True:
            if cp_folder_path is None:
                cp_folder_path= __file__
            if save_freq is None:
                save_freq = 1200*batch_size
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_folder_path,
                                                             save_weights_only=True,
                                                             verbose=1,
                                                             save_freq=save_freq)
            self.tf_model.fit(self.l_data,
                              self.l_sol,
                              epochs=self.epochs,
                              batch_size=self.batch_size,
                              callbacks=[cp_callback],
                              #validation_data=(self.t_data, self.t_sol),
                              verbose=1
                              )
        else:
            self.tf_model.fit(self.l_data,
                              self.l_sol,
                              epochs=self.epochs,
                              batch_size=self.batch_size,
                              #validation_data=(self.t_data, self.t_sol),
                              verbose=1
                              )
        pass


    def single_prediction(self, inputs):

        '''

        :param inputs: array/list containing the same amount of input values as the input nodes in the network.
        :return: network predictions.
        '''
        inputs = np.array([inputs])
        return self.tf_model(inputs).numpy()

    def model_pred(self):

        '''
        calculate model predicitons for all data in the test data and store them in self.model_sol
        Function also calculates percentage deviations from design value.
        :return: pass
        '''

        verification_samples, verification_variables = np.shape(self.t_data)
        model_prediction_time = np.zeros((verification_samples,))
        model_predictions = np.zeros((verification_samples, verification_variables))
        counter = 0
        for verification_row in self.t_data:
            #model_in = np.array([verification_row])
            m_time = time.perf_counter()
            model_predictions[counter] = self.single_prediction(verification_row)
            model_prediction_time[counter] = time.perf_counter() - m_time
            counter += 1
        self.avg_model_pred_time = np.average(model_prediction_time)
        self.model_sol = model_predictions
        self.abs_percentage_pred_errors = np.abs(np.divide(np.subtract(self.model_sol, self.t_sol), self.t_sol) * 100)
        pass

    def load_latest_pretrained_model(self, path):
        #loads the latest model weights of a pretrained model into the objects model.
        #Note that the layer architecture must be the same for this function to work.
        latest = tf.train.latest_checkpoint(path)
        self.tf_model.load_weights(latest)
        pass

    '''
    def generate_performance_data_dict(self, threshold=5):
        
        def find_affected_sets():
            affected_predictions = set()
            for index in self.performance_dict[threshold_key]['worsts']:
                affected_predictions.add(index[0])
            return len(affected_predictions)

        threshold_key = str(threshold) + 'percent'

        self.performance_dict[threshold_key] = {}
        self.performance_dict[threshold_key]['averages'] = np.average(self.abs_percentage_pred_errors, axis=0)
        self.performance_dict[threshold_key]['average'] = np.average(self.performance_dict[threshold_key]['averages'])
        self.performance_dict[threshold_key]['worsts'] = np.argwhere(self.abs_percentage_pred_errors > threshold)
        self.performance_dict[threshold_key]['affected_pred_sets_worst'] = find_affected_sets()

        pass'''

    def generate_performance_data_dict_improved(self, threshold=[5]):
        '''
        Function calculates a variety of different performance data for a given threshold value.
        :param threshold: list of thresholds for model performance evaluation
        :return: pass. Data stored within a dictionary.
        '''
        def find_affected_sets(threshold_percentage):
            affected_predictions = set()
            for index in self.performance_dict[threshold_percentage]['pred_errors_over_threshold']:
                affected_predictions.add(index[0])
            return len(affected_predictions)

        #threshold_independent_metrics

        self.performance_dict['averages'] = np.average(self.abs_percentage_pred_errors, axis=0)
        self.performance_dict['overall_average'] = np.average(self.performance_dict['averages'])
        self.performance_dict['average_angle'] = np.average(self.performance_dict['averages'][self.load_buses:])
        self.performance_dict['average_voltage'] = np.average(self.performance_dict['averages'][:self.load_buses])

        #threshold_dependent_metrics
        for threshold_number in threshold:
            threshold_key = str(threshold_number) + 'percent'
            self.performance_dict[threshold_key] = {}
            self.performance_dict[threshold_key]['pred_errors_over_threshold'] = np.argwhere(
                                                            self.abs_percentage_pred_errors > threshold_number)
            affected_sets = find_affected_sets(threshold_key)
            self.performance_dict[threshold_key]['sets_with_pred_error_over_threshold'] = affected_sets
            self.performance_dict[threshold_key]['sets_worse_than_threshold'] = affected_sets / self.t_sol.shape[0] * 100

        pass

    def gen_loss_function(self):

        '''
        Function to generate a custom loss function for tensorflow.
        :return:
        '''

        pass

    def split_data_into_vomag_voang(self):

        '''
        Function to split training data into voltage magnitude and angle to train separate networks for
        voltage magnitude and voltage angle predictions.
        :return:
        '''

        pass

