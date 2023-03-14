import os
import time

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
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
        self.norm_input = 2
        self.norm_output = 10
        self.tf_model = None
        self.loss_fn = None
        self.initializer = None
        self.avg_model_pred_time = None
        self.abs_percentage_pred_errors = None
        self.architecture = None

    def init_data(self, name_data_in, name_data_out, ver_frac, datapath='', scale_data_out=False):

        '''
        Function to initialize data
        :param name_data_in: name of the input datafile including filename extension
        :param name_data_out: Name of the output datafile, including filename extension
        :param ver_frac:
        :param datapath:
        :return: Pass. Stores data within NN obj.
        '''

        if scale_data_out:
            o_scale = self.norm_output

        inputdata = np.load(datapath + name_data_in)
        outputdata = np.load(datapath + name_data_out)

        nr_samples, nr_input_var = np.shape(inputdata)
        v_samples = int(nr_samples // (1/ver_frac))
        learn_samples = int(nr_samples - v_samples)
        self.l_data, self.l_sol = inputdata[:learn_samples] / self.norm_input, outputdata[:learn_samples]*o_scale
        self.t_data, self.t_sol = inputdata[learn_samples:] / self.norm_input, outputdata[learn_samples:]*o_scale
        pass

    def init_nn_model(self, architecture=None, const_l_rate=True):

        '''
        Function to initialize neural network using the system architecture in the list self.structure
        :param loss_fn: the loss function to be used witin the model.
        :param initializer = initializer to be used for weight initialization for each layer
        :return: none. Stores NN in object. prints a summary of the intialized nn.

        '''
        if architecture is None:
            raise Exception('No network architecture provided')
        self.architecture = architecture
        self.tf_model = tf.keras.models.Sequential()
        #dense = tf.keras.layers.Dense()
        for i in range(len(self.structure)):
            if i == 0:
                self.tf_model.add(tf.keras.layers.Flatten(input_shape=(self.structure[0],)))
                '''self.tf_model.add(tf.keras.layers.Dense({inputShape: [self.structure[0]],
                                                         units: self.structure[0],
                                                         activation: 'relu'));'''
                #Does this line need to be initialized?
            elif i == self.structure[-1]:
                self.tf_model.add(tf.keras.layers.Dense(self.structure[i], #activation='linear',
                                                        kernel_initializer=self.initializer))
            else:
                self.tf_model.add(tf.keras.layers.Dense(self.structure[i],
                                                        activation='relu', kernel_initializer=self.initializer))

        if self.l_rate is None:
            self.set_learning_rate_schedule(const_l_rate=const_l_rate)
            adam = tf.keras.optimizers.Adam(learning_rate=self.l_rate)

        self.tf_model.compile(optimizer=adam,
                              loss=self.loss_fn,
                              metrics=['mean_absolute_percentage_error'])#metrics=['accuracy'])
        self.tf_model.summary()
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

    def init_nn_model_dynamic(self, architecture=None, const_l_rate=True):

        if architecture is None:
            raise Exception('No network architecture provided')
        self.architecture = architecture
        self.tf_model = tf.keras.models.Sequential()
        self.tf_model.add(layers.Dense(self.architecture[1], activation='relu', input_shape=(self.architecture[0],)))
        self.tf_model.summary

        for layer_idx in range(2, len(self.architecture)):
            neurons = self.architecture[layer_idx]
            if layer_idx == (len(self.architecture) - 1):
                self.tf_model.add(layers.Dense(neurons, kernel_initializer=self.initializer))
            else:
                self.tf_model.add(layers.Dense(neurons, activation='relu', kernel_initializer=self.initializer))

        if self.l_rate is None:
            self.set_learning_rate_schedule(const_l_rate=const_l_rate)
            adam = tf.keras.optimizers.Adam(learning_rate=self.l_rate)
        self.tf_model.compile(optimizer=adam,
                              loss = self.loss_fn,
                              metrics=['mean_absolute_percentage_error'])
        self.tf_model.summary()
        pass

    def set_learning_rate_schedule(self, const_l_rate=False, l_rate=1e-3):
        if const_l_rate:
            self.l_rate = l_rate
        else:
            initial_l_rate = 0.1
            final_l_rate = 0.0001
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
        pass

    def load_latest_pretrained_model(self, path):
        #loads the latest model weights of a pretrained model into the objects model.
        #Note that the layer architecture must be the same for this function to work.
        latest = tf.train.latest_checkpoint(path)
        self.tf_model.load_weights(latest)
        pass

    def gen_loss_function(self):
        pass

    def eval_model_performance(self):
        '''
        #(nr_results, model_results) = self.t_sol, self.model_sol
        #samples, outputs = np.shape(nr_results)
        samples, outputs = np.shape(self.t_sol)
        #percentage_error_matrix=np.zeros((samples, outputs), dtype=float)
        self.abs_percentage_pred_errors=np.zeros((samples, outputs), dtype=float)
        for i in range(samples):
            temp = np.divide(np.subtract(self.model_sol[i], self.t_sol[i]), self.t_sol[i]) * 100
            self.abs_percentage_pred_errors[i] = temp
            #percentage_error_matrix[i] = (model_results[i] - nr_results[i]) / nr_results[i] * 100
            #self.abs_percentage_pred_errors[i] = (self.t_sol[i] - self.model_sol[i]) / self.model_sol * 100
        #self.abs_percentage_pred_errors = np.abs(percentage_error_matrix)
        '''
        self.abs_percentage_pred_errors = np.divide(np.subtract(self.model_sol, self.t_sol), self.t_sol)*100
        pass

    def eval_worst_model_performance(self):
        pass


