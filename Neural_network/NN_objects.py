import os
import time

import tensorflow as tf
import numpy as np

architecture = [6, 12, 12, 12, 6]
class NeuralNetwork:
    def __init__(self, structure=None, learning_rate=1e-4):
        self.l_rate = learning_rate
        self.epochs = 150
        self.batch_size = 20
        self.l_data = None
        self.l_sol = None
        self.t_data = None
        self.t_sol = None
        self.model_sol = None
        self.norm_input = 2
        self.tf_model = None
        self.loss_fn = None
        self.initializer = None
        self.avg_model_pred_time = None
        self.abs_percentage_pred_errors = None
        if structure is None:
            self.structure = architecture

    def init_data(self, name_data_in, name_data_out, ver_frac, datapath=''):

        '''
        Function to initialize data
        :param name_data_in: name of the input datafile including filename extension
        :param name_data_out: Name of the output datafile, including filename extension
        :param ver_frac:
        :param datapath:
        :return: Pass. Stores data within NN obj.
        '''

        inputdata = np.load(datapath + name_data_in)
        outputdata = np.load(datapath + name_data_out)

        nr_samples, nr_input_var = np.shape(inputdata)
        v_samples = int(nr_samples // (1/ver_frac))
        learn_samples = int(nr_samples - v_samples)
        self.l_data, self.l_sol = inputdata[:learn_samples] / self.norm_input, outputdata[:learn_samples]
        self.t_data, self.t_sol = inputdata[learn_samples:] / self.norm_input, outputdata[learn_samples:]
        pass

    def init_nn_model(self):

        '''
        Function to initialize neural network using the system architecture in the list self.structure
        :param loss_fn: the loss function to be used witin the model.
        :param initializer = initializer to be used for weight initialization for each layer
        :return: none. Stores NN in object. prints a summary of the intialized nn.

        '''

        self.tf_model = tf.keras.models.Sequential()
        #dense = tf.keras.layers.Dense()
        for i in range(len(self.structure)):
            if i == 0:
                self.tf_model.add(tf.keras.layers.Flatten(input_shape=(self.structure[0],)))
                #Does this line need to be initialized?
            elif i == self.structure[-1]:
                self.tf_model.add(tf.keras.layers.Dense(self.structure[i],
                                                        kernel_initializer=self.initializer))
            else:
                self.tf_model.add(tf.keras.layers.Dense(self.structure[i],
                                                        activation='relu',
                                                        kernel_initializer=self.initializer))

        adam = tf.keras.optimizers.Adam(learning_rate=self.l_rate)
        self.tf_model.compile(optimizer=adam,
                              loss=self.loss_fn,
                              metrics=['mean_absolute_percentage_error'])#metrics=['accuracy'])
        self.tf_model.summary()
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
                              validation_data=(self.t_data, self.t_sol),
                              verbose=1
                              )
        else:
            self.tf_model.fit(self.l_data,
                              self.l_sol,
                              epochs=self.epochs,
                              batch_size=self.batch_size,
                              validation_data=(self.t_data, self.t_sol),
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
        calculate model predicitons for all data in the test data an store them in self.model_sol
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
        (nr_results, model_results) = self.t_sol, self.model_sol
        samples, outputs = np.shape(nr_results)
        percentage_error_matrix=np.zeros((samples, outputs), dtype=float)
        for i in range(samples):
            percentage_error_matrix[i] = (model_results[i] - nr_results[i]) / nr_results[i] * 100
        self.abs_percentage_pred_errors = np.abs(percentage_error_matrix)
        pass

    def eval_worst_model_performance(self):
        pass


