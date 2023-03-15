import tensorflow as tf
import numpy as np

def loss_acc_for_lineflows(y_true, y_pred):

    '''
    ****Test function****

    y_true and y_pred are tensors of dimension [batch_size, outputs]

    :param y_true: the real solution for the data.
    :param y_pred: Model predicted solution for the data.
    :return: regular square loss for the data.

    '''

    print(f'y_true: {y_true}')
    print(f'y_pred: {y_pred}')
    return tf.square(y_true-y_pred)

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(self)

    def call(self, y_true, y_pred):
        return y_true-y_pred

    def calc_line_flow_matrix(self, tensor):

        '''
        Function is to take in a tensor from a prediction, and return a matrix containing
        the line flows
        :param tensor:
        :return: matrix of line flows
        '''
        batch, outputs = tf.shape(tensor)
        buses = 3
        buslist = np.linspace(0, buses)
        lineflows = np.zeros[]

