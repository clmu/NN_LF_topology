import tensorflow as tf
import numpy as np
from LF_3bus.build_sys_from_sheet import BuildSystem
from LF_3bus.ElkLoadFlow import LoadFlow

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
        super().__init__()
        self.y_bus_matrix = None

    def call(self, y_true, y_pred):
        return tf.square(y_true-y_pred)

    def calc_line_flow_matrix(self, tensor):

        '''
        Function is to take in a tensor from a prediction, and return a matrix containing
        the line flows to add them to the prediction.
        function assumes distribution grid where a single
        :param tensor:
        :return: matrix of line flows
        '''

        if self.y_bus_matrix is None:
            self.y_bus_matrix = self.init_y_bus_matrix_from_file(path_to_sys_file)

        batch, outputs = tf.shape(tensor)
        buses = 3
        buslist = np.linspace(0, buses)
        lineflows = np.zeros((buses + 1, buses + 1), dtype=complex)
        pass


    def init_y_bus_matrix_from_file(self, path):
        buslist, linelist = BuildSystem(path)
        lf_obj = LoadFlow(buslist, linelist)
        self.y_bus_matrix = lf_obj.ybus
        pass