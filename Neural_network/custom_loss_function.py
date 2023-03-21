import numpy
import tensorflow as tf
from keras import backend as K
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

    #print(f'y_true: {y_true}')
    #print(f'y_pred: {y_pred}')
    return tf.square(y_true-y_pred)

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, path='path', o_norm=10):
        '''
        do not change this function as it is used by tensorflow backend...?
        '''
        super(CustomLoss, self).__init__()
        buslist, linelist = BuildSystem(path)
        lf_obj = LoadFlow(buslist, linelist)
        self.y_bus_matrix = lf_obj.ybus
        self.output_normalizer = o_norm
        self.buses_in_sys = None
        #tf.compat.v1.enable_eager_execution()#trying to force eager execution as some methods require it.

    def init_remaining_values(self, path='path', output_normalizer=10):
        buslist, linelist = BuildSystem(path)
        lf_obj = LoadFlow(buslist, linelist)
        self.y_bus_matrix = lf_obj.ybus
        self.output_normalizer = output_normalizer
        pass

    def call(self, y_true, y_pred):
        '''
        do not change inputs of this function as it is used by the tensorflow backend....?
        Function returns the sum of line true prediction errors and
        :param y_true: correct outputs
        :param y_pred: model predicted outputs.
        :return: the loss value returned to tensorflow.
        '''
        line_true = self.calc_abs_mean_flows_for_batch(y_true) * self.output_normalizer
        line_pred = self.calc_abs_mean_flows_for_batch(y_pred) * self.output_normalizer
        regular_MSE = tf.cast(K.mean(K.square(y_true-y_pred)), tf.float64)
        line_pred_MSE = tf.cast(K.mean(K.square(line_true-line_pred)), tf.float64)
        return regular_MSE + line_pred_MSE

    def calc_line_flow_matrix(self, tensor):

        '''
        Function takes in a single output from a batch of tensors
        function assumes distribution grid using a single slack bus (typical distribution grid)
        the returned lineflow matrix contains complex nans (nan + j*nan) where there are no connections.
        :param tensor: single output tensor within the batch to be processed.
        :return: matrix of line flows for the given output.
        '''

        def gen_complex_voltage(vomag, voang):
            real = vomag * np.cos(voang)
            cplx = vomag * np.sin(voang)
            return complex(real, cplx)

        outputs = tensor.shape[0]#tf.shape(tensor).numpy()
        buses = outputs / 2 #assuming pure load buses only in network.
        tensor = tensor / self.output_normalizer # to calculate actual values, outputs are scaled since they are below 1
        if int(buses) - buses != 0:
            raise Exception('Number of buses is not an integer.')
        else:
            self.buses_in_sys = int(buses)+1#compensate for slack bus

        sys_Vs = np.ones(self.buses_in_sys)
        sys_As = np.zeros(self.buses_in_sys)
        '''
        sys_Vs[0:self.buses_in_sys-1] = tensor.numpy()[:self.buses_in_sys-1]
        sys_As[0:self.buses_in_sys-1] = tensor.numpy()[self.buses_in_sys-1:]
        sess = K.get_session()
        tensor_array = sess.run(tensor)
        sys_Vs[0:self.buses_in_sys - 1] = tensor_array[:self.buses_in_sys - 1]
        sys_As[0:self.buses_in_sys-1] = tensor_array[self.buses_in_sys-1:]
        '''
        sys_Vs[0:self.buses_in_sys - 1] = tensor.numpy()[:self.buses_in_sys - 1]
        sys_As[0:self.buses_in_sys - 1] = tensor.numpy()[self.buses_in_sys - 1:]
        cplx_Vs = np.zeros(len(sys_Vs), dtype=complex)
        for bus_idx in range(len(cplx_Vs)):
            cplx_Vs[bus_idx] = gen_complex_voltage(sys_Vs[bus_idx], sys_As[bus_idx])

        line_flows = np.zeros((self.buses_in_sys, self.buses_in_sys), dtype=complex)
        for r in range(self.buses_in_sys):
            for c in range(self.buses_in_sys):
                if r == c: #all diagonal elements in the lineflow matrix must be zero.
                    line_flows[r, c] = complex(numpy.NaN, numpy.NaN)
                    continue
                elif self.y_bus_matrix[r, c] == complex(0, 0): #continue if no connection between the buses
                    line_flows[r, c] = complex(numpy.NaN, numpy.NaN)
                    continue
                elif line_flows[r, c] == complex(0, 0):
                    lineflow = self.y_bus_matrix[r, c] * (cplx_Vs[r]-cplx_Vs[c])
                    line_flows[r, c] = lineflow
                    line_flows[c, r] = -1 * lineflow
        return line_flows

    def calc_abs_avg_lineflows(self, line_flows):
        '''
        Fucntion takes the sum of absolute lineflows connected to a specific bus,
        calculates the average for all nonzero lineflows.
        array with values corresponding to the given tensor.
        :param tensor: single output tensor.
        :param line_flows: line flow matrix corresponding to the output tensor.
        :return: array
        '''
        return np.nanmean(np.abs(line_flows[:, :self.buses_in_sys-1]), axis=0)

    def associate_mean_lineflows_to_variables(self, tensor, flows):
        abs_mean_flows = self.calc_abs_avg_lineflows(flows)
        #correction_vector = np.zeros(len(tensor.numpy()), dtype=float)
        correction_vector = np.zeros(tensor.shape, dtype=float)
        correction_vector[:self.buses_in_sys - 1], correction_vector[
                                                   self.buses_in_sys - 1:] = abs_mean_flows, abs_mean_flows
        return correction_vector

    def calc_abs_mean_flows_for_batch(self, batch_tensor):
        #batches, outputs = batch_tensor.numpy().shape #tf.shape(tensor).numpy()
        batches, outputs = batch_tensor.shape #tf.shape(batch_tensor).numpy()
        avg_flow_for_batch = np.zeros((batches, outputs), dtype=float)
        for batch_idx in range(batches):
            line_flow_matrix = self.calc_line_flow_matrix(batch_tensor[batch_idx])
            avg_flow_for_batch[batch_idx] = self.associate_mean_lineflows_to_variables(
                                                        batch_tensor[batch_idx], line_flow_matrix)
        return tf.constant(avg_flow_for_batch)

class SquaredLineFlowLoss(CustomLoss):

    '''def __init__(self, path='path', o_norm=10):
        super(CustomLoss).__init__()
        buslist, linelist = BuildSystem(path)
        lf_obj = LoadFlow(buslist, linelist)
        self.y_bus_matrix = lf_obj.ybus
        self.output_normalizer = o_norm
        self.buses_in_sys = None
        #required by backend:
        self.reduction = 'auto'
        self.name = None
        #"protected" attributes
        self._name_scope = 'SquaredLineFlowLoss'
        self._allow_sum_over_batch_size = False'''
    def __init__(self, path='no_path_provided', o_norm=10):
        super().__init__(path, o_norm)

    def call(self, y_true, y_pred):
        line_true = self.calc_abs_mean_flows_for_batch(y_true) * self.output_normalizer
        line_pred = self.calc_abs_mean_flows_for_batch(y_pred) * self.output_normalizer
        '''
        regular_sq_loss = tf.cast(K.square(y_true-y_pred), tf.float64)
        line_sq_loss = tf.cast(K.square(line_true-line_pred), tf.float64)
        regular_sq_loss = K.square(y_true-y_pred)
        line_sq_loss = K.square(line_true-line_pred)
        thesum = tf.cast(tf.add(regular_sq_loss, line_sq_loss), tf.float64)
        print(thesum)
        '''
        pure_output_element = tf.cast(K.square(y_true-y_pred), tf.float64)
        line_flow_element = tf.cast(K.square(line_true - line_pred), tf.float64)
        return pure_output_element + line_flow_element


'''
class InternalInheritance(CustomLoss):
    # initialize instance attributes
    def __init__(self, path='path', o_norm=10):
        super().__init__(path, o_norm)

    # Compute loss
    def call(self, y_true, y_pred):
        return K.square(y_true, y_pred)
'''