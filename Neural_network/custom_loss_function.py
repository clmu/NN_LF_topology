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
        self.output_normalizer = None

    def call(self, y_true, y_pred):
        return tf.square(y_true-y_pred)

    def calc_line_flow_matrix(self, tensor):

        '''
        Function takes in a single output from a batch of tensors
        function assumes distribution grid using a single slack bus (typical distribution grid)
        :param tensor: single output tensor within the batch to be processed.
        :return: matrix of line flows for the given output.
        '''

        def gen_complex_voltage(vomag, voang):
            real = vomag * np.cos(voang)
            cplx = vomag * np.sin(voang)
            return complex(real, cplx)

        outputs = tf.shape(tensor).numpy()
        buses = outputs / 2 #assuming pure load buses only in network.
        tensor = tensor / self.output_normalizer # to calculate actual values, outputs are scaled since they are below 1
        if int(buses) - buses != 0:
            raise Exception('Number of buses is not an integer.')
        else:
            buses = int(buses)+1#compensate for slack bus

        sys_Vs = np.ones(buses)
        sys_As = np.zeros(buses)
        sys_Vs[0:buses-1] = tensor.numpy()[:buses-1]
        sys_As[0:buses-1] = tensor.numpy()[buses-1:]
        cplx_Vs = np.zeros(len(sys_Vs), dtype=complex)
        for bus_idx in range(len(cplx_Vs)):
            cplx_Vs[bus_idx] = gen_complex_voltage(sys_Vs[bus_idx], sys_As[bus_idx])

        line_flows = np.zeros((buses, buses), dtype=complex)
        for r in range(buses):
            for c in range(buses):
                if r == c: #all diagonal elements in the lineflow matrix must be zero.
                    continue
                elif self.y_bus_matrix[r, c] == complex(0, 0): #continue if no connection between the buses
                    continue
                elif line_flows[r, c] == complex(0, 0):
                    lineflow = self.y_bus_matrix[r, c] * (cplx_Vs[r]-cplx_Vs[c])
                    line_flows[r, c] = lineflow
                    line_flows[c, r] = -1 * lineflow
        return line_flows

    def associate_lineflows_to_variables(self, tensor, line_flows):
        '''
        Fucntion takes the sum of absolute lineflows connected to a specific bus,
        calculates the average for all lines connected to the bus and returns an
        array with values corresponding to the given tensor.
        :param tensor: single output tensor.
        :param line_flows: line flow matrix corresponding to the output tensor.
        :return: array
        '''
        correction_vector = np.zeros(len(tensor.numpy()), dtype=float)
        r, c = line_flows.shape()
        abs_lineflows = np.abs(line_flows[:,:])
        np.average(abs_lineflows, axis=0)

        return correction_vector
    def init_remaining_values(self, path='path',output_normalizer=10 ):
        buslist, linelist = BuildSystem(path)
        lf_obj = LoadFlow(buslist, linelist)
        self.y_bus_matrix = lf_obj.ybus
        self.output_normalizer = output_normalizer
        pass