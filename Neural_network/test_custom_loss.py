import numpy as np
import tensorflow as tf
from Neural_network.custom_loss_function import CustomLoss, loss_acc_for_lineflows, SquaredLineFlowLoss
from Neural_network.NN_objects import NeuralNetwork

nn = NeuralNetwork()

path_sys_file = '/home/clemens/PycharmProjects/NN_LF_Topology/LF_3bus/4 bus 1 gen.xls'
loss = CustomLoss(path=path_sys_file, o_norm=nn.norm_output)
sqloss = SquaredLineFlowLoss(path=path_sys_file, o_norm=nn.norm_output)

path_to_data = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
nn.init_data('simple data.npy',
                 'simple o data.npy',
                 0.2,
                 datapath=path_to_data,
                 scale_data_out=True)
huber = tf.keras.losses.huber
tf_poisson = tf.keras.losses.Loss

tensor = tf.constant(nn.t_sol[:20])
tensor_data_pred_fake = tf.constant(nn.t_sol[20:40])
#lineflows = loss.calc_line_flow_matrix(tensor[0])
#lineflows = np.abs(lineflows)
#average_lineflows = np.nanmean(lineflows, axis=0)#print(type(lineflows[0][0]))
#correction_vector = loss.associate_mean_lineflows_to_variables(test_tensor[0], lineflows)

abs_mean_flow_batch_tensor = loss.calc_abs_mean_flows_for_batch(tensor)
loss_return = loss.call(tensor, tensor_data_pred_fake)
huber_loss_return = huber(tensor, tensor_data_pred_fake)
sq_loss_return = sqloss.call(tensor, tensor_data_pred_fake)
#tf_poisson_return = tf_poisson(tensor, tensor_data_pred_fake)
first_return = loss_acc_for_lineflows(tensor, tensor_data_pred_fake)