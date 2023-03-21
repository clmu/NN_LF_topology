import tensorflow as tf
import numpy as np

from Neural_network.NN_objects import NeuralNetwork as NN
def gen_filename_ext(number):
    number = str(number)
    while len(number) < 4:
        number = '0' + number
    number = number + '.index'
    return number


path_to_data = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
cp_path_custom_loss = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/cp_small_CustomLoss/cp_'
architecture = [6, 12, 12, 12, 6]

model_numbers = np.arange(1, 30+1)
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


for index in range(len(models)):
    path = cp_path_custom_loss + file_exts[index]
    models[index].tf_model.load_weights(cp_path_custom_loss + file_exts[index])








