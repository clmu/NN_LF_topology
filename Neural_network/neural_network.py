import os
import tensorflow as tf
import numpy as np

#print('TensorFlow version: ', tf.__version__) #to verify correct installation, seems to be OK.

path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'


norm = 10 #value to ensure all inputs are between 0 and 1.

small_dataset = False

about = 'large_sq_err.ckpt'

if small_dataset:
    small = '_small'
else:
    small = ''

(l_i, l_o) = np.load(path + 'learn_input' + small +'.npy')/ norm, np.load(path + 'learn_output' + small + '.npy')
(v_i, v_o) = np.load(path + 'verification_input' + small + '.npy')/norm, np.load(path + 'verification_output'
                                                                                 + small + '.npy')

#mnist = tf.keras.datasets.mnist
#(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

#learn_dataset = tf.data.Dataset.from_tensor_slices((l_i, l_o))
#ver_dataset = tf.data.Dataset.from_tensor_slices((v_i, v_o))

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(6,)),
    #tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(6)
])

test_format = l_i[:1]

prediction = model(l_i[:1]).numpy() #testing a prediction

#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn = tf.keras.losses.MeanSquaredError()
#likely a bad error function since most numbers are below 1: hence the error will always be small.
#loss_fn = tf.keras.losses.MeanAbsoluteError()

#print(loss_fn(l_o[:1], prediction).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
'''
Epoch: number of times the whole training set is worked through
batch: number of samples worked through before the weights are adjusted. 
'''
model.fit(l_i, l_o, epochs=150, batch_size=20)
loss, acc = model.evaluate(v_i, v_o)

print('model accuracy: {:5.2f}%'.format(100*acc))

test_4_bus_1_gen = np.array([[0.02, 0.04, 0.125, 0.007, 0.05, 0.08]])


result_standard_test = model(test_4_bus_1_gen)

cp_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/checkpoints/cp_' + about
cp_dir = os.path.dirname(cp_path)

#create callback to save model weights.
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path,
                                                 save_weights_only=True,
                                                 verbose=1)

