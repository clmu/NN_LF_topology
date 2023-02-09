import tensorflow as tf
import numpy as np

#print('TensorFlow version: ', tf.__version__) #to verify correct installation, seems to be OK.

path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'


norm = 10 #value to ensure all inputs are between 0 and 1.

(l_i, l_o) = np.load(path + 'learn_input.npy')/ norm, np.load(path + 'learn_output.npy')/norm
(v_i, v_o) = np.load(path + 'verification_input.npy')/norm, np.load(path + 'verification_output.npy')/norm

#mnist = tf.keras.datasets.mnist
#(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

#learn_dataset = tf.data.Dataset.from_tensor_slices((l_i, l_o))
#ver_dataset = tf.data.Dataset.from_tensor_slices((v_i, v_o))

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(6,)),
    #tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(6)
])

test_format = l_i[:1]

prediction = model(l_i[:1]).numpy() #testing a prediction

#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn = tf.keras.losses.MeanSquaredError()
#likely a bad error function since most numbers are below 1: hence the error will always be small.

#print(loss_fn(l_o[:1], prediction).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
'''
Epoch: number of times the whole training set is worked through
batch: number of samples worked through before the weights are adjusted. 
'''
model.fit(l_i, l_o, epochs=500, batch_size=30)
model.evaluate(v_i, v_o)

