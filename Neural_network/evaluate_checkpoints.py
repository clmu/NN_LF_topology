import numpy as np
from Neural_network.nn_functions import import_data
from Neural_network.data_evaluation import gen_list_of_models, load_respective_checkpoints, predict, evaluate_checkpoints
from matplotlib import pyplot as plt


datapath = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
input = np.load(datapath + 'simple data.npy')
output = np.load(datapath + 'simple o data.npy')

norm_inputs = 2
norm_outputs = 10


(l_i, l_o), (v_i, v_o) = import_data('simple data.npy', 'simple o data.npy', datapath=datapath, norm_input=norm_inputs, norm_output=norm_outputs)

number_of_models = 15
start_model = 10 #checkpoint of model

models = gen_list_of_models(number_of_models)


cp_path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/checkpoints/'


load_respective_checkpoints(models, start_model, cp_path)

list_of_predictions, list_of_prediction_times = predict(models, v_i)

accuracy_vectors = np.array(evaluate_checkpoints(v_o, list_of_predictions))

np.save(datapath + 'accuracy_vectors.npy', accuracy_vectors)
avg_accuracies = np.average(accuracy_vectors[0], axis=1)
avg_worst = np.average(accuracy_vectors[1], axis=1)

epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

#plt.subplots(figsize=(6,2))
cm = 1/2.54
fig, ax1 = plt.subplots(figsize=(16*cm, 10*cm), dpi=300)
ax2 = ax1.twinx()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Average percentage deviation')
ax2.set_ylabel('Worst percentage deviation')
avg = ax1.plot(epochs, avg_accuracies, color='blue', label='avg')
worst = ax2.plot(epochs, avg_worst, linestyle='dotted', color='red', label='worst')
'''
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
'''
lbs = avg+worst
labels = [l.get_label() for l in lbs]
ax1.legend(avg+worst, labels, loc='upper right')
plt.savefig('epochs_accuracy', format='png')
plt.show()


print('finito')