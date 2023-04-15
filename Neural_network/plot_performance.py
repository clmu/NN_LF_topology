import matplotlib.pyplot as plt
import numpy as np
from Neural_network.NN_objects import pickle_load_obj as load

path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/checkpoints/'
network_size = 'medium'
remark = 'baseline_slim'

loss_fun_list = ['MSE', 'CustomLoss']
performance_data = []

for loss in loss_fun_list:
    datapath = path + network_size + '/' + loss + '_' + remark + '/'
    performance_data.append(load(path=datapath, filename='perf_dict.obj'))

epochs = len(performance_data[0])
epoch_vec = np.arange(epochs)
averages = [[], []]

for epoch in range(epochs):
    averages[0].append(performance_data[0][epoch]['overall_average'])
    averages[1].append(performance_data[1][epoch]['overall_average'])

def inches(cm):
    return cm/2.54

fig_savepath = '/home/clemens/Dropbox/EMIL_MIENERG21/Master/Master/Figures/Results/'

ylim = [0, 12]

plt.plot(epoch_vec, averages[0], label=loss_fun_list[0])
plt.plot(epoch_vec, averages[1], label=loss_fun_list[1])
plt.xlabel('Epoch number')
plt.ylabel('Average percentage accuracy')
plt.gca().set_ylim(ylim)
plt.xticks(np.arange(min(epoch_vec), max(epoch_vec), 2))
plt.yticks(np.arange(min(ylim), max(ylim)+1, 0.5))
plt.xticks()
plt.legend()
plt.grid()
plt.gcf().set_size_inches(inches(16), inches(10))
plt.savefig(fig_savepath + 'pinn_medium_baseline', dpi=600)

