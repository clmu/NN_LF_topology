import matplotlib.pyplot as plt
import numpy as np
from Neural_network.NN_objects import pickle_load_obj as load

path = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/checkpoints/'
network_size = 'large'
remark = 'large_30batch'

loss_fun_list = ['MSE', 'CustomLoss', 'SquaredLineFlowLoss']#['MSE']#['MSE' , 'CustomLoss']
performance_data = {}

for loss in loss_fun_list:
    datapath = path + network_size + '/' + loss + '_' + remark + '/'
    performance_data[loss] = load(path=datapath, filename='perf_dict_from_list.obj')

epochs = len(performance_data[loss_fun_list[0]])
epoch_vec = np.arange(1, epochs+1)

averages, v_avg, a_avg = [], [], []

for loss in range(len(loss_fun_list)):
    averages.append([])
    v_avg.append([])
    a_avg.append([])
#averages = [[], [], []]
#v_avg = [[], [], []]
#a_avg = [[], [], []]

for epoch in range(epochs):
    for loss_idx in range(len(loss_fun_list)):
        averages[loss_idx].append(performance_data[loss_fun_list[loss_idx]][epoch]['overall_average'])
        v_avg[loss_idx].append(performance_data[loss_fun_list[loss_idx]][epoch]['average_voltage'])
        a_avg[loss_idx].append(performance_data[loss_fun_list[loss_idx]][epoch]['average_angle'])
    #averages[1].append(performance_data[loss_fun_list[1]][epoch]['overall_average'])

def inches(cm):
    return cm/2.54

fig_savepath = '/home/clemens/Dropbox/EMIL_MIENERG21/Master/Master/Figures/Results/'

ylim = [0, 10]
color_list = ['red', 'blue', 'green', 'purple']

for loss_idx in range(len(loss_fun_list)):
    plt.plot(epoch_vec, averages[loss_idx], label=loss_fun_list[loss_idx], color=color_list[loss_idx])
#plt.plot(epoch_vec, averages[0], label=loss_fun_list[0], color='red')
#plt.plot(epoch_vec, averages[1], label=loss_fun_list[1], color='blue')
#plt.plot(epoch_vec, averages[2], label=loss_fun_list[2], color='green')
plt.xlabel('Epoch number')
plt.ylabel('Average percentage accuracy')
#plt.gca().set_ylim(ylim)
plt.xticks(np.arange(min(epoch_vec)-1, max(epoch_vec)-1, 5))
#plt.yticks(np.arange(min(ylim), max(ylim)+1, 0.5))
plt.xticks()
plt.legend()
plt.grid()
plt.gcf().set_size_inches(inches(16), inches(10))
plt.savefig(fig_savepath + 'pinn_' + network_size + '_' + remark, dpi=600)

'''
fig, ax = plt.subplots()
ax.plot(epoch_vec, v_avg[0], color='red')
ax.set_xlabel('Epochs')
ax.set_ylabel('Percentage magnitude accuracy')

ax2 = ax.twinx()
ax2.plot(epoch_vec, a_avg[0], color='blue')
ax2.set_ylabel('Percentage angle accuracy')
ax.figure.legend(['v_mag', 'angle'], bbox_to_anchor=(1., 1), loc=1, bbox_transform=ax.transAxes)
ax.autoscale(enable=True, axis='both', tight=False)
plt.legend()
plt.grid()
plt.gcf().set_size_inches(inches(16), inches(10))
plt.savefig(fig_savepath + network_size + '_angle_mag_convergence', dpi=600)

'''
