import matplotlib.pyplot as plt
import numpy as np
import os
from Neural_network.NN_objects import pickle_load_obj as load
from tabulate import tabulate


def print_performance(performance_dict, loss_function_list, size, remark):

    '''

    :param performance_dict: the list of performance dictionaries from which performance is read.
    :param loss_function_list: All loss functions to be included in the plot.
    :param size: network size? (small, medium, large)
    :param remark: Training sesstion specification. ex. '30batch'
    :return: pass. Prints selected performance data after final epoch.
    '''

    thresholds = ['20percent', '10percent', '5percent', '3percent']
    print(f'------------     Performance {size}_{remark}   ----------------')
    data = {}
    table1 = [['', 'Overall errors', 'V mag error', 'V ang error']]
    table2 = [['Threshold']]
    len_loss_fun = len(loss_fun_list)
    for loss in loss_function_list:
        table2[0].append(loss)
        data[loss] = {}
        final_perf_dict = performance_dict[loss][-1]
        data[loss]['avg'] = np.round(final_perf_dict['overall_average'], decimals=2)
        data[loss]['v_avg'] = np.round(final_perf_dict['average_voltage'], decimals=2)
        data[loss]['a_avg'] = np.round(final_perf_dict['average_angle'], decimals=2)
        for threshold in thresholds:
            data[loss][threshold] = np.round(final_perf_dict[threshold]['sets_worse_than_threshold'], decimals=1)

    for threshold in thresholds:
        row = [threshold]
        for i in range(len_loss_fun):
            loss2 = loss_function_list[i]
            row.append(data[loss2][threshold])
        table2.append(row)

    for loss in loss_fun_list:
        table1.append([loss, data[loss]['avg'], data[loss]['v_avg'], data[loss]['a_avg']])

    print(f'\n{tabulate(table1)}\n')
    print(tabulate(table2))

    pass


cwd = os.getcwd()
path = cwd + '/Neural_network/checkpoints/'
network_size = 'medium'
remark = '1e-4_50batch_400_epoch'
#remark = 'baseline_slim''large_30batch' 'large_deep' 'large_low_lrate' #large_wide_baseline'

loss_fun_list = ['MSE', 'CustomLoss', 'SquaredLineFlowLoss']#['MSE']#['MSE' , 'CustomLoss']
performance_data = {}

print(f'Generating performance plot for {network_size} NN, using {remark}')
print(f'Losses to plot: {loss_fun_list}')
print(f'cwd: {cwd}')

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

# To save in proj folder.
fig_savepath = '/home/clemens/Dropbox/EMIL_MIENERG21/Master/Master/Figures/Results/'

ylim = [0, 11]
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
plt.yticks(np.arange(min(ylim), max(ylim)+1, 1))
plt.xticks()
plt.legend()
plt.grid()
plt.gcf().set_size_inches(inches(16), inches(10))
#plt.show()
plt.savefig(fig_savepath + 'pinn_' + network_size + '_' + remark, dpi=600)

print_performance(performance_data, loss_fun_list, network_size, remark)

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
