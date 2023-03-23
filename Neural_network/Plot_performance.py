import numpy as np
import matplotlib.pyplot as plt
from Neural_network.NN_objects import pickle_load_obj as load

path_to_data = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/'
custom_cp_folder = 'cp_small_CustomLoss/'
angle_cp_folder = 'cp_small_LineFlowLossForAngle/'
square_cp_folder = 'cp_small_SquaredLineFlowLoss/'
MSE_cp_folder = 'cp_small_MSE_short/'
name = 'perf_dict.obj'


custom_loss_performance = load(path=path_to_data+custom_cp_folder, filename=name)
angle_loss_performance = load(path=path_to_data+angle_cp_folder, filename=name)
square_loss_performance = load(path=path_to_data+square_cp_folder, filename=name)
mse_loss_short = load(path=path_to_data + MSE_cp_folder, filename=name)

checkpoints = np.arange(1, len(custom_loss_performance) + 1)

averages = [[],[], [], []]
for i in range(len(custom_loss_performance)):
    averages[0].append(custom_loss_performance[i]['overall_average'])
    averages[1].append(angle_loss_performance[i]['overall_average'])
    averages[2].append(square_loss_performance[i]['overall_average'])
    averages[3].append(mse_loss_short[i]['overall_average'])


def inches(cm):
    return cm/2.54


fig_savepath = '/home/clemens/Dropbox/EMIL_MIENERG21/Master/Master/Figures/Results/'

ylim = [0, 6]

plt.plot(checkpoints, averages[0], label='CustomLoss')
plt.plot(checkpoints, averages[1], label='LineFlowLossForAngle')
plt.plot(checkpoints, averages[2], label='SquaredLineFlowLoss')
plt.plot(checkpoints, averages[3], label='MSE')
plt.xlabel('Epoch number')
plt.ylabel('Average percentage accuracy')
plt.gca().set_ylim(ylim)
plt.xticks(np.arange(min(checkpoints), max(checkpoints)+1, 2))
plt.yticks(np.arange(min(ylim), max(ylim)+1, 0.5))
plt.xticks()
plt.legend()
plt.grid()
#plt.rcParams['figure.figsize'] = (inches(12), inches(2))
plt.gcf().set_size_inches(inches(16), inches(10))
plt.savefig(fig_savepath + 'pinn_small_sys_new', dpi=600)