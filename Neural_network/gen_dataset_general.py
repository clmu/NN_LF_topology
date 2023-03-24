import numpy as np
import time as t

import matplotlib.pyplot as plt
from LF_3bus.ElkLoadFlow import LoadFlow
from LF_3bus.build_sys_from_sheet import BuildSystem
from PyDSAL.DistLoadFlow_v2 import DistLoadFlow3 as dlf
from PyDSAL.BuildSystem_v2 import BuildSystem3



def check_similarity(vec_1, vec_2):
    #if len(vec_1) != len(vec_2):
    #    print('Vectors not of same length')
    #else:
    return vec_1 - vec_2

system_description_folder_large_sys= '/home/clemens/Dropbox/EMIL_MIENERG21/Master/IEEE33bus_69bus/'
medium_sys_filename = 'IEEE33BusDSAL.xls'
large_sys_filename = 'IEEE69BusDSAL.xls'
path_storage_folder = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/datasets/'
filename_medium = 'medium_dataset'
filename_large = 'large_dataset'

m_buses, m_lines = BuildSystem(system_description_folder_large_sys + medium_sys_filename)
l_buses, l_lines = BuildSystem(system_description_folder_large_sys + large_sys_filename)
m_dlf_buses, m_dlf_lines = BuildSystem3(system_description_folder_large_sys + medium_sys_filename)
l_dlf_buses, l_dlf_lines = BuildSystem3(system_description_folder_large_sys + large_sys_filename)

check_line_impedances = np.ones((2, len(l_lines)), dtype=int)
#for line_idx in l_lines:


m_lf = LoadFlow(m_buses, m_lines)
l_lf = LoadFlow(l_buses, l_lines)
m_dlf = dlf(m_dlf_buses, m_dlf_lines)
l_dlf = dlf(l_dlf_buses, l_dlf_lines)

for dlf in [m_dlf, l_dlf]:
    dlf.initialize(startBus=1)

epsilon = 0.00001
m_lf.solve_NR(tolerance=epsilon)
l_lf.solve_NR(tolerance=epsilon)
m_dlf.DistLF(epsilon=epsilon)
l_dlf.DistLF(epsilon=epsilon)

m_vomag_similarity = check_similarity(m_dlf.vomag, m_lf.vomag)
m_voang_similarity = check_similarity(m_dlf.voang, m_lf.voang)

l_vomag_similarity = check_similarity(l_dlf.vomag, l_lf.vomag)
l_voang_similarity = check_similarity(l_dlf.voang, l_lf.voang)

l_buses = np.arange(0, len(l_voang_similarity))
m_buses = np.arange(0, len(m_voang_similarity))

plt.plot(l_buses, l_voang_similarity, label='voang_sim')
plt.plot(l_buses, l_vomag_similarity, label='vomag_sim')
plt.plot(m_buses, m_voang_similarity, label='voang_sim')
plt.plot(m_buses, m_vomag_similarity, label='vomag_sim')

plt.legend()
plt.show()

#m_dlf_minus_lf_vomag =




