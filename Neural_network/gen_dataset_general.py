import numpy as np
import time as t
from LF_3bus.ElkLoadFlow import LoadFlow
from LF_3bus.build_sys_from_sheet import BuildSystem


system_description_folder_large_sys= '/home/clemens/Dropbox/EMIL_MIENERG21/Master/IEEE33bus_69bus/'
medium_sys_filename = 'IEEE33BusDSAL.xls'
large_sys_filename = 'IEEE69BusDSAL.xls'
path_storage_folder = '/home/clemens/PycharmProjects/NN_LF_Topology/Neural_network/datasets/'
filename_medium = 'medium_dataset'
filename_large = 'large_dataset'

m_buses, m_lines = BuildSystem(system_description_folder_large_sys + medium_sys_filename)
l_buses, l_lines = BuildSystem(system_description_folder_large_sys + large_sys_filename)

check_line_impedances = np.ones((2, len(l_lines)), dtype=int)
#for line_idx in l_lines:


m_lf = LoadFlow(m_buses, m_lines)
l_lf = LoadFlow(l_buses, l_lines)

m_lf.solve_NR()
l_lf.solve_NR()




