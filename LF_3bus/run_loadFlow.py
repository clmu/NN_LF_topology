

import ElkLoadFlow as elkLF
from build_sys_from_sheet import BuildSystem
import numpy as np

BusList, LineList = BuildSystem('4 bus 1 gen.xls') #import from excel
bus4_dc = elkLF.decoupled_LF(BusList, LineList) #create loadflow object to perform calculations++
bus4 = elkLF.decoupled_LF(BusList, LineList)
accuracy = 0.0001
'''
#non flat start!
bus4_dc.use_regular_nr_data()
bus4_dc.DC_PF(precision=accuracy, max_iterations=2)

bus4.vomag = bus4_dc.vomag
bus4.dc_flag = bus4_dc.dc_flag

bus4.solve_NR(accuracy, max_iterations=10, dc_flag=bus4.dc_flag)
'''

bus4.solve_NR(tolerance=accuracy, max_iterations=15)

allps = bus4.calculate_powers()

def radToDeg(x):
    return x/np.pi*180

allangles = []
for angle in bus4.voang:
    allangles.append(radToDeg(angle))

for bus in range(len(bus4.BusList)):
    print(f'voltage at bus {bus4.BusList[bus].busnum}: {bus4.vomag[bus]}')
    print(f'angle at bus   {bus4.BusList[bus].busnum}: {allangles[bus]}\n')