'''


This is the main file for my masters project.

--Clemens Müller


'''

import time as t
import DistLoadFlow_v2 as distLF

BusList, LineList = distLF.BuildSystem3()  # Import data from Excel file
#establish DistLoadFlow3 objects.
dlf = distLF.DistLoadFlow3(BusList, LineList)
dlf_nobus44_left = distLF.DistLoadFlow3(BusList, LineList)
dlf_nobus44_right = distLF.DistLoadFlow3(BusList, LineList)

#initialize the loadflow objects.
dlf.initialize(startBus=1)
#dlf_nobus44_left.initialize(startBus=1)
#dlf_nobus44_right.initialize(startBus=62)

#removing the line that connects bus 42 and 44 from the two loadflow objects.
'''
line_to_remove = dlf_nobus44_left.findline(42, 44)
line_to_remove2 = dlf_nobus44_right.findline(42,44)
dlf_nobus44_left.disconnectLine2(line_to_remove)
dlf_nobus44_right.disconnectLine2(line_to_remove2)
'''
#dlf_nobus44_right.remove_unconnected_buses(dlf_nobus44_right.topology, dlf_nobus44_right.find_connected_buses(dlf_nobus44_right.topology, 44))
#This line does not work

#dlf_nobus44.disconnectBus(44)

#dlf_nobus44.initialize(startBus=1)

#Solving the cases.
#dlf.DistLF(epsilon=0.0001)
#dlf.dispGraph(dlf.topology, name='Full_sys')

starttime = t.perf_counter()
bus = dlf.BusList[43]

inj_change = 10

tot_inj = 0
i = 0
converged = False
#while inj_change > 0.001:
while not converged:
    i += 1
    dlf.DistLF(epsilon=0.0001)
    inj_change = bus.dPlossdQ / bus.dP2lossdQ2
    tot_inj += inj_change
    print(f'\nChange in injection at bus {bus.busnum} should be {inj_change}')
    bus.qload -= inj_change
    if bus.dPlossdQ < 0.001:
        converged = True
    #the change in injection should be the negative of the variable inj change since the bus injection is given as a
    #load?

endtime = t.perf_counter()
print(f'Load minimum found in {i} iterations. The reactive injection needed at bus {bus.busnum} is: {tot_inj}\n'
      f'Runtime: {endtime-starttime}s')

'''

dlf_nobus44_left.DistLF(epsilon=0.00001) #solve case 2, note that the dist_lf re_initializes the network
dlf_nobus44_right.DistLF(epsilon=0.00001)


powers_losses = distLF.sum_powers_multiple_sys(dlf_nobus44_left, dlf_nobus44_right)

for i in range(len(powers_losses)):
    powers_losses[i] = round(powers_losses[i], 4)

print(f'The sum of all subsystems loads and losses are:')
print(f'Pload = {powers_losses[0]}, Qload = {powers_losses[1]}, Ploss = {powers_losses[2]}, Qloss{powers_losses[3]}')

print(f'The sum of the unchanged systems losses:')
print(f'Pload = {dlf.system_P}, Qload = {dlf.system_Q}, Ploss = {dlf.system_Ploss}, Qloss{dlf.system_Qloss}')

dlf_nobus44_left.dispGraph(dlf_nobus44_left.topology, name='left_sys')
#dlf_nobus44_right.remove_unconnected_buses(dlf_nobus44_right.topology, dlf_nobus44_right.connectedBuses(dlf_nobus44_right.topology))
dlf_nobus44_right.dispGraph(dlf_nobus44_right.topology, name="right_sys", feeders=[62])

#dlf_nobus44.plotVolt(fromBus=0, toBus=0, v_angle = False, same_plot=False)

#dlf.dispGraph(dlf.topology, feeders=[50], LEC= [31, 48], lowVolt = [52, 53, 65, 64], overload=[(62, 63), (63, 64)])
'''


