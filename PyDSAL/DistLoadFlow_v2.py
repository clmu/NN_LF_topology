# Copyright (c) 2021, Olav B. Fosso, NTNU
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import time as t

#import PyDSAL.DistribObjects_v2 as DistribObjects_v2 # import DistribObjects_v2
#from BuildSystem_v2 import *
#import PyDSAL.BuildSystem_v2 as BuildSystem_v2

# Graphics representation
from pyvis.network import Network
import networkx as nx
nt = Network('1000px', '2000px', directed=True, layout="Hierarchcal")
nx_graph = nx.DiGraph(arrows=True)



class DistLoadFlow3:
    """
    Common base class Radial System's (Distribution)  Load Flow
    Input:
        BusList      - List off all Bus objects
        LineList    - List of all transmission lines objects
    Returns: None
    """

    def __init__(self, Buses, Lines):
        self.BusList = Buses
        self.LineList = Lines
        self.voang = np.zeros(len(self.BusList))
        self.vomag = np.ones(len(self.BusList))
        self.topology = []
        self.system_P, self.system_Q = 0, 0
        self.system_Ploss, self.system_Qloss = 0, 0
        self.backup_feeders = []
        self.alteredTopology = False
        self.latest_load_flow_solution_time = None
        #self.initialize(1)
    def config3(self):
        """Function for making the topology - it sets up the connection between two buses by assigned the line to the to bus
        and by preparing a list of from bus connections (branching)
        Problem: Currently turn the direction of too many lines when the connection point splits the chain
        """
        self.clearTopology()
        for lobj in self.LineList:
            if lobj.ibstat:
                itr = lobj.tbus - 1 #tobusindex
                ifr = lobj.fbus - 1 #frombusindex
                self.BusList[ifr].tolinelist.append(lobj)
                self.BusList[
                    itr].toline = lobj  # Add information to each bus of a line abouth which line that connects the meighbour bus.
                self.BusList[ifr].fromline = lobj

        # Add the topology information needed to define the tree structure
        for lobj in self.LineList:
            if lobj.ibstat:
                itr = lobj.tbus - 1
                ifr = lobj.fbus - 1
                self.BusList[ifr].nextbus.append(
                    self.BusList[itr])  # Add the next bus to the list of branches of the bus

    def findtree(self, bstart=1):
        """ Finds a trestructure from a spesified node
            The from and two nodes are switched to get a positive flow direction.
        """
        def mswitch(ifrom, ito):  # To switch direction
            return ito, ifrom

        def direct(bindex, val =None):  # Recursive function for topology search and direction of a graph.
            ibus = self.BusList[bindex]
            for lobj in lineconnlist[bindex]:
                if lobj.tbus == ibus.busnum:
                    lobj.fbus, lobj.tbus = mswitch(lobj.fbus, lobj.tbus)
                if lobj not in lineconnlist[lobj.tbus - 1]:
                    print('Grid is not radial')
                    return False
                lineconnlist[lobj.tbus - 1].remove(
                    lobj)  # When a line is checked remove the object from the lineconnlist of the to-bus
                val = direct(lobj.tbus - 1)
                if val == False:
                    return val

        lineconnlist = []  # Define and initialize with sublists
        iloop = 0
        while iloop < len(self.BusList):
            lineconnlist.append([])
            iloop += 1

        # Find lines connected to all buses
        for lobj in self.LineList:
            if lobj.ibstat:
                itr = lobj.tbus - 1
                ifr = lobj.fbus - 1
                lineconnlist[ifr].append(lobj)
                lineconnlist[itr].append(lobj)

        # Build a tree structure
        ibus = self.BusList[bstart - 1]  # Identify the bus object to start with
        valid = direct(bstart - 1)
        return valid

    # Flat start
    def flatStart(self):
        iloop = 0
        while iloop < len(self.BusList):
            ibus = self.BusList[iloop]
            ibus.vomag = 1.0
            ibus.voang = 0.0
            ibus.ploadds = 0.0
            ibus.qloadds = 0.0
            ibus.pblossds = 0.0
            ibus.qblossds = 0.0
            iloop += 1

    # Set up a list for the main branch, where subbranches are stored as sublists. Handles all radial topologies
    def mainstruct4(self, startBus=None):
        """
        An algorithm to establish a tree structure based on the system data. Sets up a list for the main branch,
        with sublists wherever branching occurs. The algorithm can handle any radial topology, but not meshed grids.
        """
        if startBus is None:
            startBus = self.BusList[0]
        else:
            startBus = self.BusList[startBus - 1]
        mainlist = []                               # Make the main branch
        nextobj = [startBus]                        # Set next object to the first bus
        while len(nextobj) > 0:                     # Until we reach the end of the main branch
            if len(nextobj) == 1:                   # If no branch is present, add the bus to main branch
                mainlist.append(nextobj)
            if len(nextobj) > 1:
                mainlist.append([nextobj[0]])       # If branches occur, add the root bus to the main branch
                for i in range(1, len(nextobj)):    # Go through each sub branch
                    bra = self.branch4(nextobj, i)  # Make sub branches
                    mainlist[-2].append(bra)        # Add sub branch to the root bus
            nextobj = mainlist[-1][0].nextbus       # Set next bus to the next in main branch
        return mainlist

    def branch4(self, nextobj, i):
        """
        A recursive algorithm to follow every branch until the end. In case of sub branches, the algorithm calls itself.
        """
        sub = [[nextobj[i]]]                            # Make the sub branch, and add the first bus
        nextobj = sub[-1][0].nextbus                    # Set next bus to the first of the branch ----- there is no content here after removing the unconnected lines.
        while len(nextobj) > 0:                         # Follow until the end of the sub branch
            if len(nextobj) == 1:                       # If no further branching, add to sub branch
                sub.append(nextobj)
            if len(nextobj) > 1:
                sub.append([nextobj[0]])                # If further branching, add root of branch to sub branch
                for j in range(1, len(nextobj)):
                    subsub = self.branch4(nextobj, j)   # Go through each subsub branch(recursive step)
                    sub[-2].append(subsub)              # Add possible subsub branches
            nextobj = sub[-1][0].nextbus                # Set next bus to next bus in sub branch
        return sub

    # Return the buses connected to the grid
    def connectedBuses(self, topologyList):
        """
        The function returns a list of all buses connected to the grid.
        """
        buses = []
        for x in topologyList:
            if len(x) > 1:
                buses.append(x[0])
                iloop = 1
                while iloop < len(x):  # Do for all branches of a bus
                    am = self.connectedBuses(x[iloop])
                    for i in range(0, len(am)):
                        buses.append(am[i])
                    iloop += 1
            else:
                buses.append(x[0])
        return buses

    # Clear topology to start new configuration of the grid
    def clearTopology(self):
        """
        The function clears all topology parameters to ensure correct configuration when the system is altered.
        """
        for bus in self.BusList:
            bus.connectedLines = []
            bus.tolinelist = []
            bus.toline = 0
            bus.fromline = 0
            bus.nextbus = []

    # Connect a line
    def connectLine2(self, line):
        """
        Connects a line. Can take a line object or a line index as input.
        """
        lineindex = 0
        if type(line) is Line:
            lineindex = self.LineList.index(line)
        if type(line) is int:
            lineindex = line
        self.LineList[lineindex].ibstat = 1
        print('Connected line between bus ' + str(self.LineList[lineindex].fbus) + ' and ' + str(
            self.LineList[lineindex].tbus))
        self.alteredTopology = True #Changes marker for altered topology to true

    # Disconnect a line
    def disconnectLine2(self, line):
        """
        Disconnects a line. Can take a line object or a line index as input.
        """
        lineindex = 0
        if type(line) is Line:
            lineindex = self.LineList.index(line)
        if type(line) is int:
            lineindex = line
        self.LineList[lineindex].ibstat = 0
        print('Disconnected line between bus ' + str(self.LineList[lineindex].fbus) + ' and ' +
              str(self.LineList[lineindex].tbus))
        self.alteredTopology = True

    #Disconnect a bus
    def disconnectBus(self, busnum):
        """
        The functions disconnects a bus from the system by disconnecting all lines connected to it, and resetting
        the voltage magnitude and angle.
        """
        bind = busnum - 1
        bus = self.BusList[bind]
        self.disconnectLine2(self.LineList.index(bus.toline))
        self.BusList[bind].toline = 0
        for lobj in bus.tolinelist:
            self.disconnectLine2(self.LineList.index(lobj))
        self.BusList[bind].vomag = 0.0
        self.BusList[bind].voang = 0.0
        self.alteredTopology = True

    # Disconnect all overloaded buses
    def disconnectBuses(self, buses):
        """
        The function goes through a list of buses and disconnects them.
        """
        for bus in buses:
            self.disconnectBus(bus.busnum)
        print('Disconnected bus : ', [o.busnum for o in buses])
        self.alteredTopology = True

    # Check is any buses have too high or too low voltage
    def checkOverLoad(self):
        """
        The function goes through the list of buses to check for overloaded buses.
        Returns: All buses that have been overloaded
        """
        overloaded = []
        for bus in self.BusList:
            if bus.vmax < bus.vomag:
                overloaded.append(bus)
        if len(overloaded) > 0:
            print('Overload found at bus: ', [o.busnum for o in overloaded])
        return overloaded

    #Reset buses not in the topology
    def resetBuses(self):
        """
        Sets the voltage magnitude and angle of all buses not connected to the grid to zero for display purposes.
        """
        top = self.connectedBuses(self.topology)
        for bus in self.BusList:
            if bus not in top:
                bus.vomag = 0.0
                bus.voang = 0.0

    #Change power consumption at a bus
    def changePower(self, busnum, delta):
        """
        Function for altering the power injection or consumption at a bus.
        """
        self.BusList[busnum - 1].pload += delta

    #Checks for overflow on all lines
    def checkOverflow(self,rmult=1.0):
        """
        Function for checking for any overflows on any lin ein the system.
        """
        print('Checking for overflow on all lines:')
        found = 0
        for line in self.LineList:
            if line.ratea != 0:
                def uij(gij, bij, tetai, tetaj):
                    return gij * np.sin(tetai - tetaj) - bij * np.cos(tetai - tetaj)

                def tij(gij, bij, tetai, tetaj):
                    return gij * np.cos(tetai - tetaj) + bij * np.sin(tetai - tetaj)

                def bij(R, X):
                    return (1.0 / complex(R, X)).imag

                def gij(R, X):
                    return (1.0 / complex(R, X)).real

                ifr = line.fbus - 1
                itr = line.tbus - 1
                bsh = 0.0  # No shunts included so far
                teta1 = self.BusList[ifr].voang
                teta2 = self.BusList[itr].voang
                v1 = self.BusList[ifr].vomag
                v2 = self.BusList[itr].vomag
                b = bij(line.r, line.x)
                g = gij(line.r, line.x)

                Pfrom = g * v1 * v1 - v1 * v2 * tij(g, b, teta1, teta2)
                Pto = g * v2 * v2 - v1 * v2 * tij(g, b, teta2, teta1)
                Qfrom = -(b + bsh) * v1 * v1 - v1 * v2 * uij(g, b, teta1, teta2)
                Qto = -(b + bsh) * v2 * v2 - v1 * v2 * uij(g, b, teta2, teta1)
                Sfrom = math.sqrt(Pfrom**2 + Qfrom**2)
                Sto = math.sqrt(Pto ** 2 + Qto ** 2)
                if Sfrom > line.ratea*rmult or Sto > line.ratea*rmult:
                    print('Overflow found at line between bus: ', line.tbus, ' and ', line.fbus)
                    found += 1
        if found == 0:
            print('All line flows are within the limits.')

    #Get the potential voltage regulation at a bus
    def potential(self, bus):
        """
        Finds the maximum possible potential for voltage regulation at a bus.
        """
        # Get sensitivities
        sensP = bus.dVdP * (1.0 + bus.dPlossdP)
        sensQ = bus.dVdQ * (1.0 + bus.dQlossdQ)

        # Get available compensation
        compP = 0
        compQ = 0
        if bus.comp:
            compQ = self.SVCDroopCrtl(bus)  # Droop-based representation
        if bus.pv:
            pvobj = bus.pv
            if pvobj.cmode == 2:
                pvobj.qinj = self.PVDroopCrtl(bus) * bus.controlScale
            compP += pvobj.injPmax
            compQ += pvobj.injQmax
        if bus.battery:
            pvobj = bus.battery
            if pvobj.cmode == 2:
                pvobj.qinj = self.BatteryDroopCrtl(bus) * bus.controlScale
            compP += pvobj.injPmax
            compQ += pvobj.injQmax
        if bus.v2g:
            pvobj = bus.v2g
            if pvobj.cmode == 2:
                pvobj.qinj = self.V2GDroopCrtl(bus) * bus.controlScale
            compP += pvobj.injPmax
            compQ += pvobj.injQmax

        # Find the possible voltage regulation available
        vComp = sensP * compP + sensQ * compQ
        vComp = - vComp
        return vComp

    #Find out the needed change in power injection at a bus to correct a voltage mismatch
    def neededInjection(self, busnum, actOrReact=None):
        """
        The functions finds the needed power injection needed at a bus to get the voltage back within its limits.
        """
        bus = self.BusList[busnum - 1]
        deltaV = 0.0
        inj = 0.0
        typ = actOrReact
        sens = 0
        if typ == 'active':
            sens = bus.dVdP * (1.0 + bus.dPlossdP)
        elif typ == 'reactive':
            sens = bus.dVdQ * (1.0 + bus.dQlossdQ)
        increase = 0
        decrease = 0
        if bus.vomag > bus.vmax:
            deltaV = bus.vomag - bus.vmax
            print('The bus voltage at bus', bus.busnum, ' needs to be lowered by ', deltaV)
            inj = deltaV / sens
            if sens < 0:
                increase = 1
            if sens > 0:
                decrease = 1
        elif bus.vomag < bus.vmin:
            deltaV = bus.vmin - bus.vomag
            print('The bus voltage at bus ', bus.busnum, ' needs to be increased by ', deltaV)
            inj = deltaV / sens
            if sens < 0:
                decrease = 1
            if sens > 0:
                increase = 1
        else:
            print('The bus voltage at bus ', bus.busnum, ' is within its range')
        if increase:
            print(typ, ' power injection at bus ', bus.busnum, ' must be increased by ', abs(inj))
        if decrease:
            print(typ, ' power injection at bus ', bus.busnum, ' must be decreased by ', abs(inj))
        return inj


    def neededInjectionLine2(self, line):
        """
        Finds the needed active power injection needed at a bus in case of an overflow on a line.
        """
        lobj = None
        if type(line) is Line:
            lobj = line
        if type(line) is int:
            lobj = self.LineList[line]

        def getDelta(lobj1):
            def uij(gij, bij, tetai, tetaj):
                return gij * np.sin(tetai - tetaj) - bij * np.cos(tetai - tetaj)

            def tij(gij, bij, tetai, tetaj):
                return gij * np.cos(tetai - tetaj) + bij * np.sin(tetai - tetaj)

            ifr = lobj1.fbus - 1
            itr = lobj1.tbus - 1
            teta1 = self.BusList[ifr].voang
            teta2 = self.BusList[itr].voang
            v1 = self.BusList[ifr].vomag
            v2 = self.BusList[itr].vomag
            b = (1.0 / complex(lobj1.r, lobj1.x)).imag
            g = (1.0 / complex(lobj1.r, lobj1.x)).real

            Pfrom = g * v1 * v1 - v1 * v2 * tij(g, b, teta1, teta2)
            Qfrom = -b * v1 * v1 - v1 * v2 * uij(g, b, teta1, teta2)
            Sfrom1 = math.sqrt(Pfrom**2 + Qfrom**2)
            deltaS1 = Sfrom1 - lobj.ratea
            if deltaS1 ** 2 > Qfrom ** 2:                               #Extra check to make it compile even if it is within its limit.
                neededP = math.sqrt(deltaS1 ** 2 - Qfrom ** 2)
            else:
                neededP = None
            return deltaS1, neededP

        deltaS, neededP = getDelta(lobj)
        if deltaS <= 0:
            print('Line flow between bus ', lobj.fbus, ' and ', lobj.tbus, ' is within limits.')
            return 0.0
        if deltaS > 0:
            print('Line flow on line between bus ', lobj.fbus, ' and ', lobj.tbus, ' must be lowered by ', deltaS)
            if neededP is None:
                print('The line flow cannot be corrected solely by active injection at bus ', lobj.tbus)
            else:
                print('Active injection at bus ', lobj.tbus, ' can be increased by ', neededP)
        return deltaS, neededP


    # Handle an overload
    def handleOverload(self, overloaded):
        """
        Function to handle an overload at one or several buses. Disconnects them, and tries to connect the reserve
        lines present in the system. Finds the reserve line that connects the most buses and results in the lowest
        losses.
        """
        self.disconnectBuses(overloaded)
        print('Trying different topologies to find a solution: \n')
        reserve = []
        connected = None
        for line in self.LineList:
            if line.reserve == 1:
                reserve.append(line)
        plossmin = 10000
        numbus = 0
        for line in reserve:
            self.connectLine2(line)
            self.config3()
            mesh = self.findtree()
            if mesh is None:
                self.config3()
                self.topology = dlf.mainstruct4() #self.mainstruct4() ??
                p1, q1, p2, q2 = self.accload(self.topology, self.BusList)
                connectedbuses = self.connectedBuses(self.topology)
                if len(connectedbuses) >= numbus:
                    if p2 < plossmin:
                        numbus = len(connectedbuses)
                        plossmin = p2
                        connected = line
            self.disconnectLine2(line)
        if plossmin < 10000:
            self.connectLine2(connected)
            self.config3()
            mesh = self.findtree()
            self.config3()
            self.topology = dlf.mainstruct4() # self.mainstruct4()
            print('\nNetwork was altered due to an overload at bus: ' + str([o.busnum for o in overloaded]) + '\n' +
                  'Network was altered by connecting line: ' + str(self.LineList.index(connected)) + ' between bus: ' +
                  str(connected.tbus) + ' and ' + str(connected.fbus))
            top = self.connectedBuses(self.topology)
            print('Number of buses connected: ', len(top))
            self.resetBuses()
            print('New Load Flow Solution: \n')
            dlf.DistLF(epsilon=0.00001) #self.distLF(epsilon=0.00001) ??
        if plossmin == 10000:
            print('No alternative topology could be found to alter the network and still have a radial network')

    # Display transmission line flows
    def dispFlow(self, fromLine=0, toLine=0, tpres=False):
        """ Display the flow on the requested distribution lines
        """

        mainlist = []
        rowno = []

        def uij(gij, bij, tetai, tetaj):
            return gij * np.sin(tetai - tetaj) - bij * np.cos(tetai - tetaj)

        def tij(gij, bij, tetai, tetaj):
            return gij * np.cos(tetai - tetaj) + bij * np.sin(tetai - tetaj)

        def bij(R, X):
            return (1.0 / complex(R, X)).imag

        def gij(R, X):
            return (1.0 / complex(R, X)).real

        if toLine == 0:
            toLine = len(self.LineList)
        if tpres:
            toLine = np.minimum(fromLine + 13, toLine)

        if fromLine < len(self.LineList):
            inum = fromLine
        else:
            print('Line :', fromLine, ' does not exist')
            return()

        for line in self.LineList[fromLine:toLine]:
            ifr = line.fbus - 1
            itr = line.tbus - 1
            bsh = 0.0  # No shunts included so far
            teta1 = self.BusList[ifr].voang
            teta2 = self.BusList[itr].voang
            v1 = self.BusList[ifr].vomag
            v2 = self.BusList[itr].vomag
            b = bij(line.r, line.x)
            g = gij(line.r, line.x)

            Pfrom = g * v1 * v1 - v1 * v2 * tij(g, b, teta1, teta2)
            Pto = g * v2 * v2 - v1 * v2 * tij(g, b, teta2, teta1)
            Qfrom = -(b + bsh) * v1 * v1 - v1 * v2 * uij(g, b, teta1, teta2)
            Qto = -(b + bsh) * v2 * v2 - v1 * v2 * uij(g, b, teta2, teta1)
            # Update structures
            line.flowfromP = Pfrom
            line.flowfromQ = Qfrom
            line.flowtoP = Pto
            line.flowtoQ = Qto

            if not tpres:
                print(' FromBus :', '{:4.0f}'.format(ifr + 1), ' ToBus :', '{:4.0f}'.format(itr + 1),
                      ' Pfrom :', '{:7.4f}'.format(Pfrom), ' Qfrom : ', '{:7.4f}'.format(Qfrom),
                      ' Pto :', '{:7.4f}'.format(Pto), ' Qto :', '{:7.4f}'.format(Qto))

            sublist = [ifr + 1, itr + 1, '{:7.4f}'.format(Pfrom), '{:7.4f}'.format(Qfrom),
                       '{:7.4f}'.format(Pto), '{:7.4f}'.format(Qfrom)]
            mainlist.append(sublist)
            rowno.append('Line ' + str(inum))
            inum += 1

        if tpres:
            title = 'Transmission line flow'
            colind = ['FromBus', ' ToBus', 'Pfrom', ' Qfrom', ' Pto', ' Qto']
            self.tableplot(mainlist, title, colind, rowno, columncol=[], rowcol=[])

    # Conduct a distribution system load flow based on FBS


    def copy_voltages_to_dlf_obj(self):
        for bus_idx in range(len(self.BusList)):
            self.voang[bus_idx] = self.BusList[bus_idx].voang
            self.vomag[bus_idx] = self.BusList[bus_idx].vomag

        pass


    def DistLF(self, epsilon=0.0001, print_solution=False):

        """ Solves the distribution load flow until the convergence criteria is met for all buses.
        The two first steps are to set up additions topology information and to build the main structure
        Next,it is switched between forward sweeps(Voltage updates) and backward sweeps(load update and loss calcuation)
        """

        #Checking for altered topology, if topology has been altered since initiation, initiate again.

        if self.alteredTopology:
            '''
            Check if topology has been altered through line or bus connections/disconnections.
            re-initializes the system from the same bus as it was initialized from originally. 
            The new initialization is from the first bus in the topology list. Ensures that no changes are made to the
            current topology. 
            '''
            self.initialize(self.topology[0][0].busnum)
            self.alteredTopology = False

# Flat start option has to be considered

        start_solution_time = t.perf_counter()
        self.flatStart()

        diff = 10
        iloop = 0
        while diff > epsilon:
            self.system_P, self.system_Q, self.system_Ploss, self.system_Qloss = self.accload(self.topology, self.BusList)
            if print_solution:
                print('Iter: ', iloop + 1, 'Pload:', '{:7.4f}'.format(self.system_P), 'Qload:', '{:7.4f}'.format(self.system_Q),
                  'Ploss:', '{:7.4f}'.format(self.system_Ploss), 'Qloss:', '{:7.4f}'.format(self.system_Qloss))

            oldVs = []
            for i in range(0, len(self.BusList)):
                oldVs.append(self.BusList[i].vomag)
            self.UpdateVolt(self.topology, self.BusList)
            newVs = []
            iloop += 1
            if iloop > 15:
                #print('Convergence could not be reached.')
                raise StopIteration('Convergence could not be reached')
                #return None
            diffs = []
            for i in range(0, len(self.BusList)):
                newVs.append(self.BusList[i].vomag)
                diffs.append(abs(oldVs[i] - newVs[i]))
            diff = max(diffs)
#        overload = self.checkOverLoad()
#        if len(overload) > 0:
#            self.handleOverload(overload)
        self.latest_load_flow_solution_time = t.perf_counter() - start_solution_time
        self.copy_voltages_to_dlf_obj()
        if print_solution:
            print('\n', "****** Load flow completed in ", iloop, " iterations ******", '\n')

    # Visit all nodes in the reverse list.
    def BackwardSearch(self, topologyList):
        """ Visit all the nodes in a backward approach and prints the Bus name
        """
        for x in reversed(topologyList):
            if len(x) > 1:
                print('Bus' + str(x[0].busnum))
                iloop = 1
                while iloop < len(x):  # Do for all branches of a bus
                    self.BackwardSearch(x[iloop])
                    iloop += 1
            else:
                print('Bus' + str(x[0].busnum))

    # Visit all nodes in the forward list.
    def ForwardSearch(self, topologyList):
        """Visit all nodes in a forward approach and prints the Bus name
        """
        for x in topologyList:
            if len(x) > 1:
                print('Bus' + str(x[0].busnum))
                iloop = 1
                while iloop < len(x):  # Do for all branches of a bus
                    self.ForwardSearch(x[iloop])
                    iloop += 1
            else:
                print('Bus' + str(x[0].busnum))

    # Visit all nodes in the forward list.
    def BuildGraph(self, topologyList,top=1, feeders=[], LEC=[], charging=[], lowVolt=[]):
        """Visit all nodes in a forward approach and build the graphic representation
        """
        #  from pyvis.network import Network
        #  nt = Network('1000px', '2000px', layout=None)
        def adaptNode(node, top=1, feeders=[], LEC=[], charging=[],lowVolt=[]):
            if node == top:
                nx_graph.add_node(node, label="Main feeder", color='green')
            elif node in feeders:
                nx_graph.add_node(node,label= "Bck feeder", color='green')
            elif node in LEC:
                nx_graph.add_node(node,label= "LEC", color='#FF33F9')
            elif node in charging:
                nx_graph.add_node(node,label= "Charging", color='purple')
            elif node in lowVolt:
                nx_graph.add_node(node,label= "Low Volt", color='yellow')
            else:
                nx_graph.add_node(node, label=str(x[0].busnum)) #remove label to show the busnum
   #     print(feeders, LEC, lowVolt)
        for x in topologyList:
            if len(x) > 1:
                # print('Bus' + str(x[0].busnum))
            #    nt.add_node(int(x[0].busnum))
                adaptNode(int(x[0].busnum), top=1, feeders=feeders, LEC=LEC, charging=charging, lowVolt=lowVolt)
                iloop = 1
                while iloop < len(x):  # Do for all branches of a bus
                    self.BuildGraph(x[iloop], top=1, feeders=feeders, LEC=LEC, charging=charging, lowVolt=lowVolt)
                    iloop += 1
            else:
                #   print('Bus' + str(x[0].busnum))
              #  nt.add_node(int(x[0].busnum))
                 adaptNode(int(x[0].busnum), top=1, feeders=feeders, LEC=LEC, charging=charging, lowVolt=lowVolt)

   # Visit all nodes in the reverse list.
    def AddEdges(self, topologyList, overload=[], disconnected=[]):
        """ Visit all the nodes in a backward approach and prints the Bus name
        """
        def adaptEdge(node1, node2,  overload=[], disconnected=[]):
            if (node1, node2) in overload:
                nx_graph.add_edge(node1, node2, color='red', value=2)
            elif (node1,node2) in disconnected:
                nx_graph.add_edge(node1, node2, color='brown', value=2)
            else:
                nx_graph.add_edge(node1, node2, color=" #33AFFF", arrow=True)

        for x in reversed(topologyList):
            if len(x) > 1:
              #  print('Bus' + str(x[0].busnum))
                if x[0].toline:
         #           nt.add_edge(int(x[0].toline.fbus),int(x[0].busnum))
                    adaptEdge(int(x[0].toline.fbus), int(x[0].busnum),  overload=overload, disconnected=disconnected)

                iloop = 1
                while iloop < len(x):  # Do for all branches of a bus
                    self.AddEdges(x[iloop], overload=overload, disconnected=disconnected)

                    iloop += 1
            else:
         #       print('Bus' + str(x[0].busnum))
                if x[0].toline:
                #    nt.add_edge(int(x[0].toline.fbus), int(x[0].busnum))
                    adaptEdge(int(x[0].toline.fbus), int(x[0].busnum),  overload=overload, disconnected=disconnected)

    def dispGraph(self, topologyList, name = None,top=1, feeders=[], LEC=[], charging=[], lowVolt=[], overload= [],disconnected=[]):
        """
        Builds and display the graph as a HTML-file
        """
        if name == None:
            name = 'nt.html'
        else:
            name = str(name) + '.html'
        self.BuildGraph(topologyList,top=top,feeders=feeders, LEC=LEC, charging=charging, lowVolt=lowVolt )
        self.AddEdges(topologyList, overload=overload,disconnected=disconnected)
        nt.from_nx(nx_graph)
        nt.show(name)
        #nt.show(namehtml)

    # Calculations the load for the actual voltage at the bus
    def getload(self, busobj):
        """ Calculates the net voltage corrected load at the bus - currently a simple ZIP model is applied.
        Input: The busobject
        Returns: pLoadAct, qLoadAct
        """
        #        if busobj.vset > 0:
        #        self.voltCrtl(busobj)
        qmod = 0.0
        pmod = 0.0
        # Include all possible local sources (SVC/Statcom, PV, Battery and V2G)
        if busobj.comp:
            qmod = self.SVCDroopCrtl(busobj)  # Droop-based representation
        if busobj.pv:
            pvobj = busobj.pv
            if pvobj.cmode == 2:
                pvobj.qinj = self.PVDroopCrtl(busobj) * busobj.controlScale
            pmod += pvobj.pinj
            qmod += pvobj.qinj
        if busobj.battery:
            pvobj = busobj.battery
            if pvobj.cmode == 2:
                pvobj.qinj = self.BatteryDroopCrtl(busobj) * busobj.controlScale
            pvobj = busobj.battery
            pmod += pvobj.pinj
            qmod += pvobj.qinj
        if busobj.v2g:
            pvobj = busobj.v2g
            if pvobj.cmode == 2:
                pvobj.qinj = self.V2GDroopCrtl(busobj) * busobj.controlScale
            pmod += pvobj.pinj
            qmod += pvobj.qinj
        # Find the net load at the node (Note: load - injection)
        pLoadAct = busobj.pload * (
                busobj.ZIP[0] * busobj.vomag ** 2 + busobj.ZIP[1] * busobj.vomag + busobj.ZIP[2]) - pmod
        qLoadAct = busobj.qload * (
                busobj.ZIP[0] * busobj.vomag ** 2 + busobj.ZIP[1] * busobj.vomag + busobj.ZIP[2]) - qmod
        dPdV = busobj.pload * (busobj.ZIP[0] * 2 * busobj.vomag + busobj.ZIP[1])
        dQdV = busobj.qload * (busobj.ZIP[0] * 2 * busobj.vomag + busobj.ZIP[1])
        return pLoadAct, qLoadAct, dPdV, dQdV

    def voltCrtl(self, busobj, mode='Reactive'):
        """ Changes the net injection at voltage controlled buses
                Input: The busobject
                mode - Control mode ('Active', 'Reactive', 'Both' - default = 'Reactive')
                Returns: pLoadAct, qLoadAct
                """
        if busobj.vset > 0 and busobj.vomag < 1.0:
            if np.abs(busobj.vomag - busobj.vset) > 0.0002:
                if mode == 'Active':
                    deltap = (busobj.vset - busobj.vomag) / (busobj.dVdP * (1 + busobj.dPlossdP))
                    busobj.pload += deltap
                    print('Load corr (Active): ', busobj.busnum, deltap, busobj.pload)
                elif mode == 'Reactive':
                    deltaq = (busobj.vset - busobj.vomag) / (busobj.dVdQ * (1 + busobj.dQlossdQ))
                    busobj.qload += deltaq
                    print('Load corr (Reactive): ', busobj.busnum, deltaq, busobj.qload)

    def PVDroopCrtl(self, busobj):
        """Calculates the PV/converter contribution to voltage control"""
        pvobj = busobj.pv
        if pvobj.stat:
            qsens = busobj.dVdQ * (1.0 + busobj.dQlossdQ)
            if qsens:
                a = 1.0
                b = -(pvobj.vprev + pvobj.slopeQ / qsens)
                c = pvobj.slopeQ / qsens * busobj.vomag
                v = (-b + np.sqrt(b ** 2 - 4 * a * c)) / 2.0
                v2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / 2.0
                # print('v1 :', v, '   v2 : ', v2)
            else:
                v = pvobj.vprev
            #        v = (busobj.vomag - qsens*svcobj.vprev/svcobj.slopeQ)/(1.0 - qsens/svcobj.slopeQ)
            Qc = -1.0 / pvobj.slopeQ * v * (v - pvobj.vref)
            pvobj.vprev = v
            print(busobj.busname, '     Volt: ', v, '   Qinj = ', Qc)
            pvobj.qinj = Qc
            return Qc

    def V2GDroopCrtl(self, busobj):
        """Calculates the PV/converter contribution to voltage control"""
        v2gobj = busobj.v2g
        if v2gobj.stat:
            qsens = busobj.dVdQ * (1.0 + busobj.dQlossdQ)
            if qsens:
                a = 1.0
                b = -(v2gobj.vprev + v2gobj.slopeQ / qsens)
                c = v2gobj.slopeQ / qsens * busobj.vomag
                v = (-b + np.sqrt(b ** 2 - 4 * a * c)) / 2.0
                v2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / 2.0
            # print('v1 :', v, '   v2 : ', v2)
            else:
                v = v2gobj.vprev
            #        v = (busobj.vomag - qsens*svcobj.vprev/svcobj.slopeQ)/(1.0 - qsens/svcobj.slopeQ)
            Qc = -1.0 / v2gobj.slopeQ * v * (v - v2gobj.vref)
            v2gobj.vprev = v
            print(busobj.busname, '     Volt: ', v, '   Qinj = ', Qc)
            v2gobj.qinj = Qc
            return Qc

    def BatteryDroopCrtl(self, busobj):
        """Calculates the PV/converter contribution to voltage control"""
        batobj = busobj.battery
        if batobj.stat:
            qsens = busobj.dVdQ * (1.0 + busobj.dQlossdQ)
            if qsens:
                a = 1.0
                b = -(batobj.vprev + batobj.slopeQ / qsens)
                c = batobj.slopeQ / qsens * busobj.vomag
                v = (-b + np.sqrt(b ** 2 - 4 * a * c)) / 2.0
                v2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / 2.0
            # print('v1 :', v, '   v2 : ', v2)
            else:
                v = batobj.vprev
            #        v = (busobj.vomag - qsens*svcobj.vprev/svcobj.slopeQ)/(1.0 - qsens/svcobj.slopeQ)
            Qc = -1.0 / batobj.slopeQ * v * (v - batobj.vref)
            batobj.vprev = v
            print(busobj.busname, '     Volt: ', v, '   Qinj = ', Qc)
            batobj.qinj = Qc
            return Qc

    def SVCDroopCrtl(self, busobj):
        """Calculates the SVC contribution to voltage control"""
        svcobj = busobj.comp
        if svcobj.stat:
            qsens = busobj.dVdQ * (1.0 + busobj.dQlossdQ)
            if qsens:
                a = 1.0
                b = -(svcobj.vprev + svcobj.slopeQ / qsens)
                c = svcobj.slopeQ / qsens * busobj.vomag
                v = (-b + np.sqrt(b ** 2 - 4 * a * c)) / 2.0
                v2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / 2.0
                print('v1 :', v, '   v2 : ', v2)
            else:
                v = svcobj.vprev
            #        v = (busobj.vomag - qsens*svcobj.vprev/svcobj.slopeQ)/(1.0 - qsens/svcobj.slopeQ)
            Qc = -1.0 / svcobj.slopeQ * v * (v - svcobj.vref)
            svcobj.vprev = v
            print(busobj.busname, '     Volt: ', v, '   Qinj = ', Qc)
            svcobj.qinj = Qc
            return Qc

    def SVCCrtl2(self, busobj):
        """Calculates the SVC contribution to voltage control"""
        svcobj = busobj.comp
        if svcobj.stat:
            qsens = busobj.dVdQ * (1.0 + busobj.dQlossdQ)
            v = (busobj.vomag - qsens * svcobj.vprev / svcobj.slopeQ) / (1.0 - qsens / svcobj.slopeQ)
            Qc = -1.0 / svcobj.slopeQ * (v - svcobj.vref)
            svcobj.vprev = v
            print(busobj.busname, '     Volt: ', v, '   Qinj = ', Qc)
            svcobj.qinj = Qc
            return Qc

    # Calculate the accumulated load and losses starting on the last node
    def accload(self, topologyList, BusList):
        """Calculates the accumulated downstream active and reactive load at all buses
        and calculates the active and reactive losses of lines and make an accumulated equivalent load at the buses
        """
        pl1 = 0.0 #Total power load
        ql1 = 0.0# total reactive power
        ploss1 = 0.0 #Total losses
        qloss1 = 0.0 #total loss

        for x in reversed(topologyList):  # Start on last node
            if len(x) > 1:
                iloop = 1
                while iloop < len(x):  # Do for all branches at a bus
                    pl2, ql2, ploss2, qloss2 = self.accload(x[iloop], BusList)
                    pl1 += pl2  # Add accumulated powers and losses in a branch to the node where the brancing accurs.
                    ql1 += ql2
                    ploss1 += ploss2
                    qloss1 += qloss2
                    iloop += 1
                pla, qla, dPdV1, dQdV1 = self.getload(x[0])  # Add local loads
                pl1 += pla  # Add local loads
                ql1 += qla
                x[0].ploadds = pl1  # Add accumulated descriptions to the branching node
                x[0].qloadds = ql1
                x[0].pblossds = ploss1
                x[0].qblossds = qloss1
                if pl1 != 0:
                    x[0].dPdV = (x[0].dPdV * (pl1 - pla) + dPdV1 * pla) / pl1
                if ql1 != 0:
                    x[0].dQdV = (x[0].dQdV * (ql1 - qla) + dQdV1 * qla) / ql1
                if x[0].toline:  # Follow the next node in the main path
                    lobj = x[0].toline
                    if lobj.ibstat:
                        ifr = lobj.fbus
                        itr = lobj.tbus
                        pto = x[0].ploadds + x[0].pblossds  # Find the flow to the downstream bus
                        qto = x[0].qloadds + x[0].qblossds
                        lobj.ploss = lobj.r * (pto ** 2 + qto ** 2) / x[
                            0].vomag ** 2  # Estimate the losses of the branch
                        lobj.qloss = lobj.x * (pto ** 2 + qto ** 2) / x[0].vomag ** 2
                        ploss1 += lobj.ploss
                        qloss1 += lobj.qloss
                        x[0].pblossds = ploss1  # Add the losses to the downstream bus
                        x[0].qblossds = qloss1

            else:  # No branching at the bus
                #                pl1 += x[0].pload
                #                ql1 += x[0].qload
                pla, qla, dPdV1, dQdV1 = self.getload(x[0])
                pl1 += pla  # Add local loads
                ql1 += qla
                x[0].ploadds = pl1
                x[0].qloadds = ql1
                if pl1 != 0:
                    x[0].dPdV = (x[0].dPdV * (pl1 - pla) + dPdV1 * pla) / pl1
                if ql1 != 0:
                    x[0].dQdV = (x[0].dQdV * (ql1 - qla) + dQdV1 * qla) / ql1
                if x[0].toline:
                    lobj = x[0].toline
                    if lobj.ibstat:
                        ifr = lobj.fbus
                        itr = lobj.tbus
                        pto = x[0].ploadds + ploss1
                        qto = x[0].qloadds + qloss1
                        lobj.ploss = lobj.r * (pto ** 2 + qto ** 2) / x[0].vomag ** 2
                        lobj.qloss = lobj.x * (pto ** 2 + qto ** 2) / x[0].vomag ** 2
                        ploss1 += lobj.ploss
                        qloss1 += lobj.qloss
                        x[0].pblossds = ploss1
                        x[0].qblossds = qloss1

        return pl1, ql1, ploss1, qloss1  # Return the accumulated loads and losses from the current branch

    # Update the control scaling factors
    def UpdateControl(self, BusList):
        """ Updates the scaling factors used for Voltage and Minimum loss purposes
            Identifies number of control units on adjacent buses and updates the scaling used to improve convergence

            May be extended later
        """

        iloop = 0
        while iloop < len(BusList):
            iunit = 0
            inext = BusList[iloop].tolinelist
            # print(len(inext))
            if len(inext) > 1:
                for iloop2 in inext:  # Find the number of buses
                    itr = iloop2.tbus
                    if BusList[itr - 1].iloss == 1:
                        iunit += 1
                        print(BusList[itr - 1].busname)
                if iunit > 1:  # Update the scaling factors
                    for iloop2 in inext:
                        itr = iloop2.tbus
                        if BusList[itr - 1].iloss == 1:
                            BusList[
                                itr - 1].controlScale = 1.2 / iunit  # Use: 1.0 (default), 0.6, 0.4 and 0.3 depending on the number of control buses

            iloop += 1
        # End

    # Update the voltage profile starting on the top node
    def UpdateVolt(self, topologyList, BusList):
        """Update the voltage profile based on the accumulated load on each bus
        """

        # Function for calculating the voltages and sensitivities in the single phase case (modified sensitivity calculation)
        def nodeVoltSensSPv2(BusList, ifr, itr, tline, obj):
            """
            Calculate the node voltages and sensitivities in the single phase case - a more accurate sensitivity calculation (had just minor impact)
            :param BusList:
            :param ifr:
            :param itr:
            :param tline:
            :param obj:
            :return:
            """

            vk2 = BusList[ifr].vomag ** 2
            tpload = obj[0].ploadds + obj[0].pblossds  # Find the accumulated loads and losses flowing on the branch
            tqload = obj[0].qloadds + obj[0].qblossds
            # Voltage calculation
            term2 = 2 * (tpload * tline.r + tqload * tline.x)
            term3 = (tpload ** 2 + tqload ** 2) * (tline.r ** 2 + tline.x ** 2) / BusList[ifr].vomag ** 2
            BusList[itr].vomag = np.sqrt(
                vk2 - term2 + term3)  # Update the bus voltage magnitude on the down-stream bus
            # Calculate the sensitivities for changing the load
            #   dvdp = (-tline.r + tpload * (tline.r ** 2 + tline.x ** 2) / BusList[ifr].vomag ** 2) / BusList[
            #       itr].vomag

            dqdp = (2 * tline.x * tpload / BusList[itr].vomag ** 2) * (
                    1 + 2 * tline.x * tqload / BusList[
                itr].vomag ** 2)  # The relation between the chang in q for a change in p - simplified version to get a better dvdp

            dvdp = (-tline.r - tline.x * dqdp + (tpload + tqload * dqdp) * (tline.r ** 2 + tline.x ** 2) / BusList[
                ifr].vomag ** 2) / BusList[
                       itr].vomag

            dpdq = (2 * tline.r * tqload / BusList[itr].vomag ** 2) * (
                    1 + 2 * tline.r * tpload / BusList[
                itr].vomag ** 2)  # The relation between the change in p for a change in q - simplified version
            #  dvdq = (-tline.x + tqload * (tline.r ** 2 + tline.x ** 2) / BusList[ifr].vomag ** 2) / BusList[
            #     itr].vomag
            dvdq = (-tline.x - tline.r * dpdq + (tqload + tpload * dpdq) * (tline.r ** 2 + tline.x ** 2) / BusList[
                ifr].vomag ** 2) / BusList[
                       itr].vomag

            # dqdp = (2 * tline.x * tpload / BusList[itr].vomag ** 2) * (
            #         1 + 2 * tline.x * tqload / BusList[
            #     itr].vomag ** 2)  # The relation between the chang in q for a change in p
            dqdp = ((2 * tline.x * tqload + 2 * tline.x * tpload * dpdq) * BusList[itr].vomag ** 2 - (
                    tline.x * tpload ** 2 + tline.x * tqload ** 2) * 2 * BusList[itr].vomag * dvdp) / BusList[
                       itr].vomag ** 4

            dpdq = ((2 * tline.r * tqload + 2 * tline.r * tpload * dqdp) * BusList[itr].vomag ** 2 - (
                    tline.r * tpload ** 2 + tline.r * tqload ** 2) * 2 * BusList[itr].vomag * dvdq) / BusList[
                       itr].vomag ** 4

            #  dpldp = (2 * tline.r * tpload / BusList[itr].vomag ** 2) * (
            #          1 + 2 * tline.x * tqload / BusList[itr].vomag ** 2)  # Change in losses for a change in p
            dpldp = ((2 * tline.r * tpload + 2 * tline.r * tqload * dqdp) * BusList[itr].vomag ** 2 - (
                    tline.r * tpload ** 2 + tline.r * tqload ** 2) * 2 * BusList[itr].vomag * dvdp) / BusList[
                        itr].vomag ** 4

            BusList[itr].dVdP = BusList[ifr].dVdP + dvdp + dvdq * dqdp
            BusList[itr].dVdQ = BusList[ifr].dVdQ + dvdq + dvdp * dpdq
            # Calculate sensitivities for change in losses
            BusList[itr].dPlossdP = BusList[ifr].dPlossdP + dpldp
            BusList[itr].dPlossdQ = BusList[ifr].dPlossdQ + dpdq
            BusList[itr].dQlossdP = BusList[ifr].dQlossdP + dqdp
            #                    BusList[itr].dQlossdQ = BusList[ifr].dQlossdQ + (2 * tline.x * tqload/BusList[itr].vomag**2) * (1 + 2 * tline.r * tpload/BusList[itr].vomag**2)
            BusList[itr].dQlossdQ = BusList[ifr].dQlossdQ + 2 * tline.x * tqload / BusList[
                itr].vomag ** 2 + 2 * tline.x * tpload * BusList[itr].dPlossdQ / BusList[itr].vomag ** 2
            # Calculate the second-order derivatives
            if tqload == 0:
                term1q = 0
            else:
                term1q = dpdq / tqload
            BusList[itr].dP2lossdQ2 = BusList[ifr].dP2lossdQ2 + term1q + (
                    2 * tline.r * tqload / BusList[itr].vomag ** 2) * 2 * tline.r * dpdq / BusList[itr].vomag ** 2

            if tpload == 0:
                term1p = 0
            else:
                term1p = dpldp / tpload
            BusList[itr].dP2lossdP2 = BusList[ifr].dP2lossdQ2 + term1p + (
                    2 * tline.r * tpload / BusList[itr].vomag ** 2) * 2 * tline.x * dqdp / BusList[itr].vomag ** 2
            # Estimate the required injection to reach minimum loss
            BusList[itr].lossRatioQ = BusList[itr].dPlossdQ / BusList[itr].dP2lossdQ2  # Check this one
            BusList[itr].lossRatioP = BusList[itr].dPlossdP / BusList[itr].dP2lossdP2

            # Update the voltage for the purpose of loss minimization - adjust the sensitivity acording to the chosen step.
            if BusList[itr].iloss:
                #    if np.abs(BusList[itr].dPlossdQ) >= 1.0 / BusList[
                #        itr].pqcostRatio:  # Equivalent to that the dP cost more than pqcostRatio times dQ
                qcomp = BusList[itr].dPlossdQ / (
                        BusList[itr].dP2lossdQ2 - 1.0)  # Estimate the toerethically required adjustment

                # BusList[itr].dPlossdQ = 0.0 # In general case we should find better solution

                # Assign the correction to the right source and scale according to the choosen strategy
                if BusList[itr].pv:
                    pvobj = BusList[itr].pv
                    if pvobj.cmode == 1:  # Update only ot the cmode = 1 - NB Other objects may be added under this section when iloss = 1
                        pvobj.qinj += qcomp * BusList[itr].controlScale
                        BusList[itr].dPlossdQ = 0.0  # In general case we should find better solution
                if BusList[itr].battery:
                    pvobj = BusList[itr].battery
                    if pvobj.cmode == 1:  # Update only ot the cmode = 1 - NB Other objects may be added under this section when iloss = 1
                        pvobj.qinj += qcomp * BusList[itr].controlScale
                        BusList[itr].dPlossdQ = 0.0  # In general case we should find better solution
                if BusList[itr].v2g:
                    pvobj = BusList[itr].v2g
                    if pvobj.cmode == 1:  # Update only ot the cmode = 1 - NB Other objects may be added under this section when iloss = 1
                        pvobj.qinj += qcomp * BusList[itr].controlScale
                        BusList[itr].dPlossdQ = 0.0  # In general case we should find better solution

            # Voltage angle calculation
            busvoltreal = BusList[ifr].vomag - (tpload * tline.r + tqload * tline.x) / BusList[ifr].vomag
            busvoltimag = (tqload * tline.r - tpload * tline.x) / BusList[ifr].vomag
            BusList[itr].voang = BusList[ifr].voang + np.arctan2(busvoltimag, busvoltreal)  # Update voltage angles
            return

        #  End

        for obj in topologyList:
            if len(obj) > 1:

                if obj[0].toline:
                    tline = obj[0].toline
                    ifr = tline.fbus - 1
                    itr = tline.tbus - 1

                    # Update voltages and sensitivities Single Phase
                    nodeVoltSensSPv2(BusList, ifr, itr, tline, obj)

                iloop = 1
                while iloop < len(obj):  # Update voltages along the branches
                    self.UpdateVolt(obj[iloop], BusList)
                    iloop += 1
            else:  # Continue along the current path
                if obj[0].toline:
                    tline = obj[0].toline
                    ifr = tline.fbus - 1
                    itr = tline.tbus - 1

                    # Update voltages and sensitivities Single Phase
                    nodeVoltSensSPv2(BusList, ifr, itr, tline, obj)

    # Estimate the losses of each line based on voltage level and accumulated flow
    def lossEstimate(self, busobjects, lineobjects):
        """Estimates the losses of each line based on voltage level and accumulated flow
        """
        for lobj in reversed(lineobjects):
            ifr = lobj.fbus - 1
            itr = lobj.tbus - 1
            pto = busobjects[itr].ploadds
            qto = busobjects[itr].qloadds
            lobj.ploss = lobj.r * (pto ** 2 + qto ** 2) / busobjects[itr].vomag ** 2
            lobj.qloss = lobj.x * (pto ** 2 + qto ** 2) / busobjects[itr].vomag ** 2
            busobjects[ifr].ploadds += lobj.ploss
            busobjects[ifr].qloadds += lobj.qloss

    # Display the voltages.
    def dispVolt(self, fromBus=0, toBus=0, tpres=False):
        """
        Desc:    Display voltages at all buses
        Input:   tpres= False (Display in tableformat if True)
                 fromBus and toBus defines the block, If tpres=True, it will display 13 lines from fromBus
        Returns: None
        """
        mainlist = []
        rowno = []
        if toBus == 0:
            toBus = len(self.BusList)
        if tpres:
            toBus = np.minimum(fromBus + 13, toBus)

        iloop = fromBus
        print(' ')
        while iloop < toBus:
            oref = self.BusList[iloop]
            if not tpres:
                print(' Bus no :', '{:4.0f}'.format(oref.busnum),
                      ' Vmag :', '{:7.5f}'.format(oref.vomag),
                      ' Theta :', '{:7.5f}'.format(oref.voang * 180 / np.pi))
            # Prepare for graphics presentation
            sublist = ['{:4.0f}'.format(oref.busnum),
                       '{:7.5f}'.format(oref.vomag),
                       '{:7.5f}'.format(oref.voang * 180 / np.pi)]

            mainlist.append(sublist)
            rowno.append('Bus ' + str(iloop + 1))
            iloop += 1
        # Present table
        if tpres:
            title = 'Bus Voltages'
            colind = [' Bus no ', ' Vmag ', ' Theta ']
            colw = [0.12, 0.22, 0.22, 0.22, 0.22]
            self.tableplot(mainlist, title, colind, rowno, columncol=[], rowcol=[], colw=colw)

    def plotVolt(self, fromBus = 0, toBus = 0, v_angle = False, same_plot = True, data_output = False, color1 = 'red', color2 = 'green'):

        x_axis = []
        y_axis = []

        if toBus == 0:
            toBus = len(self.BusList)
        elif toBus != 0:
            toBus += 0

        inloop = fromBus
        while inloop < toBus:
            the_bus = self.BusList[inloop]
            if v_angle == False:
                y_axis.append(the_bus.vomag)
            else:
                y_axis.append([the_bus.vomag, the_bus.voang])
            x_axis.append(the_bus.busnum)

            inloop += 1

        if v_angle:
            voltage_magnitude = []
            voltage_angle = []
            for element in y_axis:
                voltage_angle.append(element[1])
                voltage_magnitude.append(element[0])

            if not same_plot:
                fig, ax = plt.subplots(2, 1, figsize=(12, 6))

                # ax[0].set_xlabel('Bus index no.')
                ax[0].set_ylabel('Voltage magnitude')
                ax[0].plot(x_axis, voltage_magnitude, color=color1, label='|V|')
                ax[0].tick_params(axis='y')
                # plt.legend('upper right')
                # ax2 = ax1.twinx()
                ax[1].set_xlabel('Bus index no.')
                ax[1].set_ylabel('Voltage angle')
                ax[1].plot(x_axis, voltage_angle, color=color2, label='vAng')
                ax[1].tick_params(axis='y')
                # plt.legend()
            else:
                fig, ax1 = plt.subplots(figsize=(10, 4))
                v_mag = ax1.plot(x_axis, voltage_magnitude, label='|V|', color=color1)
                ax2 = ax1.twinx()
                v_ang = ax2.plot(x_axis, voltage_angle, label='vAng', color=color2)
                ax1.set_xlabel('Bus index no.')
                ax1.set_ylabel('Voltage magnitude')
                ax2.set_ylabel('Voltage angle [rad]')

                plots = v_mag + v_ang
                labels = [l.get_label() for l in plots]
                plt.legend(plots, labels)
        else:
            fig, ax = plt.subplots(figsize = (10, 2.5))
            ax.plot(x_axis, y_axis, label='|V|', color=color1)
            ax.set_xlabel('Bus index no.')
            ax.set_ylabel('Voltage magnitude')

        plt.show()
        if data_output:
            return x_axis, y_axis


    # Display the voltages.
    def dispLowVolt(self, fromBus=0, toBus=0, tpres=False, vmax=1.1):
        """
        Desc:    Display voltages at all buses below or equal to the limit vmax
        Input:   tpres= False (Display in tableformat if True)
                 fromBus and toBus defines the block, If tpres=True, it will display 13 lines from fromBus
                 vmax = Upper voltage limit (default 1.1 pu)
        Returns: None
        """
        mainlist = []
        rowno = []
        if toBus == 0:
            toBus = len(self.BusList)
        #        if tpres:
        #            toBus = np.minimum(fromBus + 13, toBus)

        if fromBus < len(self.BusList): # Check legal range
            iloop = fromBus
        else:
            print(' Bus :', fromBus, ' does not exist')
            return()

        print(' ')
        while iloop < toBus:
            oref = self.BusList[iloop]
            if oref.vomag <= vmax:
                if not tpres:
                    print(' Bus no :', '{:4.0f}'.format(oref.busnum),
                          ' Vmag :', '{:7.5f}'.format(oref.vomag),
                          ' Theta :', '{:7.5f}'.format(oref.voang * 180 / np.pi))
                # Prepare for graphics presentation
                sublist = ['{:4.0f}'.format(oref.busnum),
                           '{:7.5f}'.format(oref.vomag),
                           '{:7.5f}'.format(oref.voang * 180 / np.pi)]

                mainlist.append(sublist)
                rowno.append('Bus ' + str(iloop + 1))
            iloop += 1
        # Present table
        if tpres:
            title = 'Bus Voltages'
            colind = ['Bus no', 'Vmag', 'Theta']
            colw = [0.12, 0.22, 0.22, 0.22, 0.22]
            self.tableplot(mainlist, title, colind, rowno, columncol=[], rowcol=[], colw=colw)

    # Display the voltages.
    def dispVoltRange(self, fromBus=0, toBus=0, tpres=False, vmin=0.9, vmax=1.1):
        """
        Desc:    Display voltages at all buses below or equal to the limit vmax
        Input:   tpres= False (Display in tableformat if True)
                 fromBus and toBus defines the block, If tpres=True, it will display 13 lines from fromBus
                 vmax = Upper voltage limit (default 1.1 pu)
                 vmin = Lower voltage limit (defualt 0.9 pu)
        Returns: None
        """
        mainlist = []
        rowno = []
        if toBus == 0:
            toBus = len(self.BusList)
        #        if tpres:
        #            toBus = np.minimum(fromBus + 13, toBus)

        if fromBus < len(self.BusList): # Check legal range
            iloop = fromBus
        else:
            print(' Bus :', fromBus, ' does not exist')
            return()

        print(' ')
        while iloop < toBus:
            oref = self.BusList[iloop]
            if vmax >= oref.vomag >= vmin:
                if not tpres:
                    print(' Bus no :', '{:4.0f}'.format(oref.busnum),
                          ' Vmag :', '{:7.5f}'.format(oref.vomag),
                          ' Theta :', '{:7.5f}'.format(oref.voang * 180 / np.pi))
                # Prepare for graphics presentation
                sublist = ['{:4.0f}'.format(oref.busnum),
                           '{:7.5f}'.format(oref.vomag),
                           '{:7.5f}'.format(oref.voang * 180 / np.pi)]

                mainlist.append(sublist)
                rowno.append('Bus ' + str(iloop + 1))
            iloop += 1
        # Present table
        if tpres:
            title = 'Bus Voltages'
            colind = ['Bus no', 'Vmag', 'Theta']
            colw = [0.12, 0.22, 0.22, 0.22, 0.22]
            self.tableplot(mainlist, title, colind, rowno, columncol=[], rowcol=[], colw=colw)

    # Display voltage estimate for a changes in active or reactive load on a bus
    def dispVoltEst(self, bus=0, deltap=0.0, deltaq=0.0, tpres=False):
        """ The method estimates the voltages for a change in active or reactive load at a bus
        deltap and deltaq must reflect the change (negative by load reduction)
        """
        itr = bus - 1
        mainlist = []
        rowno = []
        iloop = 0
        while self.BusList[itr].toline:
            busobj = self.BusList[itr]
            voltest = busobj.vomag + deltap * (1 + busobj.dPlossdP) * busobj.dVdP + deltaq * (
                    1 + busobj.dQlossdQ) * busobj.dVdQ
            if not tpres:
                print(' Bus no :', '{:4.0f}'.format(busobj.busnum),
                      ' Vmag :', '{:7.4f}'.format(busobj.vomag),
                      ' Vest :', '{:7.4f}'.format(voltest))
            # Prepare for graphics presentation
            if iloop < 14:
                sublist = ['{:4.0f}'.format(busobj.busnum),
                           '{:7.4f}'.format(busobj.vomag),
                           '{:7.4f}'.format(voltest)]
                mainlist.append(sublist)
                rowno.append('Bus ' + str(iloop + 1))
            iloop += 1
            itr = busobj.toline.fbus - 1

        # Present table
        if tpres:
            title = 'Voltage estimat for changed injection of P and Q'
            colind = ['Bus no', 'Bus volt', 'Volt est']
            colw = [0.12, 0.22, 0.22, 0.22, 0.22]
            self.tableplot(mainlist, title, colind, rowno, columncol=[], rowcol=[], colw=colw)

    # Display the voltage sensitivities
    def dispVoltSens(self, fromBus=0, toBus=0, tpres=False):
        """
        Desc:    Display Load sensitivities for change in voltage at all buses
        Input:   tpres= False (Display in table format if True)
                 fromBus and toBus defines the block, If tpres=True, it will display 13 lines from fromBus
        Returns: None
        """
        mainlist = []
        rowno = []
        if toBus == 0:
            toBus = len(self.BusList)
        if tpres:
            toBus = np.minimum(fromBus + 13, toBus)

        if fromBus < len(self.BusList): # Check legal range
            iloop = fromBus
        else:
            print(' Bus :', fromBus, ' does not exist')
            return()

        print(' ')
        while iloop < toBus:
            oref = self.BusList[iloop]
            if not tpres:
                print(' Bus no :', '{:4.0f}'.format(oref.busnum),
                      ' dV/dP :', '{:7.5}'.format(oref.dVdP * (1.0 + oref.dPlossdP)),
                      ' dPloss/dP :,{:7.5}'.format(oref.dPlossdP),
                      ' dPloss/dQ :,{:7.5}'.format(oref.dPlossdQ),
                      ' dV/dQ :', '{:7.5}'.format(oref.dVdQ * (1.0 + oref.dPlossdQ)),
                      ' dQloss/dQ :,{:7.5}'.format(oref.dQlossdQ),
                      ' dQloss/dP :,{:7.5}'.format(oref.dQlossdP))

            # Prepare for graphics presentation
            sublist = ['{:4.0f}'.format(oref.busnum),
                       '{:7.5}'.format(oref.dVdP * (1.0 + oref.dPlossdP)),
                       '{:7.5}'.format(oref.dPlossdP),
                       '{:7.5}'.format(oref.dPlossdQ),
                       '{:7.5}'.format(oref.dVdQ * np.sqrt((1.0 + oref.dPlossdQ) ** 2 + oref.dQlossdQ ** 2)),
                       '{:7.5}'.format(oref.dQlossdQ),
                       '{:7.5}'.format(oref.dQlossdP)
                       ]

            mainlist.append(sublist)
            rowno.append('Bus ' + str(iloop + 1))
            iloop += 1
        # Present table
        if tpres:
            title = 'Bus Voltage sensitivites to changes in load and loss'
            colind = ['Bus no', 'dV/dP', 'dPloss/dP', 'dPloss/dQ', 'dV/dQ', 'dQloss/dQ', 'dQloss/dP']
            self.tableplot(mainlist, title, colind, rowno, columncol=[], rowcol=[])

    # Display loss sensitivities
    def dispLossSens(self, fromBus=0, toBus=0, tpres=False):
        """
        Desc:    Display Loss sensitivities for change in active or reactive injection at all buses
        Input:   tpres= False (Display in tableformat if True)
                 fromBus and toBus defines the block, If tpres=True, it will display 13 lines from fromBus
        Returns: None
        """
        mainlist = []
        rowno = []
        if toBus == 0:
            toBus = len(self.BusList)
        if tpres:
            toBus = np.minimum(fromBus + 13, toBus)

        if fromBus < len(self.BusList): # Check legal range
            iloop = fromBus
        else:
            print(' Bus :', fromBus, ' does not exist')
            return()

        print(' ')
        while iloop < toBus:
            oref = self.BusList[iloop]
            if not tpres:
                print(' Bus no :', '{:4.0f}'.format(oref.busnum),
                      ' dV/dP :', '{:7.5}'.format(oref.dVdP * (1.0 + oref.dPlossdP)),
                      ' dPloss/dP :,{:7.5}'.format(oref.dPlossdP),
                      ' dPloss/dQ :,{:7.5}'.format(oref.dPlossdQ),
                      ' dV/dQ :', '{:7.5}'.format(oref.dVdQ * (1.0 + oref.dQlossdQ)),
                      ' dP2loss/dP2 :,{:7.5}'.format(oref.dP2lossdP2 - 1.0),
                      ' dP2loss/dQ2 :,{:7.5}'.format(oref.dP2lossdQ2 - 1.0))

            # Prepare for graphics presentation
            sublist = ['{:4.0f}'.format(oref.busnum),
                       '{:7.5}'.format(oref.dVdP * (1.0 + oref.dPlossdP)),
                       '{:7.5}'.format(oref.dPlossdP),
                       '{:7.5}'.format(oref.dPlossdQ),
                       '{:7.5}'.format(oref.dVdQ * (1.0 + oref.dQlossdQ)),
                       '{:7.5}'.format(oref.dP2lossdP2 - 1.0),
                       '{:7.5}'.format(oref.dP2lossdQ2 - 1.0)
                       ]

            mainlist.append(sublist)
            rowno.append('Bus ' + str(iloop + 1))
            iloop += 1
        # Present table
        if tpres:
            title = 'Bus Voltage sensitivites to changes in load and loss'
            colind = ['Bus no', 'dV/dP', 'dPloss/dP', 'dPloss/dQ', 'dV/dQ', 'd2Ploss/dP2', 'd2Ploss/dQ2']
            self.tableplot(mainlist, title, colind, rowno, columncol=[], rowcol=[])

    # Display loss sensitivities for active power injection
    def dispLossSensP(self, fromBus=0, toBus=0, tpres=False):
        """
        Desc:    Display Loss sensitivities for change in active or reactive injection at all buses
        Input:   tpres= False (Display in tableformat if True)
                 fromBus and toBus defines the block, If tpres=True, it will display 13 lines from fromBus
        Returns: None
        """
        mainlist = []
        rowno = []
        if toBus == 0:
            toBus = len(self.BusList)
        if tpres:
            toBus = np.minimum(fromBus + 13, toBus)

        if fromBus < len(self.BusList): # Check legal range
            iloop = fromBus
        else:
            print(' Bus :', fromBus, ' does not exist')
            return()

        print(' ')
        while iloop < toBus:
            oref = self.BusList[iloop]
            if not tpres:
                print(' Bus no :', '{:4.0f}'.format(oref.busnum),
                      ' dV/dP :', '{:7.5}'.format(oref.dVdP * (1.0 + oref.dPlossdP)),
                      ' dPloss/dP :,{:7.5}'.format(oref.dPlossdP),
                      ' dP2loss/dP2 :,{:7.5}'.format(oref.dP2lossdP2 - 1.0),
                      ' Loss Ratio P :,{:7.5}'.format(oref.lossRatioP))

            # Prepare for graphics presentation
            sublist = ['{:4.0f}'.format(oref.busnum),
                       '{:7.5}'.format(oref.dVdP * (1.0 + oref.dPlossdP)),
                       '{:7.5}'.format(oref.dPlossdP),
                       '{:7.5}'.format(oref.dP2lossdP2 - 1.0),
                       '{:7.5}'.format(oref.lossRatioP)
                       ]

            mainlist.append(sublist)
            rowno.append('Bus ' + str(iloop + 1))
            iloop += 1
        # Present table
        if tpres:
            title = 'Bus Voltage sensitivites to changes in load and loss'
            colind = ['Bus no', 'dV/dP', 'dPloss/dP', 'd2Ploss/dP2',
                      'Loss Ratio P']
            colw = [0.12, 0.22, 0.22, 0.22, 0.22]
            self.tableplot(mainlist, title, colind, rowno, columncol=[], rowcol=[], colw=colw)

    # Display loss sensitivities for reactive power injections
    def dispLossSensQ(self, fromBus=0, toBus=0, tpres=False):
        """
        Desc:    Display Loss sensitivities for change in active or reactive injection at all buses
        Input:   tpres= False (Display in tableformat if True)
                 fromBus and toBus defines the block, If tpres=True, it will display 13 lines from fromBus
        Returns: None
        """
        mainlist = []
        rowno = []
        if toBus == 0:
            toBus = len(self.BusList)
        if tpres:
            toBus = np.minimum(fromBus + 13, toBus)

        if fromBus < len(self.BusList): # Check legal range
            iloop = fromBus
        else:
            print(' Bus :', fromBus, ' does not exist')
            return()

        print(' ')
        while iloop < toBus:
            oref = self.BusList[iloop]
            if not tpres:
                print(' Bus no :', '{:4.0f}'.format(oref.busnum),
                      ' dV/dQ :', '{:7.5}'.format(oref.dVdQ * (1.0 + oref.dQlossdQ)),
                      ' dPloss/dQ :,{:7.5}'.format(oref.dPlossdQ),
                      ' dP2loss/dQ2 :,{:7.5}'.format(oref.dP2lossdQ2 - 1.0),  # 1.0 ref value
                      ' Loss Ratio Q :,{:7.5}'.format(oref.lossRatioQ))

            # Prepare for graphics presentation
            sublist = ['{:4.0f}'.format(oref.busnum),
                       '{:7.5}'.format(oref.dVdQ * (1.0 + oref.dQlossdQ)),
                       '{:7.5}'.format(oref.dPlossdQ),
                       '{:7.5}'.format(oref.dP2lossdQ2 - 1.0),
                       '{:7.5}'.format(oref.lossRatioQ)
                       ]

            mainlist.append(sublist)
            rowno.append('Bus ' + str(iloop + 1))
            iloop += 1
        # Present table
        if tpres:
            title = 'Bus Voltage sensitivites to changes in load and loss'
            colind = ['Bus no', 'dV/dQ', 'dPloss/dQ', 'd2Ploss/dQ2',
                      'Loss Ratio Q']
            colw = [0.12, 0.22, 0.22, 0.22, 0.22]
            self.tableplot(mainlist, title, colind, rowno, columncol=[], rowcol=[], colw=colw)

    # General table controlled by the application
    def tableplot(self, table_data, title, columns, rows, columncol=None, rowcol=None, colw=None):
        """
        Desc:   Make a table of the provided data. There must be a row and a column
                data correpsonding to the table
        Input:  table_data  - np.array
                title - string
                columns - string vector
                rows    - string vector
                columncol - colors of each column label (default [])
                rowcol - colors of each row lable
        """

        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(1, 1, 1)

        tdim = np.shape(table_data)
        iloop = 0
        if rowcol == []:
            while iloop < tdim[0]:
                rowcol.append('cyan')
                iloop += 1
        iloop = 0
        if columncol == []:
            while iloop < tdim[1]:
                columncol.append('cyan')
                iloop += 1

        table = ax.table(cellText=table_data, rowLabels=rows, colColours=columncol,
                         rowColours=rowcol, colLabels=columns, colWidths=colw, loc='center', cellLoc='center')
        table.set_fontsize(11)
        table.set_animated = True
        #        table.scale(1,1.5)
        table.scale(1, 1.5)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        plt.show()

    # Display total losses
    def dispLosses(self):
        pline = 0.0
        qline = 0.0
        for x in self.LineList:
            pline += x.ploss
            qline += x.qloss
        print('\n', 'Ploss:', pline, '   Qloss:', qline)

    # Display total load (no voltage correction)
    def dispLoad(self):
        aload = 0.0
        rload = 0.0
        for x in self.BusList:
            pla, qla, dPdV1, dQdV1 = self.getload(x)
            aload += pla  # Add local loads
            rload += qla
        print('\n', 'Total load  P: ', aload, '   Q: ', rload, '  Losses: P',
              BusList[1].pblossds, '   Q: ', BusList[1].qblossds)

    def zeroxq(self):
        for a in self.LineList:
            a.x = 0.0
        for a in self.BusList:
            a.qload = 0.0
    # Prepare the case
    def initialize(self, startBus):
        """
        Builds the system, creates the loafd flow object and prepared the additional configuration
        """
        self.flatStart()    # be sure to have a flat start with topology changes
        self.config3()  # Set up additional configuration based on input data
        self.findtree(startBus)     # Identify the tree from the given starting point
     #   if startBus != 1:
        self.config3()      # Update based on the the starting point
        self.topology = self.mainstruct4(startBus = startBus)   # Build the structure

    def findline(self, fbus, tbus):
        lines = []
        for line in self.LineList:
            if line.fbus == fbus and line.tbus == tbus or line.tbus == fbus and line.fbus == tbus:
                lines.append(line)
        if len(lines) == 1:
            return lines[0]
        if len(lines) > 1:
            return lines
        else:
            print(f'There is no line connecting buses {fbus} and {tbus}')
        pass

    def remove_unconnected_buses(self, topology_list, connected_buses):
        for x in topology_list:
            if len(x) == 1 and type(x[0]) == type(self.BusList[0]):
                if x[0] not in connected_buses:
                    #self.BusList.remove(x[0])
                    index = self.BusList.index(x[0])
                    self.BusList.pop(index)
            elif len(x) == 1 and type(x[0]) == type([]):
                if x[0][0] not in connected_buses:
                    #self.BusList.remove(x[0][0])
                    index = self.BusList.index(x[0][0])
                    self.BusList.pop(index)
            else:
                if x[0] not in connected_buses:
                    #self.BusList.remove(x[0])
                    index = self.BusList.index(x[0])
                    self.BusList.pop(index)
                self.remove_unconnected_buses(x[1], connected_buses)
        #create topology list again to remove the unconnected buses from the topology list.
        #item = self.topology[0][0].busnum #to test the datatype of the element.
        #self.topology = self.mainstruct4(startBus=self.topology[0][0].busnum)
        self.alteredTopology = True #to ensure a re-initialization before load flow calculation ++
        pass

    def find_connected_buses(self, topology_list, disconnected_bus):
        '''
        Function returns a list of all bused connected to the system with a line.
        :param topology_list: Topology list of the system.
        :param disconnected_bus: the bus with the highest busnmber at the disconnection. ie the receiving bus.
        :return: list of all buses connected to the system with a line.

        function bugs:
            not compatible with distLF
            Seems to mess up the sequence in the bus list.
        '''
        conn_bus = []
        for x in reversed(topology_list):
            if len(x) > 1:
                if x[0].busnum >= disconnected_bus:
                    conn_bus.append(x[0])
                iloop = 1
                while iloop < len(x):
                    conn_bus.extend(self.find_connected_buses(x[iloop], disconnected_bus))
                    iloop += 1
            else:
                if x[0].busnum >= disconnected_bus:
                    conn_bus.append(x[0])
        return conn_bus

    def find_backup_feeders(self):
        pass



class analyse_systems:
    def __init__(self, *systems):
        '''
        All systems given to the class analyse systems are taken in and made into a tuple.
        :param systems: list of system partitioning that contains a description of all 124 buses.
        '''
        self.systems_tuple = systems




def evaluate_topology():
    '''
    Return the sum of all loads in the system.
    It should also be an available network configuration such that it is possible to operate the network.
    :return:
    '''

    pass



def sum_powers_multiple_sys(*systems):
    '''
    This function takes an arbitrary amount of loadflow objects and returns the sum of powers and losses.
    useful to evaluate the performance of different system partitioning alternatives.
    The *system generates a tuple containing all the input elements.
    :param systems: load flow objects. The function takes as many load flow objects as it receives.
    :return: [The sum of all subsystems]: p, q, ploss, and qloss
    '''
    p = 0
    q = 0
    ploss = 0
    qloss = 0
    for system in systems:
        p += system.system_P
        q += system.system_Q
        ploss += system.system_Ploss
        qloss += system.system_Qloss
    return [p, q, ploss, qloss]

def compare_volt_profiles(dlf1, dlf2, fromBus = 0, toBus = 0, color1 = 'red', color2 = 'green'):
    x_axis = []
    voltages = []
    angles = []

    if toBus == 0:
        toBus = len(dlf1.BusList)

    inloop = fromBus
    while inloop < toBus:
        the_bus1 = dlf1.BusList[inloop]
        the_bus2 = dlf2.BusList[inloop]
        x_axis.append(the_bus1.busnum)
        voltages.append([the_bus1.vomag, the_bus2.vomag])
        angles.append([the_bus1.voang, the_bus2.vomag])
        inloop += 1

    voltages_to_plot = [[],[]]
    angles_to_plot = [[],[]]

    for i in range(len(voltages)):
        voltages_to_plot[0].append(voltages[i][0])
        voltages_to_plot[1].append(voltages[i][1])
        angles_to_plot[0].append(angles[i][0])
        angles_to_plot[1].append(angles[i][1])



    fig, ax = plt.subplots(2,1, figsize = (10, 8))
    ax[0].set_ylabel('Voltage magnitude')
    vomag1 = ax[0].plot(x_axis, voltages_to_plot[0],label = '|V| - dlf1', color = color1)
    vomag2 = ax[0].plot(x_axis, voltages_to_plot[1], label = '|V| - dlf2', color = color2)
    ax[1].set_ylabel('Voltage angle [rad]')
    voang1 = ax[1].plot(x_axis, angles_to_plot[0], label='vAng - dlf1', color=color1)
    voang2 = ax[1].plot(x_axis, angles_to_plot[1], label='vAng - dlf2', color=color2)

    plots = vomag1 + vomag2 + voang1 + voang2
    labels = [plot.get_label() for plot in plots]
    plt.legend(plots,labels)
    plt.show()
    pass

#
# Demo case (Illustration of how to build up a script)
#
# BusList, LineList = BuildSystem3()  # Import data from Excel file
# dlf = DistLoadFlow3(BusList, LineList)  # Create object
# dlf.config3()  # Set up additional configuration
# dlf.findtree(1)
# #svc = DistribObjects3.SVC(dlf.BusList[43], svcstat=1, vref=1, injQmax=1.0, injQmin=0.0, slopeQ=0.05 )
# #dlf.BusList[43].comp = svc
# dlf.config3()
# dlf.topology = dlf.mainstruct4(startBus = 1)

#BusList, LineList = BuildSystem3()  # Import data from Excel file
#dlf = DistLoadFlow3(BusList, LineList)  # Create object
#dlf_nobus44 = DistLoadFlow3(BusList, LineList) #Create another object

#dlf.initialize(startBus=1) # Initialize
#dlf_nobus44.initialize(startBus=1)

#dlf.disconnectBus(44)
#dlf_nobus44.disconnectBus(44)

#dlf.DistLF(epsilon=0.00001)# Solve the case
#dlf_nobus44.DistLF(epsilon=0.00001) #solve case 2

#dlf.dispVolt(fromBus=0, tpres=True)  # Display voltages for the first 13 buses

#dlf.plotVolt(fromBus=0, toBus=0, v_angle=True, same_plot=True)

#dlf.dispFlow(tpres=True) # Display the flow on the 13 first lines When tpres=True it will display maximum 13 rows. tpres= Fasle it displays all to screen.
#dlf.dispLossSens(fromBus=15, tpres=True)
#dlf.dispLossSens(fromBus=38, toBus=46, tpres=True)
#dlf.dispVoltSens(fromBus=15, tpres=True)
#dlf.dispVoltSens(fromBus=38, toBus=46, tpres=True)
#dlf.ForwardSearch(dlf.topology)   # Show the sequence of nodes in a forward run
#dlf.BackwardSearch(dlf.topology)   # Show the sequence of nodes in a backward run
#
#
#dlf.dispGraph(dlf.topology, feeders=[1], LEC= [31, 48], lowVolt = [52, 53, 65, 64], overload=[(62, 63), (63, 64)])  # Needs som graphics libraries

#compare_volt_profiles(dlf, dlf_nobus44, toBus=25)

# New feeder

#dlf.initialize(startBus=50) # Initialize

#dlf.DistLF(epsilon=0.00001)     # Solve the case

#dlf.dispGraph(dlf.topology, feeders=[50], LEC= [31, 48], lowVolt = [52, 53, 65, 64], overload=[(62, 63), (63, 64)])

# dlf.checkOverLoad()
# dlf.checkOverflow()
# dlf.resetBuses()
# dlf.resetBuses()
# dlf.resetBuses()
# dlf.neededInjection(42, 'reactive')
# dlf.neededInjectionLine2(11)
#dlf.dispVolt(fromBus=35, tpres=True)
#dlf.dispLossSens(fromBus=35, tpres=True)
#dlf.dispFlow(fromLine=10, tpres=True)
# dlf.disconnectBus(53)
# dlf.connectLine2(70)
#dlf.findtree()
#dlf.config3()
#dlf.topology = dlf.mainstruct4()  # Set up the configuration for recursive

#Checking for splitting the network
#BusList2, LineList2 = BuildSystem3()
#dlf2 = DistLoadFlow3(BusList2, LineList2)
#dlf2.config3()
#dlf.disconnectBus(10)
#dlf2.disconnectBus(10)
#dlf2.connectLine2(69)
#dlf.findtree(1)
#dlf2.findtree(11)
#dlf.config3()
#svc = DistribObjects3.SVC(dlf.BusList[59], svcstat=1, vref=0.97, injQmax=1.0, injQmin=0.0, slopeQ=0.05 )
#dlf.BusList[44].comp = svc
#dlf2.config3()
#dlf.topology = dlf.mainstruct4()
#dlf2.topology = dlf2.mainstruct4(11)
#print('Topology first network: ')
#dlf.ForwardSearch(dlf.topology)
#print('Topology second network: ')
#dlf2.ForwardSearch(dlf2.topology)
# dlf.ForwardSearch(dlf.topology)
# dlf.UpdateControl(BusList)                    # Update the scaling factors in case of voltage control
#dlf.DistLF(epsilon=0.00001)  # Solve load flow
#dlf.overflow()
#connected = dlf.connectedBuses(dlf.topology)
#dlf.potential(dlf.BusList[60])
#dlf.highestPotential(connected)
#dlf.findCompensation(dlf.BusList[61])
#dlf.resetBuses()
#dlf.dispVolt(fromBus=0, tpres=True)  # Display voltages for the firste 13 buses
#dlf.dispVolt(fromBus=15, tpres=True)
#dlf.dispVolt(fromBus=38, toBus=46, tpres=True)
#dlf.dispFlow(tpres=True)
#dlf.dispLossSens(fromBus=15, tpres=True)
#dlf.dispLossSens(fromBus=38, toBus=46, tpres=True)
#dlf.dispVoltSens(fromBus=15, tpres=True)
#dlf.dispVoltSens(fromBus=38, toBus=46, tpres=True)
#dlf2.DistLF(epsilon=0.00001)  # Solve load flow
#dlf2.resetBuses()
#dlf2.dispVolt(fromBus=0, tpres=True)  # Display voltages for the firste 13 buses
#dlf2.dispVolt(fromBus=15, tpres=True)
#dlf2.dispVolt(fromBus=50, tpres=True)
#dlf.ForwardSearch(dlf.topology)
# dlf.dispVolt(fromBus=53,toBus=65, tpres=True)
# dlf.dispVoltSens(fromBus=53,toBus=65,tpres=True)
# dlf.dispLossSens(fromBus=53,toBus=65,tpres=True)
##dlf.dispVoltSens(fromBus=10, toBus=23, tpres=True)  # Voltage sensitivities for reduced load at the same bus and the sensitivity in reduced losses
##dlf.dispLossSens(fromBus=10, toBus=23, tpres=True)  # Loss sensitivities for reduced load at the same bus and the rate of change of loss sensitivities
#dlf.dispFlow(tpres=True)                      # Display flow on transmission lines (in graphic pres only 13 is deplayed (spes start point)
##dlf.dispFlow(fromLine=10, tpres=True)
