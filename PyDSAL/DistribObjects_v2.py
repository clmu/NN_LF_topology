#!/usr/bin/python
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
# Definition of common classes

class Bus:
    'Common base class for all distribution buses'
    busCount = 0

    def __init__(self, busnum=0, pload=0.0, qload=0.0, ZIP=[0.0, 0.0 ,1.0], vset=0.0, iloss=0, pqcostRatio=100,
                 vmin=0.9, vmax=1.1, island=0, bustype=1):
        self.busnum = busnum
        self.busext = 0
        self.pload = pload
        self.qload = qload
        self.ZIP = ZIP
        self.vset = vset
        self.iloss = iloss
        self.pqcostRatio = pqcostRatio
        self.vmin = vmin
        self.vmax = vmax
        self.controlScale = 1.0     # Scaling factor to be used during voltage control and loss minimization
        self.comp = 0               # Compensation present
        self.pv = 0                 # PV present
        self.battery = 0            # Battery present
        self.v2g = 0                # V2G present
        self.ploadds = 0.0
        self.qloadds = 0.0
        self.pblossds = 0.0
        self.qblossds = 0.0
        self.dPdV = 0.0
        self.dQdV = 0.0
        self.dVdP = 0.0
        self.dVdQ = 0.0
        self.dPlossdP = 0.0
        self.dPlossdQ = 0.0
        self.dQlossdP = 0.0
        self.dQlossdQ = 0.0
        self.dP2lossdP2 = 1.0   # To be able to run the voltage optimization also in the first iteration
        self.dP2lossdQ2 = 1.0   # To be able to run the voltage optimization also in the first iteration
        self.lossRatioP = 0.0
        self.lossRatioQ = 0.0
        self.voang = 0.0
        self.vomag = 1.0
        self.busname = 'Bus' + str(busnum)
        self.toline = 0
        self.fromline = 0
        self.tolinelist = []
        self.nextbus = []
        self.bustype = bustype
        self.visited = False #might be unneccesary.
        Bus.busCount += 1

class Line:
    'Common base class for all distribution lines'
    lineCount = 0

    def __init__(self, fbus=0, tbus=0, r=0.0, x=0.0, ratea=0.0, ibstat=1, reserve = 0):
        self.fbus = fbus
        self.tbus = tbus
        self.r = r
        self.x = x
        self.ratea = ratea
        self.ibstat = ibstat #what does this do?
        self.ploss = 0.0
        self.qloss = 0.0
        self.reserve = reserve
        self.flowfromP = 0.0
        self.flowfromQ = 0.0
        self.flowtoP = 0.0
        self.flowtoQ = 0.0
        self.visited = False
        Line.lineCount += 1


class Statcom:
    'Common class for Statcom'
    statcomCount = 0
    def __init__(self, bus, scstat = 1, vref=0.0, injQmax = 0.0, injQmin = 0.0, slopeQ = 0.0 ):
        self.bus = bus
        self.scstat = scstat
        self.vref = vref
        self.injQmax = injQmax
        self.injQmin = injQmin
        self.qinj = 0.0
        self.slopeQ = slopeQ
        Statcom.statcomCount += 1


class SVC:
    'Common class for Static Var Compensator'
    svcCount = 0
    def __init__(self, bus, svcstat = 1, vref=0.0, injQmax = 0.0, injQmin = 0.0, slopeQ = 0.0 ):
        self.bus = bus
        self.stat = svcstat
        self.vref = vref
        self.vprev = vref
        self.injQmax = injQmax
        self.injQmin = injQmin
        self.qinj = 0.0
        self.slopeQ = slopeQ
        SVC.svcCount += 1

class Battery:
    'Common class for Batteries'
    batteryCount = 0
    def __init__(self, bus, svcstat = 1, vref=0.0, injPmax = 0.0, injPmin = 0.0, injQmax = 0.0, injQmin = 0.0, slopeP = 0.0, slopeQ = 0.0 ):
        self.bus = bus
        self.stat = svcstat
        self.vref = vref
        self.vprev = vref
        self.injPmax = injPmax
        self.injPmin = injPmin
        self.injQmax = injQmax
        self.injQmin = injQmin
        self.pinj = 0.0
        self.qinj = 0.0
        self.Estorage = 0.0
        self.slopeP = slopeP
        self.slopeQ = slopeQ
        Battery.batteryCount += 1

class V2G:
    'Common class for Electrical Vehicles'
    v2gCount = 0
    def __init__(self, bus, v2gstat = 1, vref=0.0, injPmax = 0.0, injPmin = 0.0, injQmax = 0.0, injQmin = 0.0, slopeP = 0.0, slopeQ = 0.0 ):
        self.bus = bus
        self.stat = v2gstat
        self.vref = vref
        self.vprev = vref
        self.injPmax = injPmax
        self.injPmin = injPmin
        self.injQmax = injQmax
        self.injQmin = injQmin
        self.pinj = 0.0
        self.qinj = 0.0
        self.Estorage = 0.0
        self.slopeP = slopeP
        self.slopeQ = slopeQ
        V2G.v2gCount += 1


class Capacitor:
    'Common class for capacitors'
    capacitorCount = 0
    def __init__(self, bus, capstat = 1, vref=0.0, blockSize = 0.0, numBlocks = 1):
        self.bus = bus
        self.capstat = capstat
        self.vref = vref
        self.blockSize = blockSize
        self.numBlocks = numBlocks
        self.currentStep = 0
        Capacitor.capacitorCount += 1

class PV:
    'Common class for PhotoVoltaic (PV)'
    pvCount = 0
    def __init__(self, bus, pvstat = 1, cmode = 1, vref=0.0, convCap = 0.0, injPmax = 0.0, injPmin = 0.0, injQmax = 0.0, injQmin = 0.0, slopeP = 0.0, slopeQ = 0.0 ):
        self.bus = bus
        self.stat = pvstat
        self.cmode = cmode          # cmode = 1 (PV) , cmode = 2 (P - droop Q, cmode = 3 (Droop P, droop Q)
        self.vref = vref            # Voltage reference - interpreted according to the control mode (cmode)
        self.vprev = vref           # Needed for droop control - iterative procedure
        self.convCap = convCap      # Total converter capability - S^2 = P^2 + Q^2 - limits calculated accordingly or specified
        self.injPmax = injPmax
        self.injPmin = injPmin
        self.injQmax = injQmax
        self.injQmin = injQmin
        self.pinj = 0.0
        self.qinj = 0.0
        self.slopeP = slopeP
        self.slopeQ = slopeQ
        PV.pvCount += 1
