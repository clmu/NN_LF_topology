#!/usr/bin/python
# Copyright (c) 2022, Olav B. Fosso, NTNU
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

    def __init__(self, busnum=0, pload=0.0, qload=0.0, ZIP = [0.0, 0.0, 1.0],vmin=0.9,vmax=1.1,bustype=1):
        self.busnum = busnum
        self.bustype = bustype
        self.pload = pload
        self.qload = qload
        self.ZIP = ZIP
        self.vmin = vmin
        self.vmax = vmax
        self.voang = 0.0
        self.vomag = 1.0
        self.busname = 'Bus' + str(busnum)
        self.busP = None
        self.BusQ = None
        self.PF = None
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
        self.ibstat = ibstat
        self.ploss = 0.0
        self.qloss = 0.0
        self.dcflow = None
        self.ptdfs = None
        self.capacity = None
        Line.lineCount += 1

