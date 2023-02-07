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

import numpy as np
from ElkObjects import *
import pandas as pd

from ElkMenuFunctions import ViewFileName

def BuildSystem(file=None):
    def renumber(BusList, LineList):
        iloop1 = 0
        sbase = 1   # assume that input values of load are in PU.
        temp = np.zeros(2000,dtype=int)
        while iloop1 < len(BusList):
            obj = BusList[iloop1]
            obj.busext = obj.busnum
            obj.busnum = iloop1 +1
            temp[obj.busext] = obj.busnum
            obj.pload = obj.pload/sbase
            obj.qload = obj.qload / sbase
            iloop1 += 1

        iloop1 = 0
        while iloop1 < len(LineList):
            obj = LineList[iloop1]
            obj.fbus = temp[obj.fbus]
            obj.tbus = temp[obj.tbus]
            iloop1 += 1
        return




    BusList = []
    LineList = []
    if file == None:
        file = ViewFileName(filext="xls")
    xls = pd.ExcelFile(file)
    df2 = pd.read_excel(xls, 'Bus')
    values = df2.values
    # Read Bus data  --------------------------------------------
    iloop = 0
    # print(' ')
    while iloop < len(values):
        BusList.append(Bus(busnum=int(values[iloop, 0]), bustype=int(values[iloop,1]),  pload=values[iloop, 2], qload=values[iloop, 3],
                           vmax=values[iloop, 7], vmin=values[iloop, 8]))
        iloop += 1
    df2 = pd.read_excel(xls, 'Branch')
    values = df2.values
    # Read branch data  --------------------------------------------
    iloop = 0
    # print(' ')
    while iloop < len(values):
        LineList.append(Line(fbus=int(values[iloop, 0]), tbus=int(values[iloop, 1]), r=values[iloop, 2],
                             x=values[iloop, 3], ratea=values[iloop, 5], ibstat=int(values[iloop, 10]),
                             ))
        iloop += 1

    renumber(BusList, LineList)

    return BusList, LineList
