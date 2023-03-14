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
import math
# import cmath


import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv as inv

from LF_3bus.ElkObjects import *
from LF_3bus.build_sys_from_sheet import *
import time


class LoadFlow:
    """
    Common base class   Load Flow
    Input:
        BusList      - List off all Bus objects
        LineList    - List of all transmission lines objects
    Returns: None
    """

    def __init__(self, Buses, Lines):
        self.BusList = Buses
        self.LineList = Lines
        self.voang = np.zeros(len(self.BusList))  # Flat start
        self.vomag = np.ones(len(self.BusList))  # Flat start
        self.ploads, self.qloads = self.initialize_Pload_Qload_vector()
        self.topology = []
        self.mis, self.var = self.establish_mismatch_variable_vector()
        self.iterations = None
        self.convergence_time = None


    def calc_all_powers(self):
        old_mis = self.mis
        self.mis = self.establish_all_powers()
        calculated_powers = self.calculate_powers()
        self.mis = old_mis
        return calculated_powers


    def establish_mismatch_variable_vector(self):
        '''
        Perhaps the most important function in my load flow implementation, as it defines the dimensions of the jacobian
        This function iterates through the BusList twice. During the first round it checks if the bus is either a load
        or generator bus, and will append a list containing the variable and busnumber to the mistmatch vector.
        similarly, for all powers it appends to the mismatch vector it appends the variable and its corresponding
        busnumber to the mistmatch vector
        once finished with the first round, having appended all powers ('P') to the mismatch vector, and all angles
        ('D') to the variable vector, it checks if the bus is a load bus, and appends the reactive power to the mismatch
        vector and the voltage magnitude to the variable vector.
        Now there are two vectors: one containing all known variables to the left (mismatches), and all unknown
        variables to the right.
        :return: vector of all known powers, vector of all known variables.
        '''
        mismatches = []
        variables = []

        for bus in self.BusList:
            if bus.bustype == 1 or bus.bustype == 2:
                mismatches.append(['P', bus.busnum])
                variables.append(['D', bus.busnum])
        for bus in self.BusList:
            if bus.bustype == 1:
                mismatches.append(['Q', bus.busnum])
                variables.append(['|V|', bus.busnum])
        return mismatches, variables


    def establish_all_powers(self):
        mismatches = []
        for bus in self.BusList:
            if bus.bustype == 1 or bus.bustype == 2:
                mismatches.append(['P', bus.busnum])
            elif bus.bustype == 3:
                mismatches.append(['P', bus.busnum])
        for bus in self.BusList:
            if bus.bustype == 1:
                mismatches.append(['Q', bus.busnum])
            elif bus.bustype == 2 or bus.bustype == 3:
                mismatches.append(['Q', bus.busnum])
        return mismatches


    def initialize_Pload_Qload_vector(self):
        '''
        Function establishes a vector containing the loads at known buses.
        function is used to establish vectors for the load flow class.
        :return: two vectors containing all loads and reactive loads.
        '''
        p_loads = []
        q_loads = []
        for bus in self.BusList:
            p_loads.append(bus.pload)
            q_loads.append(bus.qload)
        return p_loads, q_loads

    def linelookup(self,fbus,tbus):
        '''
        function returns the line object that connects the two buses.
        note the indent of the return statement: This must be there, as the line_obj will not be created for lines not
        matching the criteria. If the criteria do not match there will be no line to return.
        :param fbus: from bus
        :param tbus: to bus
        :return: line object connecting the two buses
        '''
        for line in self.LineList:
            if (line.fbus == fbus and line.tbus == tbus) or (line.fbus == tbus and line.tbus == fbus):
                line_obj = line
                return line_obj

    def buslookup(self, busnum):

        '''
        Function returns the bus with a given busnumber.
        :param busnum: number of the bus in the system.
        :return bus object with number busnum.
        '''
        for bus in self.BusList:
            if bus.busnum == busnum:
                break
        return bus

    def admittance(self, fbus, tbus):
        if fbus != tbus:
            line = self.linelookup(fbus,tbus)
            return 1.0 / complex(line.r, line.x)
        elif fbus == tbus:
            connected_lines = self.find_lines_from_bus(fbus)
            admittance = complex(0,0)
            for line in connected_lines:
                admittance += 1.0 / complex(line.r, line.x)
            return admittance

    def gij(self,fbus,tbus): #conductance
        return self.admittance(fbus, tbus).real

    def bij(self,fbus,tbus): #susceptance
        bij = self.admittance(fbus, tbus).imag
        return bij

    def tij(self, fbus, tbus):
        '''
        :param fbus: from bus
        :param tbus: to bus
        :return: T_ij element
        '''

        fbusidx = fbus - 1
        tbusidx = tbus - 1
        fvoang = self.voang[fbusidx]
        tvoang = self.voang[tbusidx]
        tij_new = self.gij(fbus, tbus) * np.cos(fvoang - tvoang) + self.bij(fbus, tbus) * np.sin(fvoang - tvoang)
        return tij_new

    def uij(self, fbus, tbus):
        '''
        :param fbus: from bus
        :param tbus: to bus
        :return: U_ij element
        '''

        fbusidx = fbus - 1
        tbusidx = tbus - 1
        fvoang = self.voang[fbusidx]
        tvoang = self.voang[tbusidx]
        uij_new = self.gij(fbus, tbus) * np.sin(fvoang - tvoang) - self.bij(fbus, tbus) * np.cos(fvoang - tvoang)
        return uij_new

    def buildJacobian(self):
        '''
        Fucntion builds the jacobian based on the active power derivatives given in lecture
        ELK14_L2_LoadFlow_Basic_Modelling_NR_2022. The function depends on self.vomag, uij, and tij. uij and tij both
        obtain their angles from self.voang
        :return: np.array containing all the active power derivatives as FLOATS
        '''
        jac = np.zeros([len(self.mis), len(self.var)], dtype=float)
        row = 0
        for mismatch in self.mis:
            column = -1
            '''
            set to -1 to make the first index 0, and to allow use of continue statement. 
            Using the continue statement should improve the the function speed. 
            '''
            for variable in self.var:
                column += 1
                PorQ = mismatch[0]
                var = variable[0]
                fbusnum = mismatch[1]
                tbusnum = variable[1]
                fbusidx = fbusnum - 1
                tbusidx = tbusnum - 1
                if PorQ == 'P' and var == 'D': #dP/dD, D represents theta
                    if fbusnum != tbusnum:
                        jac[row,column] = -self.vomag[fbusidx] * self.vomag[tbusidx] * self.uij(fbusnum,tbusnum)
                        continue
                    else:
                        connected_lines = self.find_lines_from_bus(fbusnum)
                        summation = 0
                        V_i = self.vomag[fbusidx]
                        for line in connected_lines:
                            i = line.fbus
                            j = line.tbus
                            summation += self.vomag[j - 1] * self.uij(i, j)
                        jac[row, column] = V_i * summation
                        continue
                if PorQ == 'P' and var == '|V|': #dP/dV
                    if fbusnum != tbusnum:
                        jac[row, column] = -self.vomag[fbusidx] * self.tij(fbusnum, tbusnum)
                        continue
                    else:
                        connected_lines = self.find_lines_from_bus(fbusnum)
                        summation = 0
                        for line in connected_lines:
                            i = line.fbus
                            j = line.tbus
                            summation += self.vomag[j - 1] * self.tij(i, j)
                        jac[row, column] = 2 * self.vomag[fbusidx] * self.gij(fbusnum, fbusnum) - summation
                        continue
                if PorQ == 'Q' and var == 'D': #dQ/dD
                    if fbusnum != tbusnum:
                        jac[row, column] = self.vomag[fbusidx] * self.vomag[tbusidx] * self.tij(fbusnum, tbusnum)
                        continue
                    else:
                        connected_lines = self.find_lines_from_bus(fbusnum)
                        summation = 0
                        for line in connected_lines:
                            i = line.fbus
                            j = line.tbus
                            summation += self.vomag[j-1] * self.tij(i, j)
                        jac[row, column] = -self.vomag[fbusidx] * summation
                        continue
                if PorQ == 'Q' and var == '|V|':
                    if fbusnum != tbusnum:
                        jac[row, column] = -self.vomag[fbusidx] * self.uij(fbusnum,tbusnum)
                        continue
                    else:
                        connected_lines = self.find_lines_from_bus(fbusnum)
                        summation = 0
                        for line in connected_lines:
                            i = line.fbus
                            j = line.tbus
                            summation += self.vomag[j - 1] * self.uij(i, j)
                        jac[row, column] = -2 * self.vomag[fbusidx] * self.bij(fbusnum,fbusnum) - summation
            row += 1
        return jac

    def find_lines_from_bus(self, busnum):
        '''
        Function takes a bus, iterates through the list of lines returning a list of all connected
        lines.
        The function returns all lines as they connect the bus from the initiating bus.
        :param busnum: bus to check for connections
        :return: list of lines connected to the bus.
        '''
        connected_lines = []
        for line in self.LineList:
            if line.fbus == busnum:
                connected_lines.append(line)
            elif line.tbus == busnum:
                line.tbus = line.fbus #changing fbus and tbus, useful for further use of this function.
                line.fbus = busnum
                connected_lines.append(line)
        return connected_lines

    def calculate_powers(self):
        '''
        Function calculates the relevant powers based on the current voltages and angles in the network. The function is
        based on the self.mis vector, which should make the code flexible enough to accept different arrangement of
        buses.
        note that: the return vector is only valid for the current voltages and angles in the LoadFlow class.
        :return:vector containing all powers for the mismatch calculation.
        '''
        calculated_powers = []
        for P_or_Q in self.mis:
            connected_lines = self.find_lines_from_bus(P_or_Q[1])
            if P_or_Q[0] == 'P':
                power_summation_part = 0
                for line in connected_lines:
                    fbusnum = line.fbus
                    tbusnum = line.tbus
                    tbusidx = tbusnum - 1
                    power_summation_part += self.vomag[tbusidx] * self.tij(fbusnum, tbusnum)
                power = self.vomag[P_or_Q[1]-1]**2 * self.gij(P_or_Q[1], P_or_Q[1]) - self.vomag[P_or_Q[1]-1] * power_summation_part
                calculated_powers.append(power)
            elif P_or_Q[0] == 'Q':
                q_power_summation_part = 0
                for line in connected_lines:
                    fbusnum = line.fbus
                    tbusnum = line.tbus
                    tbusidx = tbusnum - 1
                    q_power_summation_part += self.vomag[tbusidx] * self.uij(fbusnum, tbusnum)
                q_power = - (self.vomag[P_or_Q[1]-1]**2) * self.bij(P_or_Q[1], P_or_Q[1]) - self.vomag[P_or_Q[1]-1] * q_power_summation_part
                calculated_powers.append(q_power)
            else:
                raise Exception("Error in calculate powers: There is likely an error in the mis vector of the function")
        return calculated_powers

    def calcMismatch(self):
        calculated_powers = self.calculate_powers()
        mismatch = []
        if len(calculated_powers) == len(self.mis):
            for i in range(len(self.mis)):
                p_counter = 0
                q_counter = 0
                if self.mis[i][0] == 'P':
                    mismatch.append(- calculated_powers[i] - self.ploads[self.mis[i][1] - 1])
                    p_counter += 1
                elif self.mis[i][0] == 'Q':
                    mismatch.append(- calculated_powers[i] - self.qloads[self.mis[i][1] - 1])
                    q_counter += 1
        else:
            raise Exception("Neither P nor Q found in mis vector at index [0]")
        mismatch = np.array(mismatch) #converting to numpy array
        return mismatch

    def solve_NR(self, tolerance = 0.00001, max_iterations=15, dc_flag=False):
        '''
        Function solves the NR LF for a given tolerance and an accepted number of iterations.
        NB: many of the load flow classes functions rely on data in the bus objects. Thus, these have to be updated.
        :param tolerance: tolerance/accuracy for the NR LF algorithm
        :param max_iterations: maximum number of allowed algorithms.
        :return: vector of voltage magnitudes, vector of angles
        '''
        time_start = time.perf_counter()
        if not dc_flag:
            next_voltages = np.ones(len(self.vomag))
            next_angles = np.zeros(len(self.voang))
        else:
            next_voltages = self.vomag
            next_angles = np.zeros(len(self.voang))
        max_mismatch = 10
        number_of_iterations = 0

        while max_mismatch > tolerance:
            number_of_iterations += 1
            jac = self.buildJacobian()
            mismatch = self.calcMismatch()
            delta_variables = np.linalg.solve(jac, mismatch)
            counter = 0
            for variable in self.var:
                busnum = variable[1]
                if variable[0] == 'D':
                    next_angles[busnum-1] += delta_variables[counter]
                elif variable[0] == '|V|':
                    next_voltages[busnum-1] += delta_variables[counter]
                counter += 1
            max_mismatch = max(np.abs(mismatch))
            self.vomag = next_voltages  # update current voltages
            self.voang = next_angles  # update current angles
            if number_of_iterations > max_iterations:
                raise Exception("NR algorithm does not converge")
                break
        self.voang = next_angles
        self.vomag = next_voltages
        self.iterations = number_of_iterations
        time_end = time.perf_counter()
        self.convergence_time = time_end - time_start
        pass
        #return self.vomag, self.voang


    def calc_NR_lineflow(self, fbusnum, tbusnum):

        '''
        Function calculates the lineflow on a line using the complex voltages from the loadflow solution.
            NEGLECTING SHUNT ADMITTANCE
        :param fbus: from bus number
        :param tbus: to bus number
        :return:
        '''
        def cartesian_complex(vomag, voang):
            real = vomag * np.cos(voang)
            imag = vomag * np.sin(voang)
            return complex(real, imag)
        line = self.linelookup(fbusnum, tbusnum)
        fbus, tbus = self.buslookup(fbusnum), self.buslookup(tbusnum)
        y_ij = 1 / complex(line.r, line.x)
        return y_ij * (cartesian_complex(fbus.vomag, fbus.voang)
                       - cartesian_complex(tbus.vomag, tbus.voang))

    def generate_lineflow_matrix(self):
        dim = len(self.LineList)
        lineflows = np.zeros((dim, dim), dtype=float)
        for line in self.LineList:
            fbus_idx = line.fbus - 1
            tbus_idx = line.tbus - 1
            self.cplxflow = self.calc_NR_lineflow(line.fbus, line.tbus)
            sign = -1
            if self.cplxflow.real >= 0:
                sign = 1
            lineflows[fbus_idx, tbus_idx] = np.abs(self.cplxflow) * sign
            lineflows[tbus_idx, fbus_idx] = np.abs(self.cplxflow) * sign * -1
        return


class cont_power_flow(LoadFlow):
    #   This is  a subclass of continuation power flow.
    def __init__(self, Buses, Lines, cont_parameter='V'):
        '''
        The super function makes sure that the buses and lines are initiated by the init statement of the parent class.
        This also means that the init of cont power flow can be initiated with more variables. such as the cont parameter.
        '''
        super().__init__(Buses, Lines)
        self.cont_par = cont_parameter
        self.step = self.stepsize()
        self.large_jacobian = self.build_large_jacobian()

    def establish_beta(self, beta = None):
        #Hope to add more general code here later

        if type(beta) == type([]) or type(beta) == type(np.array([1])):
            return beta
        elif beta != None:
            return beta
        else:
            powers = []
            for P_or_Q in self.mis:
                if P_or_Q[0] == 'P':
                    powers.append(P_or_Q)
            beta = np.ones(len(powers))
            beta[0], beta[1] = 0.3, 0.7
            return beta


    def establish_alpha(self, alpha = None):
        #Hope to add more general code here later
        '''
        if type(alpha) == type([]) or type(alpha) == type(np.array([1])):
            return alpha
        elif alpha != None:
            return alpha
        '''
        powers = []
        for P_or_Q in self.mis:
            if P_or_Q[0] == 'Q':
                powers.append(P_or_Q)
        alpha = np.ones(len(powers))
        alpha[0], alpha[1] = 0, 0

        return alpha

    def build_large_jacobian(self, beta = None, alpha = None, cont_par_idx = None):
        beta, alpha = np.asarray(beta), np.asarray(alpha) #changing the parameters to arrays
        if cont_par_idx == None:
            cont_par_idx = len(self.var) - 1
        if beta.any() == None and alpha.any() == None:
            beta_alpha = np.append(self.establish_beta(), self.establish_alpha())
        elif beta.any() == None and alpha.any() != None:
            beta_alpha = np.append(self.establish_beta(), alpha)
        elif alpha.any() == None and beta.any() != None:
            beta_alpha = np.append(beta, self.establish_alpha())
        else:
            beta_alpha = np.append(beta, alpha)

        small_jacobian = self.buildJacobian()
        [rows, columns] = small_jacobian.shape
        rows += 1
        columns += 1
        large_jacobian = np.zeros([rows, columns])

        for row in range(rows):
            for column in range(columns):
                if column < (columns - 1) and row < (rows - 1):
                    large_jacobian[row, column] = small_jacobian[row, column]
                elif column == ( columns - 1 ) and row < ( rows - 1 ):
                    large_jacobian[row, column] = beta_alpha[row]
                elif row == (rows - 1) and self.cont_par == 'V' and column == cont_par_idx: #column == (columns - 2):
                    large_jacobian[row, column] = 1
                elif row == (rows - 1) and self.cont_par == 'P' and column == (columns - 1):
                    large_jacobian[row, column] = 1

        return large_jacobian

    def stepsize(self):
        '''
        Placeholder for calculation of the stepsize.
        :return: one
        '''
        return 0.3

    def update_state_variables(self, delta_variables, predictor = True):
        next_angles = self.voang
        next_voltages = self.vomag

        if predictor:
            step = self.step
        else:
            step = 1

        counter = 0
        for variable in self.var:
            busnum = variable[1]
            if variable[0] == 'D':
                next_angles[busnum - 1] += delta_variables[counter] * step
            elif variable[0] == '|V|':
                next_voltages[busnum - 1] += delta_variables[counter] * step
            counter += 1

        self.voang = next_angles
        self.vomag = next_voltages

        pass

    def power_alpha_beta_increment(self, beta, alpha, stepSize  = None, S = 1):

        if stepSize == None:
            stepSize = self.step

        number_of_powers = 0
        number_of_Rpowers = 0

        for loads in self.mis:
            if loads[0] == 'P':
                number_of_powers += 1
            if loads[0] == 'Q':
                number_of_Rpowers += 1
        counterP = 0
        counterQ = 0
        for P_or_Q in self.mis:
            if counterP < number_of_powers:
                self.ploads[counterP] += stepSize * beta[counterP] * S
                counterP += 1
            else:
                self.qloads[counterQ] += stepSize * alpha[counterQ] * S
                counterQ += 1
        pass

    def cont_predictor(self, cont_par = 'V', alphainput = None, betainput = None, sensitivities = False):
        b = np.append(np.zeros(len(self.var)), 1)  # gives vector as long as the var and appends 1
        self.cont_par = 'P'
        self.large_jacobian = self.build_large_jacobian(beta=betainput, alpha=alphainput)
        self.cont_par = cont_par
        x = np.linalg.solve(self.large_jacobian, b)
        print('Sensitivities: ', x)
        self.update_state_variables(x, predictor=True)
        beta = self.establish_beta(beta=betainput)
        alpha = self.establish_alpha(alpha=alphainput)
        self.power_alpha_beta_increment(beta, alpha, stepSize=self.step)
        print('Voltages: ', self.vomag)
        print('Angles: ', self.voang)
        if sensitivities:
            return x
        else:
            pass
    #def cont_corrector(self, precision=0.00001, alphainput = None, betainput = None, max_iterations = 15, cont_idx = None, stepsize = None):
    def cont_corrector(self, precision=0.00001, alphainput=None, betainput=None, max_iterations=15, cont_idx=None,
                           stepsize=None):
        if cont_idx == None:
            cont_idx = len(self.var) - 1
        converged = False  # add convergence check.
        counter = 0
        if betainput == None:
            betainput = self.establish_beta(beta=betainput)
        if alphainput == None:
            alphainput = self.establish_alpha(alpha=alphainput)
        if stepsize != None:
            self.step = stepsize
        while not converged:
            print(f'-----------Correction iteration {counter + 1}-------------')
            print('All powers: ', self.calc_all_powers(), '\n')
            self.large_jacobian = self.build_large_jacobian(beta=betainput, alpha=alphainput, cont_par_idx =  cont_idx)
            print('The extended jacobian: \n', self.large_jacobian)
            mismatch = self.calcMismatch() #there seems to be an error in the mismatch calculation.
            mismatch = np.append(mismatch, [0])
            #mismatch.append(0)
            print('RHS-Vector: ', mismatch, '\n')
            # old_voltages, old_angles = self.vomag, self.voang
            # np.append(mismatch, 0)
            variable_increments = np.linalg.solve(self.large_jacobian, mismatch)
            print('Correction (including s at last index): ', variable_increments, '\n')
            self.update_state_variables(variable_increments, predictor=False)
            print('Voltages: ', self.vomag)
            print('Angles: ', self.voang)
            if self.cont_par == 'V':
                self.power_alpha_beta_increment(betainput, alphainput, S=variable_increments[-1])  # including the last value of Alpha beta as increment
            random = abs(variable_increments[0:len(variable_increments)-1:])
            if max(abs(variable_increments[0:len(variable_increments)-1:])) < precision:
                converged = True
            if counter == max_iterations:
                print(f'Corrector iteration did not converge within {max_iterations} iterations')
                break
            counter += 1
        all_powers = self.calc_all_powers()
        print('Ps for all buses:', all_powers[:3:])
        print('Qs for all buses:', all_powers[3::])
        pass

class decoupled_LF(LoadFlow):
    def __init__(self, Buses, Lines, cont_parameter='V'):
        '''
        The super function makes sure that the buses and lines are initiated by the init statement of the parent class.
        This also means that the init of cont power flow can be initiated with more variables. such as the cont parameter.
        '''
        super().__init__(Buses, Lines)
        self.num_of_pow = self.number_of_powers()
        self.num_of_qpow = self.number_of_Qpowers()
        self.dc_flag = False

    def number_of_powers(self):
        p_count = 0
        for P_or_Q in self.mis:
            if P_or_Q[0] == 'P':
                p_count += 1
        return p_count

    def number_of_Qpowers(self):
        q_count = 0
        for P_or_Q in self.mis:
            if P_or_Q[0] == 'Q':
                q_count += 1
        return q_count

    def decoupled_jacobians(self):
        normal_jac = self.buildJacobian()
        p_jac = np.zeros([self.num_of_pow,self.num_of_pow], dtype=float)
        q_jac = np.zeros([self.num_of_qpow,self.num_of_qpow],dtype=float)

        rows, columns = np.shape(normal_jac)

        q_row_idx = 0
        q_column_idx = 0
        for row in range(rows):
            for column in range(columns):
                if row < (self.num_of_pow - 0) and column < (self.num_of_pow - 0):
                    p_jac[row, column] = normal_jac[row, column]
                elif row >= self.num_of_pow and column >= self.num_of_pow:
                    q_jac[q_row_idx, q_column_idx] = normal_jac[row, column]
                    q_column_idx += 1
            if row >= self.num_of_pow and column >= self.num_of_pow:
                q_row_idx += 1
                q_column_idx = 0

        return p_jac, q_jac

    def update_angles(self, d_angle):
        next_angles = self.voang
        counter = 0
        for variable in self.var:
            if variable[0] == 'D':
                next_angles[counter] += d_angle[counter]
                counter += 1
        self.voang = next_angles
        pass

    def update_voltages(self, d_voltage):
        next_voltages = self.vomag
        counter = 0
        for variable in self.var:
            if variable[0] == '|V|':
                next_voltages[counter] += d_voltage[counter]
                counter += 1
        self.vomag = next_voltages
        pass

    def use_regular_nr_data(self):
        for line in self.LineList:
            line.r = 0
        for qload in self.qloads:
            qload = 0
        for pload in self.ploads:
            pload *= -1

    def DC_PF(self, precision=0.00001, max_iterations=100, p_jac=None, q_jac=None, dual=False):
        '''
        Do not recalculate the subjacobians in the iterative procedure.
        if it does not receive p_jac and q_jac, the function makes them itself.
        :return:
        '''
        time_start = time.perf_counter()
        if np.asarray([p_jac]).any() == None and np.asarray([q_jac]).any() == None:
            p_jac, q_jac = self.decoupled_jacobians()
        converged =  False
        iteration = 0
        while not converged:
            '''
            note to self:
            during the primal algorithm the angles should be calculated first. This is the way it is always done here. 
            During the dual algorithm, the voltages should be calculated first. This is not implemented. 
                because i did not think of a way to do this in the same function.
                
            Add variable to define if it is primal or dual. 
            '''
            mismatches = self.calcMismatch()
            if dual:
                q_mis = mismatches[self.num_of_pow::]
                d_voltage = np.linalg.solve(q_jac, q_mis)
                self.update_voltages(d_voltage)
                mismatches = self.calcMismatch()
                p_mis = mismatches[:self.num_of_pow:]
                d_angle = np.linalg.solve(p_jac, p_mis)
                self.update_angles(d_angle)
            else:
                p_mis = mismatches[:self.num_of_pow:]
                d_angle = np.linalg.solve(p_jac, p_mis)
                self.update_angles(d_angle)
                mismatches = self.calcMismatch()
                q_mis = mismatches[self.num_of_pow::]
                d_voltage = np.linalg.solve(q_jac, q_mis)
                self.update_voltages(d_voltage)
            iteration += 1
            if max(abs(mismatches)) < precision:
                converged = True
                self.iterations = iteration
                break
            if iteration >= max_iterations:# and not converged:
                print('The load flow did not converge!')
                break
        time_end = time.perf_counter()
        self.convergence_time = time_end-time_start
        self.dc_flag = True
        pass



    def DC_PF_suboptimal(self, precision = 0.00001, max_iterations = 20, p_jac = None, q_jac = None):
        '''
        Do not recalculate the subjacobians in the iterative procedure.
        if it does not receive p_jac and q_jac, the function makes them itself.
        :return:
        '''
        time_start = time.perf_counter()
        if p_jac == None and q_jac == None:
            p_jac, q_jac = self.decoupled_jacobians()
        converged =  False
        iteration = 0
        while not converged:
            mismatches = self.calcMismatch()
            p_mis = mismatches[:self.num_of_pow:]
            q_mis = mismatches[self.num_of_pow::]
            d_angle = np.linalg.solve(p_jac,p_mis)
            d_voltage = np.linalg.solve(q_jac, q_mis)
            self.update_voltages(d_voltage)
            self.update_angles(d_angle)
            iteration += 1
            if max(abs(mismatches)) < precision:
                converged = True
                self.iterations = iteration
            if iteration >= max_iterations and not converged:
                print('The load flow did not converge!')
                self.iterations = max_iterations
                break
            time_end = time.perf_counter()
            self.convergence_time = time_end-time_start
        pass

    def b_bus(self, fbus = None, tbus = None):
        full_b_bus = np.zeros((len(self.BusList), len(self.BusList)), dtype=float)
        for line in self.LineList:
            '''
            Gives all the off-diagonal elements in the ybus. The following line is repeated with flipped indices to 
            create a complete Y_bus matrix.
            '''
            full_b_bus[line.fbus - 1, line.tbus - 1] -= 1 / line.x
            full_b_bus[line.tbus - 1, line.fbus - 1] -= 1 / line.x

            # gives alle the diagonal elements
            full_b_bus[line.fbus - 1, line.fbus - 1] += 1 / line.x
            full_b_bus[line.tbus - 1, line.tbus - 1] += 1 / line.x

        if fbus == None:
            return full_b_bus
        else:
            fbus_idx = fbus - 1
            tbus_idx = tbus - 1
            full_b_bus = full_b_bus[fbus_idx: tbus_idx + 1, fbus_idx: tbus_idx + 1]
            return full_b_bus


class DC_LF(LoadFlow):
    def __init__(self, Buses, Lines):
        '''
        The super function makes sure that the buses and lines are initiated by the init statement of the parent class.
        This also means that the init of cont power flow can be initiated with more variables. such as the cont parameter.
        '''
        super().__init__(Buses, Lines)
        self.slackbuses = []
        self.powers = [[], []]
        self.buspowers = np.zeros(len(self.BusList))
        self.b_bus = []
        self.X_mat = None# to be the inverse of the b bus
        self.find_slackbuses()

    def find_slackbuses(self):
        for bus in self.BusList:
            if bus.bustype == 3:
                self.slackbuses.append(bus.busnum)
        pass

    def establish_power_vector(self):
        for bus in self.BusList:
            if bus.bustype != 3:
                self.powers[0].append(bus.busnum)
                self.powers[1].append(bus.pload)
        pass

    def find_lines_connected_to_slack(self):
        slack_connected_lines = []
        for line in self.LineList:
            if line.fbus in self.slackbuses or line.tbus in self.slackbuses:
                slack_connected_lines.append(line)
            if line.tbus in self.slackbuses: #ensures that all slackbuses are the fbuses.
                old_tbus, old_fbus = line.tbus, line.fbus
                line.fbus, line.tbus = old_tbus, old_fbus
        return slack_connected_lines

    def build_b_bus(self):#note: as of now, it is beneficial if the slackbus is the last bus, as this makes indexing easier.
        dimension = len(self.BusList)-len(self.slackbuses)
        b_bus = np.zeros((dimension, dimension), dtype=float)
        for line in self.LineList:
            if (line.fbus not in self.slackbuses) and (line.tbus not in self.slackbuses):
                b_bus[line.fbus - 1, line.tbus - 1] -= 1 / line.x
                b_bus[line.tbus - 1, line.fbus - 1] -= 1 / line.x

                # gives alle the diagonal elements
                b_bus[line.fbus - 1, line.fbus - 1] += 1 / line.x
                b_bus[line.tbus - 1, line.tbus - 1] += 1 / line.x
        slack_connected_lines = self.find_lines_connected_to_slack()
        for slackline in slack_connected_lines: #gives the remaining diagonal elements.
            b_bus[slackline.tbus - 1, slackline.tbus - 1] += 1 / slackline.x
        self.b_bus = b_bus
        pass

    def update_angles(self, d_angles):
        for angle_idx in range(len(self.powers[0])):
            #update angles at correct indexes.
            voang_idx = self.powers[0][angle_idx] - 1
            self.voang[voang_idx] += d_angles[angle_idx]
        pass

    def solve_DC(self):
        #new_angles = np.zeros(len(self.BusList) - len(self.slackbuses), dtype=float)
        self.establish_power_vector()
        self.build_b_bus()
        d_angles = np.linalg.solve(self.b_bus, self.powers[1])
        self.update_angles(d_angles)

        counter = 0
        powersum = 0
        for bus_num in self.powers[0]:
            bus_idx = bus_num - 1
            self.buspowers[bus_idx] = self.powers[1][counter]
            powersum += self.powers[1][counter]
            counter += 1
        self.buspowers[-1] = -powersum #given that the slackbus is at the end.
        pass

    def generate_wrong_x_mat(self):
        #X_mat as inverse of B (B: all line reactances)
        self.X_mat = np.linalg.inv(self.b_bus)
        pass

    def generate_x_mat(self):
        #x_mat as all line reactances. Note that the diagonal remains 0, as it has no logical value in this interpretation
        self.X_mat = np.zeros((len(self.LineList), len(self.LineList)), dtype=float)
        for line in self.LineList:
            self.X_mat[line.fbus-1, line.tbus-1] = line.x
            self.X_mat[line.tbus - 1, line.fbus - 1] = line.x
        pass

    def a_ij_fast(self, i, j,): #i and j should be busnumbers
        if self.X_mat is None:
            self.generate_x_mat()
        rhs = np.zeros(np.shape(self.b_bus)[0])
        reactance = self.X_mat[i-1, j-1]
        if i not in self.slackbuses:
            rhs[i-1] += 1 / reactance#self.X_mat[i-1, j-1]
        if j not in self.slackbuses:
            rhs[j-1] -= (1 / reactance) #self.X_mat[i-1, j-1]
        return np.linalg.solve(self.b_bus, rhs)

    def switch_line_fbus_tbus(self):
        for line in self.LineList:
            if line.fbus > line.tbus:
                old_fbus = line.fbus
                old_tbus = line.tbus
                line.fbus, line.tbus = old_tbus, old_fbus
        pass

    def calc_lineflow(self, fbus=None, tbus=None, line=None):
        if line is not None:
            fbus = line.fbus
            tbus = line.tbus
        if self.X_mat is None:
            self.generate_x_mat()
        counter = (self.voang[fbus - 1] - self.voang[tbus - 1])
        denominator = self.X_mat[fbus-1, tbus-1]
        lineflow = counter / denominator
        return lineflow

    def add_ptdfs_to_line(self):
        self.switch_line_fbus_tbus()
        for line in self.LineList:
            line.ptdfs = self.a_ij_fast(line.fbus, line.tbus)
        pass

    def add_lineflows(self):
        self.switch_line_fbus_tbus()
        for line in self.LineList:
            line.dcflow = self.calc_lineflow(line=line)
        pass

    def imml_singleline(self, fbus, tbus, d_x=1):
        '''
        Function adapts the load flow results to the line change using the IMML scheme to avoid unneccesary calculations
        currently only implemented for single lines
        :param fbus:
        :param tbus:
        :param d_x: how much the line admittance should be changed. 0.5 for removing a parallel line
        :return: none. Changes are made directly to the loadflow object's properties
        '''
        d_h = - 1 / (self.linelookup(fbus, tbus).x * d_x)
        if self.b_bus is None:
            self.build_b_bus()
        h = self.b_bus
        h_inv = inv(h) #np.linalg.inv()
        m = np.zeros((self.b_bus.shape[0], 1))
        m[fbus-1][0], m[tbus-1][0] = 1, -1
        m_T = m.transpose()
        z = m_T @ h_inv @ m
        c = inv(z) * inv((d_h + inv(z))) * d_h
        random = len(m_T[0])
        d_delta = -h_inv @ m @ c @ m_T @ self.voang[0:len(m_T[0]):]
        self.update_angles(d_delta)
        pass

class Opf(DC_LF):
    def __int__(self, Buses, Lines):
        super().__int__(Buses, Lines)
        self.gen_costs = []
        self.gen_limit = []
        self.gen_min_limit = []



# -------------------------------------------------------------------------------------------------------------
#
# Demo case (Illustration of how to build up a script)
#
#BusList, LineList = BuildSystem()  # Import data from Excel file
# BusList and LineList contain lists of Bus-objects and Line-objects
#lf = LoadFlow(BusList, LineList)  # Create object


# How to access objects

#Hello_Assingment_2_Cont_PF

'''
for x in BusList:
    print('Busname :', x.busname)

for x in LineList:
    print('Frombus :', x.fbus, '  ToBus :', x.tbus)

'''

'''
mismatch_v, variable_v = lf.establish_mismatch_variable_vector()

print(f'Mismatch:{mismatch_v}')

print(f'variables: {variable_v}')


admittance = lf.admittance(1,1)

print(admittance)



lines_connecting_to_bus = lf.find_lines_from_bus(1)

print(f'From bus: {lines_connecting_to_bus[0].fbus}\n To bus: {lines_connecting_to_bus[0].tbus}\n')
print(f'From bus: {lines_connecting_to_bus[1].fbus}\n To bus: {lines_connecting_to_bus[1].tbus}')

uij_1 = lf.uij(lines_connecting_to_bus[0].fbus,lines_connecting_to_bus[0].tbus)

uij_2 = lf.uij(lines_connecting_to_bus[1].fbus,lines_connecting_to_bus[1].tbus)

print(f'\nuij_1: {uij_1},\tuij_2: {uij_2}\n')



It is created an environment call "lf" where a BusList[x] where x is numbered from 0 to nbus -1 and where
a number of attributes are avaiable.
The command "lf = lf.BusList[2].pload" will return the load at the third bus (all numbered from zero in Python)
The command "rating = lf.LineList[2].ratea" will return the rating of the third line in the system
"itr = lf.LineList[2].fbus" returns the from bus of the line
"itr = lf.LineList[2].tbus" returns the to bus of the line

'''
