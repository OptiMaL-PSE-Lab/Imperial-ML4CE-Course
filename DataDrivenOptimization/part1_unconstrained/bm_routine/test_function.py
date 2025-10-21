# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:00:56 2023

@author: adelrioc
"""

import numpy as np
import copy
from bm_routine import *


class Test_function:
    #################################
    # --- initializing function --- #
    #################################
    def __init__(self, func_type, n_x, track_x, x_shift, bounds):
        """
        x_shift: this is the starting point that is set for each repitition (5 per algorithm and function)
        it was randomly chosen for each dimension between -3 and 3
        """

        # initialize lists
        self.f_list = []
        self.x_list = []
        self.traj_pid = {'T':[], 'Tc':[], 't_c':[], 'Fin':[]}

        # function specifications
        self.func_type = func_type
        self.n_x = n_x
        self.track_x = track_x
        self.x_shift = x_shift

        # for opt_DYCORS !!
        self.dim = n_x
        self.lb = bounds[:, 0]
        self.ub = bounds[:, 1]
        self.int_var = np.array([])
        self.cont_var = np.arange(0, n_x)
        self.history = []

        if self.func_type == "Rosenbrock_f":
            self.x0 = np.array(
                ([[-3.5], [4]], 
                 [[-3.5], [4]],
                 [[-3.5], [4]],
                 [[-3.5], [4]],
                 [[-3.5], [4]],
                 [[-4.7], [0]], 
                 [[-1], [1]]),
            )

            print("remember to reset the starting values in test function")

            self.init_points = np.array(
                [
                    [  # TODO these are manually copied random points and need to be automated
                        [-3.5, 4],
                        [-3.695999175468267, 4.298600636679204],
                        [-3.324485345153229, 4.34397331429392],
                        [-3.6551956617560606, 4.445719043701324],
                        [-3.2894861583353707, 4.372417455324243],
                        [-3.1358366982310306, 3.7318957927275074],
                        [-3.3750653613706514, 4.031628585876453],
                        [-3.6560822903968844, 4.2695814211741565],
                        [-3.402599951172502, 4.290894965399081],
                        [-3.58554000313634, 3.8960851498181146],
                    ],
                    [
                        [-4.7, 0.0],
                        [-4.872829217553934, 0.07939168757488668],
                        [-4.596064847385586, -0.00678504534014468],
                        [-4.3628806545158065, 0.13337499704552347],
                        [-4.900695899797675, 0.32541746346155087],
                        [-4.6986158824203, 0.42788750770802386],
                        [-4.855153739318994, 0.07256667216710011],
                        [-4.371620789116707, -0.2625845865624066],
                        [-4.417138114611893, 0.1138350903311236],
                        [-4.968454704380232, -0.011770250298933527],
                    ],
                    [
                        [-1, 1],
                        [-0.9094882325785648, 1.0522645964081918],
                        [-0.8257802160551404, 1.2511523667547042],
                        [-1.073769078982123, 1.437282168432578],
                        [-1.0035681771364549, 0.7201507013205384],
                        [-1.0956277155456018, 1.2812525332988598],
                        [-0.7949204439192279, 0.5982623762449268],
                        [-0.6452272945520198, 1.2229489570716696],
                        [-0.8448267653830063, 0.875666652279921],
                        [-0.5734729013792419, 1.2194629193024646],
                    ],
                ]
            )

        if self.func_type == "Antonio_f":
            self.x0 = np.array(
                ([[-4.3],[-2.5]], 
                 [[-4.3],[-2.5]],
                 [[-4.3],[-2.5]],
                 [[-4.3],[-2.5]],
                 [[-4.3],[-2.5]],
                 [[-3], [1]], 
                 [[1], [4]]),
            )

            self.init_points = np.array(
                [
                    [
                        [-4, -4],
                        [-4.133735443252803, -3.9070889874067998],
                        [-4.2554621696650266, -3.839167539044557],
                        [-3.6991499017924774, -4.067343702881785],
                        [-4.2204588351029315, -3.9322351439533074],
                        [-3.8665994738407723, -4.294931903565754],
                        [-3.5352717073354922, -3.93764772917438],
                        [-4.388286226588492, -3.971974085341454],
                        [-3.8820488343028465, -4.3689041148478545],
                        [-4.2178130348809955, -3.603861244697474],
                    ],
                    [
                        [-3, 1],
                        [-3.434556216336288, 1.2060580959472942],
                        [-3.1610557040339065, 1.219792127067329],
                        [-4.3628806545158065, 1.4144437341072003],
                        [-3.1908791866381243, 0.6233448076862496],
                        [-3.3527141888662424, 1.1464401407819036],
                        [-2.648462121006375, 0.9002924215843959],
                        [-2.961973468399645, 0.8925521968287324],
                        [-3.3935600071374625, 0.8615225940317944],
                        [-3.364708378971091, 1.2354171389730733],
                    ],
                    [
                        [1, 4],
                        [1.0347934616792243, 4.345938241368554],
                        [1.351421611414327, 3.925567990718291],
                        [1.4884654332087441, 4.092294655773182],
                        [0.982431117780894, 3.565937812134032],
                        [0.6798258164392602, 4.189425695691275],
                        [0.8496647453266832, 3.550606273479979],
                        [1.1510961542739127, 4.1176517876307726],
                        [1.261537220618603, 3.755053800587087],
                        [1.045742654831773, 3.52680512392919],
                    ],
                ]
            )

        if self.func_type == "Ackley_f":
            self.x0 = np.array(
                ([[-3], [3]], 
                 [[-3], [3]], 
                 [[-3], [3]], 
                 [[-3], [3]], 
                 [[-3], [3]], 
                 [[-3], [1]], 
                 [[1], [4]]),
            )
            self.init_points = np.array(
                [
                    [
                        [-3, 3],
                        [-3.133735443252803, 3.9070889874067998],
                        [-3.2554621696650266, 3.839167539044557],
                        [-3.6991499017924774, 4.067343702881785],
                        [-3.2204588351029315, 3.9322351439533074],
                        [-2.8665994738407723, 3.294931903565754],
                        [-3.5352717073354922, 3.93764772917438],
                        [-3.388286226588492, 3.971974085341454],
                        [-2.8820488343028465, 3.3689041148478545],
                        [-3.2178130348809955, 3.603861244697474],
                    ],
                    [
                        [-3, 1],
                        [-3.434556216336288, 1.2060580959472942],
                        [-3.1610557040339065, 1.219792127067329],
                        [-4.3628806545158065, 1.4144437341072003],
                        [-3.1908791866381243, 0.6233448076862496],
                        [-3.3527141888662424, 1.1464401407819036],
                        [-2.648462121006375, 0.9002924215843959],
                        [-2.961973468399645, 0.8925521968287324],
                        [-3.3935600071374625, 0.8615225940317944],
                        [-3.364708378971091, 1.2354171389730733],
                    ],
                    [
                        [1, 4],
                        [1.0347934616792243, 4.345938241368554],
                        [1.351421611414327, 3.925567990718291],
                        [1.4884654332087441, 4.092294655773182],
                        [0.982431117780894, 3.565937812134032],
                        [0.6798258164392602, 4.189425695691275],
                        [0.8496647453266832, 3.550606273479979],
                        [1.1510961542739127, 4.1176517876307726],
                        [1.261537220618603, 3.755053800587087],
                        [1.045742654831773, 3.52680512392919],
                    ],
                ]
            )

        if self.func_type == "Levy_f":
            self.x0 = np.array(
                ([[-3], [3]], 
                 [[-3], [3]], 
                 [[-3], [3]], 
                 [[-3], [3]], 
                 [[-3], [3]], 
                 [[-3], [1]], 
                 [[1], [4]]),
            )
            self.init_points = np.array(
                [
                    [
                        [-3, 3],
                        [-3.133735443252803, 3.9070889874067998],
                        [-3.2554621696650266, 3.839167539044557],
                        [-3.6991499017924774, 4.067343702881785],
                        [-3.2204588351029315, 3.9322351439533074],
                        [-2.8665994738407723, 3.294931903565754],
                        [-3.5352717073354922, 3.93764772917438],
                        [-3.388286226588492, 3.971974085341454],
                        [-2.8820488343028465, 3.3689041148478545],
                        [-3.2178130348809955, 3.603861244697474],
                    ],
                    [
                        [-3, 1],
                        [-3.434556216336288, 1.2060580959472942],
                        [-3.1610557040339065, 1.219792127067329],
                        [-4.3628806545158065, 1.4144437341072003],
                        [-3.1908791866381243, 0.6233448076862496],
                        [-3.3527141888662424, 1.1464401407819036],
                        [-2.648462121006375, 0.9002924215843959],
                        [-2.961973468399645, 0.8925521968287324],
                        [-3.3935600071374625, 0.8615225940317944],
                        [-3.364708378971091, 1.2354171389730733],
                    ],
                    [
                        [1, 4],
                        [1.0347934616792243, 4.345938241368554],
                        [1.351421611414327, 3.925567990718291],
                        [1.4884654332087441, 4.092294655773182],
                        [0.982431117780894, 3.565937812134032],
                        [0.6798258164392602, 4.189425695691275],
                        [0.8496647453266832, 3.550606273479979],
                        [1.1510961542739127, 4.1176517876307726],
                        [1.261537220618603, 3.755053800587087],
                        [1.045742654831773, 3.52680512392919],
                    ],
                ]
            )

        if self.func_type == "cstr_pid_f":
            self.x0 = np.array(
                [
                    4.46715341e-05,
                    4.99993980e00,
                    4.99995537e00,
                    2.03103290e00,
                    3.00941720e00,
                    1.18583680e00,
                    1.99993758e00,
                    1.99993758e00,
                    1.99993758e00,
                    1.99993758e00,
                    1.99993758e00,
                    1.99993758e00,
                    1.99993758e00,
                    1.99993758e00,
                    1.99993758e00,
                    1.83520459e02,
                    4.99995535e00,
                    1.46760347e00,
                    2.39717774e00,
                    3.14432454e00,
                    1.79223282e00,
                    8.76534798e-01,
                    1.52490607e-01,
                    4.08452593e-01,
                    -9.30694191e-01,
                    -1.39481735e00,
                    -6.57091280e-01,
                    -1.85731238e00,
                    1.85499957e00,
                    1.57295435e00,
                    -1.54164575e00,
                    3.80361850e01,
                ],
            )

    ################
    # run function #
    ################

    def fun_test(self, x):
        if self.func_type == "Rosenbrock_f":
            """
            Unimodal_function
            Test function: Rosenbrock function (http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html)
            Note: Website does not work anymore (checked Feb_24_2024), this one does: https://www.sfu.ca/~ssurjano/rosen.html
            Note:
                works only for vectors

            
            x: this is the argument that the function gets during the evaluation.
            The way this benchmarking is coded is, that there is a coordinate shift for each repitition to x_shift. Meaning all 
            optimisation algorithms provide solutions based on x_shift. In order to receive the true value of the objective
            function during evaluation, the candidate x has to be added to x_shift. 
            """

            x = np.array(x)
            x = x.reshape((self.n_x, 1))
            x = x + self.x_shift
            z = (1.0 - x).T @ (1.0 - x) + 100 * (x[1:] - x[:-1] ** 2).T @ (
                x[1:] - x[:-1] ** 2
            )

            # track f
            self.f_list.append(z[0, 0])
            # track x
            if self.track_x:
                self.x_list.append(x)

            # return objective
            return z[0, 0]

        elif self.func_type == "Levy_f":
            """
            Multimodal function
            Test function: Levy function (https://www.sfu.ca/~ssurjano/levy.html)
            Note:
                works only for vectors
            """
            x = np.array(x)
            x = x.reshape((self.n_x, 1))

            x = x + self.x_shift
            w = 1.0 + (x - 1.0) / 4.0
            z = (
                np.sin(np.pi * w[0]) ** 2
                + np.sum(
                    (w[:-1] - 1.0) ** 2
                    * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2)
                )
                + (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]))
            )

            # track f
            self.f_list.append(z[0])
            # track x
            if self.track_x:
                self.x_list.append(x)

            # return objective
            return z[0]

        elif self.func_type == "Rastringin_f":
            """
            Multimodal function 
            Test function: Rastringin function (https://www.sfu.ca/~ssurjano/rastr.html)
            Note:
                works only for vectors
            """
            x = np.array(x)
            x = x.reshape((self.n_x, 1))
            x = x + self.x_shift
            z = 10.0 * self.n_x + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))

            # track f
            self.f_list.append(z)
            # track x
            if self.track_x:
                self.x_list.append(x)

            # return objective
            return z

        elif self.func_type == "Antonio_f":
            """
            Unimodal function
            """

            a = 1.9
            d = self.n_x
            x = np.array(x)
            x = x.reshape((self.n_x, 1))
            x = x + self.x_shift
            i = np.arange(1, d + 1)

            z = np.sum((i * x) ** 2 + (a * i / d) * x * x[-1])

            # track f
            self.f_list.append(z)
            # track x
            if self.track_x:
                self.x_list.append(x)

            return z

        elif self.func_type == "Ackley_f":
            """
            Test function: Ackely function (https://www.sfu.ca/~ssurjano/ackley.html)
            Note:
                works only for vectors
            """
            x = np.array(x)
            x = x.reshape((self.n_x, 1))
            x = x + self.x_shift

            a = 20.0
            b = 0.2
            c = 2.0 * np.pi

            x = x.reshape((self.n_x, 1))
            z = (
                -a * np.exp(-b * np.sqrt(1.0 / self.n_x * np.sum(x**2)))
                - np.exp(1.0 / self.n_x * np.sum(np.cos(c * x)))
                + a
                + np.exp(1.0)
            )

            # track f
            self.f_list.append(z)
            # track x
            if self.track_x:
                self.x_list.append(x)

            # return objective
            return z

        elif self.func_type == "cstr_pid_f":
            """
            Test function: cstr_pid case study function
            """
            # here we need the objective function now
            CSTR_PID_instance = CSTRSimulation()
            z = CSTR_PID_instance.J_ControlCSTR(x)

            self.traj_pid['T'].append(CSTR_PID_instance.traj_pid['T'])
            self.traj_pid['Tc'].append(CSTR_PID_instance.traj_pid['Tc'])
            self.traj_pid['t_c'].append(CSTR_PID_instance.traj_pid['t_c'])
            self.traj_pid['Fin'].append(CSTR_PID_instance.traj_pid['Fin'])

            # track f
            self.f_list.append(z)
            if self.track_x:
                self.x_list.append(x)

# return objective
            return z

    ####################
    # re-arrange lists #
    ####################

    def best_f_list(self):
        """
        Returns
        -------
        List of best points so far
        """
        self.best_f = [min(self.f_list[:i+1]) for i in range(len(self.f_list))]
        self.best_x = [self.x_list[self.f_list.index(f)] for f in self.best_f]

    #############
    # cut lists #
    #############

    def pad_or_truncate(self, n_p):
        """
        n_p: number of desired elements on list
        if the list is to long it truncates, if for some
        reason the algorithm returns less than the maximum budget (which is possible)
        it simply 'padds' with the best element known
        -------
        Truncate or pad list
        """
        # get last element
        b_last = copy.deepcopy(self.best_f[:n_p])[-1]
        l_last = copy.deepcopy(self.f_list[:n_p])[-1]

        # pad or truncate
        self.best_f_c = copy.deepcopy(self.best_f[:n_p]) + [b_last] * (
            n_p - len(self.best_f[:n_p])
        )
        self.f_list_c = copy.deepcopy(self.f_list[:n_p]) + [l_last] * (
            n_p - len(self.f_list[:n_p])
        )


# TODO -> plot every algorithm trajectory with best, meadian, 25% and 75%
# computer best trajectory inside class algorithm
# compute all statistics later.
