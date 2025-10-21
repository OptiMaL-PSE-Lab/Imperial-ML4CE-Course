# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:00:56 2023

@author: adelrioc
"""

import numpy as np
import copy
from ML4CE_WO import *

##########################################
### threshold for constraint violation ###
##########################################

# This was originally suggested to be 0.1
# I changed it to 0.001 to count
vio = 0.001


class Test_function:
    """
    This is the test function class, that contains
    - the function itself
    - the constraints per function
    - the starting points per function
    """

    #################################
    # --- initializing function --- #
    #################################
    def __init__(
        self,
        func_type,
        n_x,
        track_x,
    ):

        # initialize lists
        self.f_list = []
        self.x_list = []
        self.g_list = []
        self.g_value_list = []

        # for WO
        self.g_list1 = []
        self.g_list2 = []

        # function specifications
        self.func_type = func_type
        self.n_x = n_x
        self.track_x = track_x

        if self.func_type == "WO_f":

            self.x0 = np.array(
                (
                    [[6.9], [83]],
                    [[6.75], [72.5]],
                    [[6.75], [75]],
                    [[6], [75]],
                    [[6], [80]],
                    [[5.5], [77.5]],
                    [[5.5], [82.5]],
                    [[6.75], [87.5]],
                    [[6.25], [85]],
                    [[6.9], [91]],
                )
            )

    ################
    # run function #
    ################

    def fun_test(self, x):

        #################
        # WO case-stuy ##
        #################

        if self.func_type == "WO_f":

            x = np.array(x).flatten()

            # here we need the objective function now
            WO_instance = WO_system()
            # z = WO_instance.WO_obj_sys_ca_noise_less(x)
            z = WO_instance.WO_obj_sys_ca(x)
            # track f
            self.f_list.append(z)
            if self.track_x:
                self.x_list.append(x.flatten())

            # return objective
            return z

    ##################
    # run constraint #
    ##################

    def con_test(self, x):

        ################################
        ## Constraints Williams Otto ###
        ################################

        if self.func_type == "WO_f":

            WO_instance = WO_system()

            # g1 = WO_instance.WO_con1_sys_ca_noise_less(x)
            # g2 = WO_instance.WO_con2_sys_ca_noise_less(x)
            g1 = WO_instance.WO_con1_sys_ca(x)
            g2 = WO_instance.WO_con2_sys_ca(x)

            if isinstance(g1, float) and g1 > vio:

                # track g
                self.g_list1.append(g1)

            else:
                self.g_list1.append(None)

            if isinstance(g2, float) and g2 > vio:

                # track g
                self.g_list2.append(g2)

            else:
                self.g_list2.append(None)
            self.g_value_list.append([g1, g2])
            return [-g1, -g2]

    ########################
    # FOR BO-related algorithms ##
    # run constraint for WO#
    #######################

    def WO_con1_test(self, x):
        WO_instance_con1 = WO_system()
        con1_val = WO_instance_con1.WO_con1_sys_ca_noise_less(x)
        # con1_val = WO_instance_con1.WO_con1_sys_ca(x)
        if isinstance(con1_val, float) and con1_val > vio:

            # track g
            self.g_list1.append(con1_val)
            self.g_list.append(
                con1_val
            )  # These are still added because they're needed for the statistics in utils ML4CE_con_table

        else:
            self.g_list1.append(None)
            self.g_list.append(None)

        return -con1_val

    def WO_con2_test(self, x):
        WO_instance_con2 = WO_system()

        con2_val = WO_instance_con2.WO_con2_sys_ca_noise_less(x)
        # con2_val = WO_instance_con2.WO_con2_sys_ca(x)
        if isinstance(con2_val, float) and con2_val > vio:

            # track g
            self.g_list2.append(con2_val)
            self.g_list.append(con2_val)

        else:
            self.g_list2.append(None)
            self.g_list.append(None)

        return -con2_val

    ###################
    # plot constraint #
    ###################

    def WO_con1_plot(self, x):
        """
        6.92x1+49.815-x2<=0
        """
        x1 = x
        return 6.55 * x1 + 51.84

    def WO_con2_plot(self, x):
        """
        -1.09052x1^2+4.32428x1+84.1511 - x2 <= 0
        1.14707 x^2-18.2376 x+138.597 - x2 <= 0
        """
        x1 = x
        return 1.14707 * x1**2 - 18.2376 * x1 + 138.537

    ####################
    # re-arrange lists #
    ####################

    def best_f_list(self):
        """
        Returns
        -------
        List of best points so far
        """
        # self.best_f = [min(self.f_list[:i]) for i in range(1,len(self.f_list))]
        self.best_f = [min(self.f_list[: i + 1]) for i in range(len(self.f_list))]
        self.best_x = [self.x_list[self.f_list.index(f)] for f in self.best_f]

        ######### NOT USED #########
        # feasible only
        self.f_feas = [
            f for f in self.f_list if self.g_list[self.f_list.index(f)] is None
        ]
        self.x_feas = [self.x_list[self.f_list.index(f)] for f in self.f_feas]
        # make sure to adjust_lists before this one as g_list and f_list might not have the same length
        self.best_f_feas = [min(self.f_feas[: i + 1]) for i in range(len(self.f_feas))]
        self.best_x_feas = [self.x_list[self.f_list.index(f)] for f in self.best_f_feas]

    #############
    # cut lists #
    #############

    def pad_or_truncate(self, n_p):
        """
        n_p: number of desired elements on list
        -------
        Truncate or pad list
        """
        # get last element
        try:  # in case called without best_f_list() before
            b_last = copy.deepcopy(self.best_f[:n_p])[-1]
            self.best_f_c = copy.deepcopy(self.best_f[:n_p]) + [b_last] * (
                n_p - len(self.best_f[:n_p])
            )
        except:
            b_last = None

        l_last = copy.deepcopy(self.f_list[:n_p])[-1]
        x_last = copy.deepcopy(self.x_list[:n_p])[-1]

        self.f_list_c = copy.deepcopy(self.f_list[:n_p]) + [l_last] * (
            n_p - len(self.f_list[:n_p])
        )
        self.x_list_c = copy.deepcopy(self.x_list[:n_p]) + [x_last] * (
            n_p - len(self.x_list[:n_p])
        )
