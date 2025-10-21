# This is taken from [1]:
"""
Mendoza, D. F., Graciano, J. E. A., Dos Santos Liporace, F. & Le Roux, G. A. C.
Assessing the Reliability of Different Real-time Optimization Methodologies. The
Canadian Journal of Chemical Engineering 94, 485–497. issn: 0008-4034, 1939-019X.
(2024) (Mar. 2016).
"""

import numpy as np
from casadi import *


# solver options
options = {"disp": False, "maxiter": 10000}

# Parameters
Fa = 1.8275  # Mass flowrate Reactand A [kg/s]
Mt = 2105.2  # Total mass hold-up [kg]


class WO_system:

    def __init__(self):
        (
            self.xd,
            self.xa,
            self.u,
            self.ODEeq,
            self.Aeq,
            self.states,
            self.algebraics,
            self.inputs,
        ) = self.DAE_system()
        self.eval = self.integrator_system()
        self.f_list = []
        self.g_list = []
        self.x_list = []

    def DAE_system(self):
        """
        Algebraic variables:
        Xa - mass fraction reactand A [kg/kg]
        Xb - mass fraction reactand B [kg/kg]
        Xc - mass fraction reactand C [kg/kg]
        Xp - mass fraction product P [kg/kg]
        Xe - mass fraction product E [kg/kg]
        Xg - mass fraction product (waste) G [kg/kg]

        Inputs:
        Fb - mass flow rate of reactand B [kg/s]
        Tr - reactor tempterature [K]
        """

        # Define states
        states = ["x"]
        nd = len(states)
        # define vector containing symbolic
        # variables named xd for state derivatives
        xd = SX.sym("xd", nd)
        for i in range(nd):
            globals()[states[i]] = xd[i]

        # Define algebraics
        algebraics = ["Xa", "Xb", "Xc", "Xe", "Xp", "Xg"]
        na = len(algebraics)
        xa = SX.sym("xa", na)
        for i in range(na):
            globals()[algebraics[i]] = xa[i]

        # Define inputs
        inputs = ["Fb", "Tr"]
        nu = len(inputs)
        u = SX.sym("u", nu)
        for i in range(nu):
            globals()[inputs[i]] = u[i]

        # Reparametrization with plant parameters from [1]
        k1 = 1.6599e6 * np.exp(-6666.7 / (Tr + 273.15))
        k2 = 7.2117e8 * np.exp(-8333.3 / (Tr + 273.15))
        k3 = 2.6745e12 * np.exp(-11111.0 / (Tr + 273.15))

        # total mass flowrate
        Fr = Fa + Fb

        # reaction rate
        r1 = k1 * Xa * Xb * Mt
        r2 = k2 * Xb * Xc * Mt
        r3 = k3 * Xc * Xp * Mt

        # residual for x each algrbraic equation
        x_res = np.zeros((6, 1))
        x_res[0, 0] = (Fa - r1 - Fr * Xa) / Mt  # residual for mass balance reactand A
        x_res[1, 0] = (
            Fb - r1 - r2 - Fr * Xb
        ) / Mt  # residual for mass balance reactand B
        x_res[2, 0] = (
            +2 * r1 - 2 * r2 - r3 - Fr * Xc
        ) / Mt  # residual for mass balance reactand C
        x_res[3, 0] = (+2 * r2 - Fr * Xe) / Mt  # residual for mass balance product E
        x_res[4, 0] = (
            +r2 - 0.5 * r3 - Fr * Xp
        ) / Mt  # residual for mass balance product P
        x_res[5, 0] = (
            +1.5 * r3 - Fr * Xg
        ) / Mt  # residual for mass balance waste (product) G

        # Define vectors with banes of input variables
        ODEeq = [0 * x]

        # Declare algebraic equations
        Aeq = []
        Aeq += [(Fa - r1 - Fr * Xa) / Mt]  # mass balance reactand A
        Aeq += [(Fb - r1 - r2 - Fr * Xb) / Mt]  # mass balance reactand B
        Aeq += [(+2 * r1 - 2 * r2 - r3 - Fr * Xc) / Mt]  # mass balance reactand C
        Aeq += [(+2 * r2 - Fr * Xe) / Mt]  # mass balance product E
        Aeq += [(+r2 - 0.5 * r3 - Fr * Xp) / Mt]  # mass balance product P
        Aeq += [(+1.5 * r3 - Fr * Xg) / Mt]  # mass balance waste (product) G

        return xd, xa, u, ODEeq, Aeq, states, algebraics, inputs

    def integrator_system(self):
        """
        This function constructs the integrator to be
        suitable with casadi environment, for the equations
        of the model and the objective function with variable
        time step.

        inputs: NaN
        outputs: F: Function([x, u, dt]--> [xf, obj])
        """

        xd, xa, u, ODEeq, Aeq, states, algebraics, inputs = self.DAE_system()
        VV = Function("vfcn", [xa, u], [vertcat(*Aeq)], ["w0", "u"], ["w"])
        solver = rootfinder("solver", "newton", VV)

        return solver

    def WO_obj_sys_ca(self, u):
        """
        inputs: mass flowrate reactand b u[0] and reactor temperature u[1]
        intermediates: solutions of DAE system: x, containing  mass fractions of desired products P (x[4]) and E (x[3])
        outputs: obj (profit) for intermediates given inputs
        """

        # Solve DAE system for the given states and input u
        x = self.eval(
            np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u
        )
        Fb = u[0]
        Tr = u[
            1
        ]  # The reactor temperature Tr has no direct impact on the profit objective function, but on the reaction kinetiks.
        Fa = 1.8275
        Fr = Fa + Fb

        obj = -(
            1043.38 * x[4] * Fr + 20.92 * x[3] * Fr - 79.23 * Fa - 118.34 * Fb
        ) + 0.5 * np.random.normal(0.0, 1)

        return float(obj)

    def WO_obj_sys_ca_noise_less(self, u):
        x = self.eval(
            np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u
        )

        # read mass flowrate B
        Fb = u[0]

        # calculate total mass flowrate (inlet and outlet)
        Fr = Fa + Fb

        # calculate profit flow
        obj = -(1043.38 * x[4] * Fr + 20.92 * x[3] * Fr - 79.23 * Fa - 118.34 * Fb)

        self.f_list.append(float(obj))
        self.x_list.append(u)

        return float(obj)

    def WO_con1_sys_ca(self, u):
        """
        mass fraction of reactand A in the outlet less or equal 12 %mass
        """
        x = self.eval(
            np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u
        )
        pcon1 = x[0] - 0.12 + 5e-4 * np.random.normal(0.0, 1)

        return float(pcon1.toarray()[0])

    def WO_con2_sys_ca(self, u):
        """
        mass fraction of waste G in the outlet less or equal 8 %mass
        """
        x = self.eval(
            np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u
        )
        pcon2 = x[5] - 0.08 + 5e-4 * np.random.normal(0.0, 1)

        return float(pcon2.toarray()[0])

    def WO_con1_sys_ca_noise_less(self, u):
        x = self.eval(
            np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u
        )
        pcon1 = x[0] - 0.12  # + 5e-4*np.random.normal(0., 1)
        self.g_list.append(float(pcon1.toarray()[0]))

        return float(pcon1.toarray()[0])

    def WO_con2_sys_ca_noise_less(self, u):
        x = self.eval(
            np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u
        )
        pcon2 = x[5] - 0.08  # + 5e-4*np.random.normal(0., 1)
        self.g_list.append(float(pcon2.toarray()[0]))

        return float(pcon2.toarray()[0])
