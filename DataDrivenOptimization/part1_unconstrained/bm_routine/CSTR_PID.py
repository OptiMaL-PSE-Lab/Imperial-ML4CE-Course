import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
from pylab import grid
eps = np.finfo(float).eps
import copy
import time
import pickle

class CSTRSimulation:
    def __init__(self, repetitions=1):
        self.repetitions = repetitions
        self.traj_pid = {'t_c': [], 'T': [], 'Tc': [], 'Fin':[]}

        '''
        T is the control variable (Reactor temperature)
        Tc is the manipulated variable (Cooling jacket temperature)
        '''

        # Initial conditions for the states
        x0 = np.zeros(5)
        x0[0] = 0.87725294608097  # Initial concentration of A in CSTR (mol/m^3)
        x0[1] = 0.0  # Initial concentration of B in CSTR (mol/m^3)
        x0[2] = 0.0  # Initial concentration of C in CSTR (mol/m^3)
        x0[3] = 324.475443431599  # Initial temperature in CSTR (K)
        x0[4] = 100  # Initial volume in CSTR (m^3)

        # Time interval (min)
        n = 11  # number of intervals
        Tf = 25  # process time (min)
        t = np.linspace(0, Tf, n)
        
        # Store initial conditions and time
        self.data_res = {}
        self.data_res["x0"] = x0
        self.data_res["t"] = t
        self.data_res["n"] = n

        # Store results for plotting
        Ca = np.zeros(len(t))
        Ca[0] = x0[0]
        Cb = np.zeros(len(t))
        Cb[0] = x0[1]
        Cc = np.zeros(len(t))
        Cc[0] = x0[2]
        T = np.zeros(len(t))
        T[0] = x0[3]
        V = np.zeros(len(t))
        V[0] = x0[4]
        Tc = np.zeros(len(t) - 1)
        Fin = np.zeros(len(t) - 1)

        self.data_res["Ca_dat"] = copy.deepcopy(Ca)
        self.data_res["V_dat"] = copy.deepcopy(V)
        self.data_res["Cb_dat"] = copy.deepcopy(Cb)
        self.data_res["Tc_dat"] = copy.deepcopy(Tc)
        self.data_res["Cc_dat"] = copy.deepcopy(Cc)
        self.data_res["Fin_dat"] = copy.deepcopy(Fin)
        self.data_res["T_dat"] = copy.deepcopy(T)

        # noise level
        noise = 0  # [0,0.1,1]
        self.data_res["noise"] = noise

        # control upper and lower bounds
        self.data_res["Tc_ub"] = 302
        self.data_res["Tc_lb"] = 290
        self.data_res["Fin_ub"] = 105
        self.data_res["Fin_lb"] = 97

        # Step cooling temperature to (295 - 302)
        Tc = np.zeros(len(t) - 1)
        Tc[0:2] = 302.0
        Tc[2:4] = 295.0
        Tc[4:7] = 300.0
        Tc[7:] = 298.0

        # Step for volumetric flowrate at inlet (97,102)
        Fin = np.zeros(len(t) - 1)
        Fin[0:3] = 100
        Fin[3:6] = 97
        Fin[6:8] = 99
        Fin[8:] = 102

        self.data_res["Tc_ct"] = Tc
        self.data_res["Fin_ct"] = Fin

        # Bounds for optimization variables
        self.boundsK = np.array(
            [[0.0, 5.0]] * 6
            + [[-2, 2]] * 9
            + [[0.0, self.data_res["Tc_lb"]]]
            + [[0.0, 5.0]] * 6
            + [[-2, 2]] * 9
            + [[0.0, self.data_res["Fin_lb"]]]
        )

        #############################
        # process control operation #
        #############################

        # Time interval (min)
        n_c = 101  # 41#21 # 11
        Tf_c = 30  # 30 # process time (min) 25
        t_c = np.linspace(0, Tf_c, n_c)
        self.data_res["t_c"] = t_c

        # desired setpoints
        n_1 = int(n_c / 2)
        n_2 = n_c - n_1
        Cb_des = [0.2 for i in range(n_1)] + [0.2 for i in range(n_2)]
        T_des = [330 for i in range(n_1)] + [325 for i in range(n_2)]
        self.data_res["Cb_des"] = Cb_des
        self.data_res["T_des"] = T_des

        # Store results for plotting
        Ca = np.zeros(len(t_c))
        Ca[0] = copy.deepcopy(x0[0])
        Cb = np.zeros(len(t_c))
        Cb[0] = copy.deepcopy(x0[1])
        Cc = np.zeros(len(t_c))
        Cc[0] = copy.deepcopy(x0[2])
        T = np.zeros(len(t_c))
        T[0] = copy.deepcopy(x0[3])
        V = np.zeros(len(t_c))
        V[0] = copy.deepcopy(x0[4])
        Tc = np.zeros(len(t_c) - 1)
        Fin = np.zeros(len(t_c) - 1)
        self.data_res["Ca_ct"] = copy.deepcopy(Ca)
        self.data_res["V_ct"] = copy.deepcopy(V)
        self.data_res["Cb_ct"] = copy.deepcopy(Cb)
        self.data_res["Tc_ct"] = copy.deepcopy(Tc)
        self.data_res["Cc_ct"] = copy.deepcopy(Cc)
        self.data_res["Fin_ct"] = copy.deepcopy(Fin)
        self.data_res["T_ct"] = copy.deepcopy(T)

    def add_data_point(self, t_c, T, Tc, Fin):
        self.traj_pid['t_c'].append(t_c)
        self.traj_pid['T'].append(T)
        self.traj_pid['Tc'].append(Tc)
        self.traj_pid['Fin'].append(Fin)


    def cstr(self, x, t, Tc, Fin):
        Ca, Cb, Cc, T, V = x

        # Process parameters
        Tf, Caf, Fout, rho, Cp, UA = 350, 1, 100, 1000, 0.239, 5e4
        mdelH_AB, EoverR_AB, k0_AB = 5e4, 8750, 7.2e10
        mdelH_BC, EoverR_BC, k0_BC = 5e4, 10750, 8.2e10

        # Reaction rates
        rA = k0_AB * np.exp(-EoverR_AB / T) * Ca
        rB = k0_BC * np.exp(-EoverR_BC / T) * Cb

        # Derivatives
        dCadt = (Fin * Caf - Fout * Ca) / V - rA
        dCbdt = rA - rB - Fout * Cb / V
        dCcdt = rB - Fout * Cc / V
        dTdt = (Fin / V * (Tf - T) + mdelH_AB / (rho * Cp) * rA + mdelH_BC / (rho * Cp) * rB + UA / V / rho / Cp * (Tc - T))
        dVdt = Fin - Fout

        return np.array([dCadt, dCbdt, dCcdt, dTdt, dVdt])
    
    def plot_result_ct(self, Ks):
        data_plot = self.data_res
        repetitions = self.repetitions

        # Load data
        Ca, V, Cb, Tc, Cc, Fin, T, t_c = (copy.deepcopy(data_plot[key]) for key in ["Ca_ct", "V_ct", "Cb_ct", "Tc_ct", "Cc_ct", "Fin_ct", "T_ct", "t_c"])
        Cb_des, T_des = data_plot["Cb_des"], data_plot["T_des"]
        Tc_ub, Tc_lb, Fin_ub, Fin_lb = data_plot["Tc_ub"], data_plot["Tc_lb"], data_plot["Fin_ub"], data_plot["Fin_lb"]
        x0, noise = copy.deepcopy(data_plot["x0"]), data_plot["noise"]

        # creating lists
        Ca_dat = np.zeros((len(t_c), repetitions))
        Fin_dat = np.zeros((len(t_c) - 1, repetitions))
        Cb_dat = np.zeros((len(t_c), repetitions))
        Tc_dat = np.zeros((len(t_c) - 1, repetitions))
        Cc_dat = np.zeros((len(t_c), repetitions))
        error_dat = np.zeros((len(t_c) - 1, repetitions))
        V_dat = np.zeros((len(t_c), repetitions))
        u_mag_dat = np.zeros((len(t_c) - 1, repetitions))
        T_dat = np.zeros((len(t_c), repetitions))
        u_cha_dat = np.zeros((len(t_c) - 2, repetitions))

        # multiple runs
        for rep_i in range(repetitions):
            # re-loading data
            Ca, V, Cb, Cc, T, t_c, x0 = (copy.deepcopy(data_plot[key]) for key in ["Ca_ct", "V_ct", "Cb_ct", "Cc_ct",  "T_ct", "t_c", "x0"])

            # initiate
            x = x0
            e_history = []

            # main process simulation loop
            for i in range(len(t_c) - 1):
                # delta t
                ts = [t_c[i], t_c[i + 1]]
                # desired setpoint
                x_sp = np.array([x0[0], x0[1], x0[2], T_des[i], x0[4]])
                # compute control
                if i == 0:
                    Tc[i], Fin[i] = self.PID(Ks, x0, x_sp, np.array([[x0[0], Tc_lb, x0[2], Fin_lb, x0[4]]]))
                else:
                    Tc[i], Fin[i] = self.PID(Ks, x0, x_sp, np.array(e_history))
                # simulate reactor
                y = odeint(self.cstr, x, ts, args=(Tc[i], Fin[i]))
                # adding stochastic behaviour
                s = np.random.normal(0, 0.5, size=5)
                Ca[i + 1] = max([y[-1][0] * (1 + s[0] * 0.01 * noise), 0])
                Cb[i + 1] = max([y[-1][1] * (1 + s[1] * 0.01 * noise), 0])
                Cc[i + 1] = max([y[-1][2] * (1 + s[2] * 0.01 * noise), 0])
                T[i + 1] = max([y[-1][3] * (1 + s[3] * 0.01 * noise), 0])
                V[i + 1] = max([y[-1][4] * (1 + s[4] * 0.01 * noise), 0])
                # state update
                x[0] = Ca[i + 1]
                x[1] = Cb[i + 1]
                x[2] = Cc[i + 1]
                x[3] = T[i + 1]
                x[4] = V[i + 1]
                # compute tracking error
                e_history.append(x_sp - x)

            # == objective == #
            # production
            error = np.abs(np.array(e_history)[:, 0])
            error = error
            # penalize magnitud of control action
            u_mag = (Tc - Tc_lb) / (12) + (Fin - Fin_lb) / (8)
            u_mag = u_mag / 50
            # penalize change in control action
            u_cha = (Tc[1:] - Tc[0:-1]) ** 2 / (12) ** 2 + (Fin[1:] - Fin[0:-1]) ** 2 / (8) ** 2
            u_cha = u_cha / 50

            # data collection
            Ca_dat[:, rep_i] = copy.deepcopy(Ca)
            Fin_dat[:, rep_i] = copy.deepcopy(Fin)
            Cb_dat[:, rep_i] = copy.deepcopy(Cb)
            Tc_dat[:, rep_i] = copy.deepcopy(Tc)
            Cc_dat[:, rep_i] = copy.deepcopy(Cc)
            error_dat[:, rep_i] = copy.deepcopy(error)
            V_dat[:, rep_i] = copy.deepcopy(V)
            u_mag_dat[:, rep_i] = copy.deepcopy(u_mag)
            T_dat[:, rep_i] = copy.deepcopy(T)
            u_cha_dat[:, rep_i] = copy.deepcopy(u_cha)

        # Define the font properties for Times New Roman
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12,
            'axes.titlesize': 12,
            'axes.labelsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
        })

        # Plot the results
        plt.figure(figsize=(20, 7))  # Adjusted figsize to be wider for three plots in one row

        # Commenting out unnecessary subplots
        # plt.subplot(2, 4, 1)
        # plt.plot(t_c, np.median(Ca_dat, axis=1), "r-", lw=3)
        # plt.gca().fill_between(t_c, np.min(Ca_dat, axis=1), np.max(Ca_dat, axis=1), color="r", alpha=0.2)
        # plt.ylabel("A [mol m$^{-3}$]")
        # plt.xlabel("Time [min]")
        # plt.legend(["Concentration of A in CSTR"], loc="best")

        # plt.subplot(2, 4, 2)
        # plt.plot(t_c, np.median(Cb_dat, axis=1), "g-", lw=3)
        # plt.gca().fill_between(t_c, np.min(Cb_dat, axis=1), np.max(Cb_dat, axis=1), color="g", alpha=0.2)
        # plt.plot(t_c, Cb_des, "--", lw=2)
        # plt.ylabel("B [mol m$^{-3}$]")
        # plt.xlabel("Time [min]")
        # plt.legend(["Concentration of B in CSTR"], loc="best")

        # plt.subplot(2, 4, 3)
        # plt.plot(t_c, np.median(Cc_dat, axis=1), lw=3)
        # plt.gca().fill_between(t_c, np.min(Cc_dat, axis=1), np.max(Cc_dat, axis=1), alpha=0.2)
        # plt.ylabel("C [mol m$^{-3}$]")
        # plt.xlabel("Time [min]")
        # plt.legend(["Concentration of C in CSTR"], loc="best")

        plt.subplot(1, 3, 1)  # Adjusted for 3 plots in one row
        plt.plot(t_c, np.median(T_dat, axis=1), "c-", lw=3)
        plt.gca().fill_between(t_c, np.min(T_dat, axis=1), np.max(T_dat, axis=1), color="c", alpha=0.2)
        plt.plot(t_c, T_des, "--", lw=2)
        plt.ylabel("Temperature [K]")
        plt.xlabel("Time [min]")
        plt.legend(["Temperature in CSTR"], loc="best")

        # plt.subplot(2, 4, 5)
        # plt.plot(t_c, np.median(V_dat, axis=1), "m-", lw=3)
        # plt.gca().fill_between(t_c, np.min(V_dat, axis=1), np.max(V_dat, axis=1), color="m", alpha=0.2)
        # plt.ylabel("V [L]")
        # plt.xlabel("Time [min]")
        # plt.legend(["Volume in CSTR"], loc="best")

        plt.subplot(1, 3, 2)  # Adjusted for 3 plots in one row
        plt.plot(t_c[:-1], np.median(Fin_dat, axis=1), lw=3)
        plt.gca().fill_between(t_c[:-1], np.min(Fin_dat, axis=1), np.max(Fin_dat, axis=1), alpha=0.2)
        plt.ylabel("Fin [L min$^{-1}$]")
        plt.xlabel("Time [min]")
        plt.legend(["Flow rate into CSTR"], loc="best")

        plt.subplot(1, 3, 3)  # Adjusted for 3 plots in one row
        plt.plot(t_c[:-1], np.median(Tc_dat, axis=1), "k", lw=3)
        plt.gca().fill_between(t_c[:-1], np.min(Tc_dat, axis=1), np.max(Tc_dat, axis=1), color="k", alpha=0.2)
        plt.ylabel("Tc [K]")
        plt.xlabel("Time [min]")
        plt.legend(["Cooling Temperature"], loc="best")

        # plt.subplot(2, 4, 8)
        # plt.plot(t_c[:-1], np.median(error_dat, axis=1), "y-", lw=3)
        # plt.gca().fill_between(t_c[:-1], np.min(error_dat, axis=1), np.max(error_dat, axis=1), color="y", alpha=0.2)
        # plt.ylabel("Tracking Error")
        # plt.xlabel("Time [min]")
        # plt.legend(["Error"], loc="best")

        plt.tight_layout()
        plt.show()

        # Store the results in the instance variable
        self.results = {
            "Ca_dat": Ca_dat,
            "Fin_dat": Fin_dat,
            "Cb_dat": Cb_dat,
            "Tc_dat": Tc_dat,
            "Cc_dat": Cc_dat,
            "error_dat": error_dat,
            "V_dat": V_dat,
            "u_mag_dat": u_mag_dat,
            "T_dat": T_dat,
            "u_cha_dat": u_cha_dat,
        }

        #return self.results
    
    def cstr_ss(self, x, t, u1, u2):
        # ==  Inputs (2) == #
        Tc = u1  # Temperature of Cooling Jacket (K)
        Fin = u2  # Volumetric Flowrate at inlet (m^3/sec) = 100

        # == States (5) == #
        Ca = x[0]  # Concentration of A in CSTR (mol/m^3)
        Cb = x[1]  # Concentration of B in CSTR (mol/m^3)
        Cc = x[2]  # Concentration of C in CSTR (mol/m^3)
        T = x[3]  # Temperature in CSTR (K)
        V = x[4]  # Volume in CSTR (K)

        # == Process parameters == #
        Tf = 350  # Feed Temperature (K)
        Caf = 1  # Feed Concentration of A (mol/m^3)
        Fout = Fin  # Volumetric Flowrate at outlet (m^3/sec)
        # V       = 100    # Volume of CSTR (m^3)
        rho = 1000  # Density of A-B Mixture (kg/m^3)
        Cp = 0.239  # Heat Capacity of A-B-C Mixture (J/kg-K)
        UA = 5e4  # U -Heat Transfer Coefficient (W/m^2-K) A -Area - (m^2)
        # Reaction A->B
        mdelH_AB = 5e4  # Heat of Reaction for A->B (J/mol)
        EoverR_AB = 8750  # E -Activation Energy (J/mol), R -Constant = 8.31451 J/mol-K
        k0_AB = 7.2e10  # Pre-exponential Factor for A->B (1/sec)#
        rA = k0_AB * np.exp(-EoverR_AB / T) * Ca  # reaction rate
        # Reaction B->C
        mdelH_BC = 5e4  # Heat of Reaction for B->C (J/mol) => 5e4
        EoverR_BC = (
            10750  # E -Activation Energy (J/mol), R -Constant = 8.31451 J/mol-K !! 10
        )
        k0_BC = 8.2e10  # Pre-exponential Factor for A->B (1/sec)# !! 8
        rB = k0_BC * np.exp(-EoverR_BC / T) * Cb  # reaction rate !! **2
        # play with mdelH_BC, factor on Cb**2 and k0_BC, maybe even EoverR_BC

        # == Concentration Derivatives == #
        dCadt = (Fin * Caf - Fout * Ca) / V - rA  # A Concentration Derivative
        dCbdt = rA - rB - Fout * Cb / V  # B Concentration Derivative
        dCcdt = rB - Fout * Cc / V  # B Concentration Derivative
        dTdt = (
            Fin / V * (Tf - T)
            + mdelH_AB / (rho * Cp) * rA
            + mdelH_BC / (rho * Cp) * rB
            + UA / V / rho / Cp * (Tc - T)
        )  # Calculate temperature derivative
        dVdt = 0

        # == Return xdot == #
        xdot = np.zeros(5)
        xdot[0] = dCadt
        xdot[1] = dCbdt
        xdot[2] = dCcdt
        xdot[3] = dTdt
        xdot[4] = dVdt
        return xdot
    
    def PID(self, Ks, x, x_setpoint, e_history):

        '''
        returns the controll actions for the cooling jacket tmperature and the inlet flowrate
        '''

        Ks = np.array(Ks)
        Ks = Ks.reshape(32, order="C")
        # u_T gains for Ca, Cb, T, V, and bias
        KpCbT = Ks[0]
        KiCbT = Ks[1]
        KdCbT = Ks[2]
        KpTT = Ks[3]
        KiTT = Ks[4]
        KdTT = Ks[5]
        KpCaT = Ks[6]
        KiCaT = Ks[7]
        KdCaT = Ks[8]
        KpCcT = Ks[9]
        KiCcT = Ks[10]
        KdCcT = Ks[11]
        KpVT = Ks[12]
        KiVT = Ks[13]
        KdVT = Ks[14]
        KT = Ks[15]
        # u_F K gains for Ca, Cb, T, V, and bias
        KpCbF = Ks[16]
        KiCbF = Ks[17]
        KdCbF = Ks[18]
        KpTF = Ks[19]
        KiTF = Ks[20]
        KdTF = Ks[21]
        KpCaF = Ks[22]
        KiCaF = Ks[23]
        KdCaF = Ks[24]
        KpCcF = Ks[25]
        KiCcF = Ks[26]
        KdCcF = Ks[27]
        KpVF = Ks[28]
        KiVF = Ks[29]
        KdVF = Ks[30]
        KF = Ks[31]

        # setpoint error
        e = x_setpoint - x

        # control action Tc
        u_T = (
            KpCbT * e[1] + KiCbT * sum(e_history[:, 1]) + KdCbT * (e[1] - e_history[-1, 1])
        )
        u_T += KpTT * e[3] + KiTT * sum(e_history[:, 3]) + KdTT * (e[3] - e_history[-1, 3])
        u_T += KpCaT * e[0] + KiCaT * sum(e_history[:, 0]) + KdCaT * (e[0] - e_history[-1, 0])
        u_T += KpCcT * e[2] + KiCcT * sum(e_history[:, 2]) + KdCcT * (e[2] - e_history[-1, 2])
        u_T += KpVT * e[4] + KiVT * sum(e_history[:, 4]) + KdVT * (e[4] - e_history[-1, 4])
        u_T += KT
        u_T = min(max(u_T, self.data_res['Tc_lb']), self.data_res['Tc_ub'])

        # control action Fin
        u_F = (
            KpCbF * e[0] + KiCbF * sum(e_history[:, 0]) + KdCbF * (e[1] - e_history[-1, 0])
        )
        u_F += KpTF * e[3] + KiTF * sum(e_history[:, 3]) + KdTF * (e[3] - e_history[-1, 3])
        u_F += KpCaF * e[0] + KiCaF * sum(e_history[:, 0]) + KdCaF * (e[0] - e_history[-1, 0])
        u_F += KpCcF * e[2] + KiCcF * sum(e_history[:, 2]) + KdCcF * (e[2] - e_history[-1, 2])
        u_F += KpVF * e[4] + KiVF * sum(e_history[:, 4]) + KdVF * (e[4] - e_history[-1, 4])
        u_F += KF
        u_F =  min(max(u_F,  self.data_res['Fin_lb']), self.data_res['Fin_ub'])
        return u_T, u_F
    
    def J_ControlCSTR(self, Ks, full_output=False):

        '''
        This is the PID controller problem formulated as an objective function
        it takes the 32 controller gains (16 for the cooling jacket control action and 16 for the inlet flowrate control action)
        it returns the error + the penalty for the change in controller action as an array with the length being the number of discrete timesteps
        '''


        data_res = self.data_res
        # load data
        # state variables
        Ca = copy.deepcopy(data_res["Ca_ct"])
        Cb = copy.deepcopy(data_res["Cb_ct"])
        Cc = copy.deepcopy(data_res["Cc_ct"])
        T = copy.deepcopy(data_res["T_ct"])
        V = copy.deepcopy(data_res["V_ct"])


        Tc = copy.deepcopy(data_res["Tc_ct"])
        Fin = copy.deepcopy(data_res["Fin_ct"])
        t_c = copy.deepcopy(data_res["t_c"])
        x0 = copy.deepcopy(data_res["x0"])
        noise = data_res["noise"]
        # setpoints
        Cb_des = data_res["Cb_des"]
        T_des = data_res["T_des"]
        # upper and lower bounds
        Tc_ub = data_res["Tc_ub"]
        Tc_lb = data_res["Tc_lb"]
        Fin_ub = data_res["Fin_ub"]
        Fin_lb = data_res["Fin_lb"]

        # initiate
        x = x0
        e_history = []

        # main loop
        for i in range(len(t_c) - 1):
            '''
            This is the main loop where we simulate the system for every timestep t_c
            Hereby, the main loop produces e_history, which documents the deviation of the state variables to their setpoint over the simulated trajectory
            the number of entries corresponds to the number of discretized time-steps
            '''
            # delta t
            ts = [t_c[i], t_c[i + 1]]
            # desired setpoint
            x_sp = np.array([x0[0], Cb_des[i], x0[2], T_des[i], x0[4]])
            # compute control
            # everytime this objective function is called, it calculates an entire trajectory of cooling jacket temperature, and inlet flowrate. 
            # Then, based on these controller actions, the deviation of the reactor temperature set point is calculated and serves as the objective function value
            # starting from scratch
            if i == 0:
                Tc[i], Fin[i] = self.PID(Ks, x, x_sp, np.array([[x0[0], Tc_lb, x0[2], Fin_lb, x0[4]]]))
            # starting from a higher iteration
            else:
                Tc[i], Fin[i] = self.PID(Ks, x, x_sp, np.array(e_history))
            # simulate reactor
            y = odeint(self.cstr, x, ts, args=(Tc[i], Fin[i]))
            # adding stochastic behaviour with y being the reactor output for the state variables and s being the noise
            s = np.random.normal(0, 0.5, size=5)
            Ca[i + 1] = max([y[-1][0] * (1 + s[0] * 0.01 * noise), 0])
            Cb[i + 1] = max([y[-1][1] * (1 + s[1] * 0.01 * noise), 0])
            Cc[i + 1] = max([y[-1][2] * (1 + s[2] * 0.01 * noise), 0])
            T[i + 1] = max([y[-1][3] * (1 + s[3] * 0.01 * noise), 0])
            V[i + 1] = max([y[-1][4] * (1 + s[4] * 0.01 * noise), 0])
            # state update
            x[0] = Ca[i + 1]
            x[1] = Cb[i + 1]
            x[2] = Cc[i + 1]
            x[3] = T[i + 1]
            x[4] = V[i + 1]
            # compute tracking error as difference between setpoint and current values of the state variables 
            e_history.append(x_sp - x)

        self.add_data_point(t_c, T, Tc, Fin)

        # == objective == #
        # production
        # get the error in reactor Temperaure (column 4)
        error = np.abs(np.array(e_history)[:, 3])

        # penalize magnitude of control action
        u_mag = (Tc - Tc_lb) / (12) + (Fin - Fin_lb) / (8)
        u_mag = u_mag / 50
        # penalize change in control action
        u_cha = (Tc[1:] - Tc[0:-1]) ** 2 / (12) ** 2 + (Fin[1:] - Fin[0:-1]) ** 2 / (8) ** 2
        u_cha = u_cha / 50

        # store current step data in nested dictionary
        if full_output:
            # == outputs == #
            return t_c, Ca, Cb, Cc, T, V, Tc, Fin, error, u_mag, u_cha
        else:
            # == objective == #
            error = np.sum(error)
            u_mag = np.sum(u_mag)
            u_cha = np.sum(u_cha)
            return error + u_cha  # + u_mag
        

    def training_plot(self):
        # Setting up data for plotitng
        T_des = self.data_res["T_des"]
        temp_trajectories = self.traj_pid['T']
        tc_trajectories = self.traj_pid['Tc']
        time_span = self.traj_pid['t_c'][0]

        # Calculate alpha values using a cubic function
        alpha_values = ((np.arange(len(temp_trajectories)) / len(temp_trajectories))**2)

        plt.figure(figsize=(10, 12))

        # First subplot for temperature trajectories
        plt.subplot(2, 1, 1)
        for idx, trajectory in enumerate(temp_trajectories):
            alpha = alpha_values[idx]
            plt.plot(time_span, trajectory, color='blue', lw=1.5, alpha=alpha)
        plt.step(time_span, T_des, '--', lw=1.5, color='black')
        plt.ylabel('T (K)')
        plt.xlabel('Time (min)')
        plt.legend(['Reactor Temperature'], loc='upper right')#, frameon=False)
        plt.ylim([317, 335])
        plt.xlim([0, 30])
        plt.grid(True)

        # Second subplot for Tc trajectories
        plt.subplot(2, 1, 2)
        for idx, trajectory in enumerate(tc_trajectories):
            alpha = alpha_values[idx]
            plt.plot(time_span[:-1], trajectory, color='cyan', lw=1.5, alpha=alpha)

        plt.xlim([0, 30])
        plt.ylabel('Tc (K)')
        plt.xlabel('Time (min)')
        plt.legend(['Coolant Temperature'], loc='upper right')#, frameon=False)
        plt.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

# # Example Use
# iter_tot = 100
# CSTR = CSTRSimulation()
# Kpowell, f_opt, team_names, cids = opt_Powell(CSTR.J_ControlCSTR, int(32), CSTR.boundsK, iter_tot)
# print("Ks = ", Kpowell)
# CSTR.plot_result_ct(Kpowell)
# CSTR.training_plot()