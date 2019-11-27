"""lmfit models for resonator responses"""

import numpy as np
import lmfit
import scipy.constants as constants

e2h = constants.e**2/constants.h
twopi = np.pi * 2

class Qi_vs_conductance(lmfit.Model):
    """
    Calculates internal quality factor of a textbook
    rf-conductance circuit:

    ---- Resistor ------ Inductor -------------------
                                    |               |
                                    |               |
                                Parasitic        Device
                                capacitance      resistance
                                    |               |
                                    |               |
                                  Ground          Ground

    Parameters:
    - G - nanowire conductance (SI units)
    - L - inductance
    - C - (parasitic) capacitance
    - R - resistance accounting for losses other than the device
    """

    @staticmethod
    def func(G,L,C,R):
        return Helpers.Qi_fun(G,L,C,R)

    def __init__(self):
        super().__init__(self.func)

class Qi_vs_conductance_log(Qi_vs_conductance):
    """
    Same as Qi_vs_conductance, except calculates log(Qi)
    for fitting in semi-log scale.
    """
    @staticmethod
    def func(G,L,C,R):
        return np.log(Helpers.Qi_fun(G,L,C,R))


########## Class with helper functions ##########
class Helpers():
    # impedance of a RLC circuit with low-conductance component
    @staticmethod
    def Z_fun(G,L,C,R,omg):
        return R + 1j*omg*L + 1/(G + 1j*omg*C)

    # resonant frequency
    @staticmethod
    def omg_fun(G,L,C):
        return np.sqrt(C/L + G**2)/C
    @staticmethod
    def f_fun(G,L,C):
        return Helpers.omg_fun(G,L,C)/2/np.pi

    # effective capacitance
    @staticmethod
    def C_fun(G,L,C):
        pt1 = (Helpers.omg_fun(G,L,C)**2 * C**2 + G**2)
        return pt1/C/Helpers.omg_fun(G,L,C)**2

    # effective resistance (RLC+QPC converted to effective RLC)
    @staticmethod
    def R_fun(G,L,C,R):
        return R+G/(Helpers.omg_fun(G,L,C)**2 * C**2 + G**2)

    # characteristic impedance
    @staticmethod
    def Zc_fun(G,L,C):
        return np.sqrt(L/ Helpers.C_fun(G,L,C))
        
    # internal quality factor
    @staticmethod
    def Qi_fun(G,L,C,R):
        return Helpers.Zc_fun(G,L,C)/Helpers.R_fun(G,L,C,R)