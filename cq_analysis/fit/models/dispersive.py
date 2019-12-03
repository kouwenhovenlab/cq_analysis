import numpy as np
import lmfit

class DispersiveHangerModel(lmfit.Model):
    @staticmethod
    def hanger(tc, e, wr, gc, k1, k2, ki, w0, gamma):
        return 1 - 0.5 * (k1 + 1j * k2) * DispersiveHangerModel.displacement(tc, e, wr, gc, k1, 0, ki, w0, gamma)

    @staticmethod
    def reflection(tc, e, wr, gc, k1, k2, ki, w0, gamma):
        return -k1 * DispersiveHangerModel.displacement(tc, e, wr, gc, k1, k2, ki, w0, gamma) + 1

    @staticmethod
    def transmission(tc, e, wr, gc, k1, k2, ki, w0, gamma):
        return -np.sqrt(k1 * k2) * DispersiveHangerModel.displacement(tc, e, wr, gc, k1, k2, ki, w0, gamma)

    @staticmethod
    def displacement(tc, e, wr, gc, k1, k2, ki, w0, gamma):
        omega_s = 4 * tc ** 2 + e ** 2
        cg = 4 * gc ** 2 * tc ** 2 / (omega_s * (wr + 1j * gamma / 2 - np.sqrt(omega_s)))
        return -1j / (-1j / 2 * (k1 + k2 + ki) - wr + w0 + cg)

    @staticmethod
    def func(r, tc, gc, k1, k2, ki, w0, gamma, e_conv, detuning_offset, slope, offset, ampl_slope, ampl_offset):
        e = r[0] - detuning_offset
        wr = r[1]
        S = DispersiveHangerModel.hanger(tc=np.absolute(tc),
                                   e=e * e_conv,
                                   wr=wr,
                                   gc=gc,
                                   k1=np.absolute(k1),
                                   k2=k2,
                                   ki=np.absolute(ki),
                                   w0=np.absolute(w0),
                                   gamma=gamma)
        S = np.conj(S)
        S *= ampl_offset * (1 + ampl_slope * (wr - w0) / w0)
        S *= np.exp(1j * (offset + slope * (wr - wr[0][0])))

        return S.ravel()

    @staticmethod
    def A(r, tc, gc, k1, k2, ki, w0, gamma, e_conv, detuning_offset, slope, offset, ampl_slope, ampl_offset):
        e = (r[0] - detuning_offset) * e_conv
        wr = r[1]
        omega_s = 4 * tc ** 2 + e ** 2
        geff = gc * 2 * tc / np.sqrt(omega_s)
        A = geff ** 2 / (gamma ** 2 / 4 + (np.sqrt(omega_s) - w0) ** 2)
        return A

    @staticmethod
    def w0s(r, tc, gc, k1, k2, ki, w0, gamma, e_conv, detuning_offset, slope, offset, ampl_slope, ampl_offset):
        e = (r[0] - detuning_offset) * e_conv
        wr = r[1]
        A = DispersiveHangerModel.A(r, tc, gc, k1, k2, ki, w0, gamma, e_conv, detuning_offset, slope, offset, ampl_slope,
                              ampl_offset)
        omega_s = 4 * tc ** 2 + e ** 2
        return (w0 - A * np.sqrt(omega_s)) / (1 - A)

    @staticmethod
    def k1s(r, tc, gc, k1, k2, ki, w0, gamma, e_conv, detuning_offset, slope, offset, ampl_slope, ampl_offset):
        A = DispersiveHangerModel.A(r, tc, gc, k1, k2, ki, w0, gamma, e_conv, detuning_offset, slope, offset, ampl_slope,
                              ampl_offset)
        return k1 / (1 - A)

    @staticmethod
    def kis(r, tc, gc, k1, k2, ki, w0, gamma, e_conv, detuning_offset, slope, offset, ampl_slope, ampl_offset):
        A = DispersiveHangerModel.A(r, tc, gc, k1, k2, ki, w0, gamma, e_conv, detuning_offset, slope, offset, ampl_slope,
                              ampl_offset)
        return (ki + A * gamma) / (1 - A)

    def __init__(self):
        super().__init__(self.func)

    def guess(self, data, gatedata, freqdata):
        p0 = self.make_params(
            tc=5e9,
            gc=30e6,
            k1=8456310,
            k2=3447530.,
            ki=1,
            w0=np.average(freqdata),
            gamma=2e10,
            e_conv=1.7888858e12,
            detuning_offset=np.average(gatedata),
            slope=-3.6126e-07,
            offset=0.7,
            ampl_slope=-6.50322269,
            ampl_offset=0.34895203)
        p0['tc'].min = 0
        p0['e_conv'].vary = False
        p0['ki'].min = 0
        p0['k1'].min = 0
        p0['gamma'].min = 0
        # p0['slope'].vary = False
        # p0['k1'].vary = False
        # p0['k2'].vary = False
        # p0['w0'].vary = False
        return p0