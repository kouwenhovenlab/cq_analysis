import numpy as np
import matplotlib.pyplot as plt

from cq_analysis.fit.models import resonators
from cq_analysis import fit

class ResonanceData:
    @classmethod
    def from_db(cls, meshgrid, freq_name, magnitude_name, phase_name):
        freq_data = meshgrid[freq_name]['values']
        mag_data = meshgrid[magnitude_name]['values']
        phase_data = meshgrid[phase_name]['values']
        cdata = 10**(mag_data/20)*np.exp(1j*phase_data)
        return ResonanceData(freq_data, np.real(cdata), np.imag(cdata))

    def __init__(self, freqdata, Idata, Qdata):
        self.freqdata = freqdata
        self.Idata = Idata
        self.Qdata = Qdata

        self.cdata = self.Idata + 1j * self.Qdata
        self.ampdata = np.sqrt(self.Idata ** 2 + self.Qdata ** 2)
        self.phidata = np.arctan2(self.Qdata, self.Idata)

    def plot(self):
        fig, axs = plt.subplots(1, 3)
        axs[0].plot(self.freqdata, self.ampdata)
        axs[1].plot(self.freqdata, self.phidata)
        axs[2].plot(self.Idata, self.Qdata)
        plt.show()

def fit_resonance(resonancedata: ResonanceData, plot=False):
    model = resonators.HangerModel_kappa()

    fitresult, _ = fit.fit1d(model, resonancedata.cdata.ravel(), resonancedata.freqdata,
                             guess_kws=dict(fs=resonancedata.freqdata), plot=plot, plot_guess=False)

    return fitresult