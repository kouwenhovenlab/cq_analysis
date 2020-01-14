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
    
    def fit_resonance(self, plot=False, **kw):
        model = resonators.HangerModel_kappa()

        fitresult, _ = fit.fit1d(model, self.cdata.ravel(), self.freqdata,
                                guess_kws=dict(fs=self.freqdata), plot=plot, plot_guess=False, **kw)

        return fitresult

    def get_init_params(self, fitresult=None):
        """Get value and properties of constant fit parameters.

        Keyword Arguments:
        ------------------
        fitresult (lmfit.model.FitResult): Optionally pass in a
            custom fitting (even corresponding to another dataset)
            for which the constant parameters are to be extracted
            so that a fit calculation need not be repeated.

        Returns:
        --------
        dict: Returns nested dictionary of all the lmfit.Parameter properties
            (eg. {'vary': False}) of the HangerModel_kappa fit parameters 
            which are constant. These parameters include 'k_e_mag', 'k_e_phase',
            'amp_slope', 'amp_offset', 'phase_winding', 'phase_offset'. The
            remaining fit parameters of 'f0' and 'k_i' are not included as these
            are dependent on the measured sample's state.
        """
        if fitresult is None:
            fitresult = self.fit_resonance() # Calculate resonator fit
        init_params = dict(fitresult) # Cast fit result as dictionary
        # Delete varying fit parameters to leave only constant ones
        del init_params['f0'], init_params['k_i']
        # Rewrite init_params as nested dictionary of Parameter properties
        for key, val in init_params.items():
            init_params[key] = {
                'value': val.value,
                'vary': False, # These parameters should be fixed across fits
                'max': val.max,
                'min': val.min
            }
        return init_params