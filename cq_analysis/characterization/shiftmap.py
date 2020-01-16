import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import qcodes as qc

from plottr.data.datadict import datadict_to_meshgrid
from plottr.data.qcodes_dataset import ds_to_datadict

from cq_analysis.fit.models import dispersive
from cq_analysis.fit.models import resonators
from cq_analysis import fit
from cq_analysis.characterization import resonance


class ShiftMap():
    """Class for fitting data-cubes of frequency scans along with 2 other axes."""
    @classmethod
    def from_dataset(cls, ds, gate1_name, gate2_name, freq_name, magnitude_name, phase_name, transpose_gate2=False):
        meshgrid = datadict_to_meshgrid(ds_to_datadict(ds))
        return cls.from_meshgrid(meshgrid, gate1_name, gate2_name, freq_name, magnitude_name, phase_name, 
                                 transpose_gate2=transpose_gate2)

    @classmethod
    def from_meshgrid(cls, meshgrid, gate1_name, gate2_name, freq_name, magnitude_name, phase_name, transpose_gate2=False):
        for gatesquare in meshgrid[gate1_name]['values']:
            if not np.all(gatesquare == gatesquare[0]):
                raise Exception('Gate 1 values are not cubic!')
        for gatesquare in meshgrid[gate2_name]['values']:
            if not np.all(gatesquare.T == gatesquare.T[0]):
                raise Exception('Gate 2 values are not cubic!')
        for freqcolumn in meshgrid[freq_name]['values'].transpose():
            if not np.all(freqcolumn == freqcolumn[0]):
                raise Exception('Frequency values are not cubic!')

        gate1_data = meshgrid[gate1_name]['values'].transpose()[0,0,:]
        gate2_data = meshgrid[gate2_name]['values'].transpose()[0,:,0]
        freq_data = meshgrid[freq_name]['values'][0][0]
        mag_data = meshgrid[magnitude_name]['values']
        phase_data = meshgrid[phase_name]['values']
        
        data_shape = (len(gate1_data), len(gate2_data), len(freq_data))
        if mag_data.shape != data_shape:
            raise Exception('magnitude data_shape wrong'+str(mag_data.shape)+" != "+str(data_shape))
        if phase_data.shape != data_shape:
            raise Exception('Phase data_shape wrong'+str(mag_data.shape)+" != "+str(data_shape))
        cdata = 10**(mag_data/20)*np.exp(1j*phase_data)
        return ShiftMap(gate1_data, gate2_data, freq_data, np.real(cdata), np.imag(cdata), 
                        transpose_gate2=transpose_gate2)

    def __init__(self, gate1data, gate2data, freqdata, Idata, Qdata, transpose_gate2=False):
        self.gate1data = gate1data
        self.gate2data = gate2data
        self.freqdata = freqdata.T if transpose_gate2 else freqdata
        self.Idata = Idata.T if transpose_gate2 else Idata
        self.Qdata = Qdata.T if transpose_gate2 else Qdata

        self.cdata = self.Idata + 1j*self.Qdata
        self.ampdata = np.sqrt(self.Idata**2 + self.Qdata**2)
        self.phidata = np.arctan2(self.Qdata, self.Idata)
        
    def make_resonancedata(self, i, j):
        """Create a ResonanceData object for a specific gate-space point.

        Parameters:
        -----------
        i (int): Index of gate voltage point on the gate1data axis.
        j (int): Index of gate voltage point on the gate2data axis.

        Returns:
        --------
        ResonanceData: ResonanceData object corresponding to gate-space
            point (i, j).
        """
        return resonance.ResonanceData(self.freqdata, self.Idata[i][j], self.Qdata[i][j])
    
    def plot_CSD(self, freqindex):
        """Plot charge stability diagrams of various resonator quantities.

        These quantities include the I and Q data, the magnitude of the resonator
        response, and the phase.

        Parameters:
        -----------
        freqindex (int): index of the frequency point to determine the slice of the data
        cube to display. freqindex must be < len(freqdata).
        """
        fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
        im = axs[0, 0].pcolormesh(self.gate1data, self.gate2data, self.Idata.T[freqindex,...])
        cb = fig.colorbar(im, ax=axs[0, 0])
        cb.set_label('I')
        im = axs[0, 1].pcolormesh(self.gate1data, self.gate2data, self.Qdata.T[freqindex,...])
        cb = fig.colorbar(im, ax=axs[0, 1])
        cb.set_label('Q')
        im = axs[1, 0].pcolormesh(self.gate1data, self.gate2data, self.ampdata.T[freqindex,...])
        cb = fig.colorbar(im, ax=axs[1, 0])
        cb.set_label('Magnitude')
        im = axs[1, 1].pcolormesh(self.gate1data, self.gate2data, self.phidata.T[freqindex,...])
        cb = fig.colorbar(im, ax=axs[1, 1])
        cb.set_label('Phase')
        axs[1,0].set_xlabel("Gate 1 Voltage (V)")
        axs[1,1].set_xlabel("Gate 1 Voltage (V)")
        axs[0,0].set_ylabel("Gate 2 Voltage (V)")
        axs[1,0].set_ylabel("Gate 2 Voltage (V)")
        plt.show()

    def get_init_params(self, i, j):
        """Alias for make_resonancedata().get_init_params()."""
        return self.make_resonancedata(i, j).get_init_params()
        
    def fit_resonances(self, plot=True, init_params=None):
        """Fit (and plot) resonator response at each point in gate-space.

        Fits all resonators using resonators.HangerModel_kappa, which guesses
        any fit parameters that are not provided in init_params.

        Keyword Arguments:
        ------------------
        plot (bool): Whether or not to plot fitted resonator frequencies as
            a charge stability diagram.
        init_params (dict): Dictionary of all custom parameters of the fit 
            essentially in the form of a lmfit.Parameters object if converted 
            to a dict -- each key of the dict is a parameter name, whose value
            is a dict with keys corresponding to 'max', 'min' values of the 
            parameter, if it will 'vary' or not, and the 'value' of the parameter.
        
        Returns:
        --------
        numpy.array[lmfit.model.ModelResult]: Array with the same shape as
            (gate1data, gate2data) consisting of frequency fits at each
            point.
        """
        orig_shape = self.cdata.shape # For unflattening fit result
        model = resonators.HangerModel_kappa()
        init_params = {} if init_params is None else init_params
        fitresult = fit.array_fit1d(
            model, 
            # Flatten gate axes to make effectively 1D
            self.cdata.reshape((orig_shape[0]*orig_shape[1], 
                                orig_shape[2])), 
            self.freqdata, 
            guess_kws=dict(fs=self.freqdata.flatten()),
            init_params=init_params
        )
        # Finally, unflatten the fitresult to restore the disctinction between
        # gate axes
        reshaped_fitresult = fitresult
        for key in fitresult.keys():
            reshaped_fitresult[key] = np.array(fitresult[key]).reshape(orig_shape[0:2]).T

        if plot:
            plt.pcolormesh(self.gate1data, self.gate2data, reshaped_fitresult['f0'] / 1e9)
            plt.xlabel('Gate Voltage 1 (V)')
            plt.ylabel('Gate Voltage 2 (V)')
            cb = plt.colorbar()
            cb.set_label('Fitted Frequency f0 (GHz)')
        return reshaped_fitresult
