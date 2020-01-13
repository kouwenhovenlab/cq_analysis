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
    @classmethod
    def from_dataset(cls, ds, gate1_name, gate2_name, freq_name, magnitude_name, phase_name):
        meshgrid = datadict_to_meshgrid(ds_to_datadict(ds))
        return cls.from_meshgrid(meshgrid, gate1_name, gate2_name, freq_name, magnitude_name, phase_name)

    @classmethod
    def from_meshgrid(cls, meshgrid, gate1_name, gate2_name, freq_name, magnitude_name, phase_name):
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
        print(gate1_data)
        gate2_data = meshgrid[gate2_name]['values'].transpose()[0,:,0]
        print(gate2_data)
        freq_data = meshgrid[freq_name]['values'][0][0]
        mag_data = meshgrid[magnitude_name]['values']
        phase_data = meshgrid[phase_name]['values']
        
        data_shape = (len(gate1_data), len(gate2_data), len(freq_data))
        if mag_data.shape != data_shape:
            raise Exception('magnitude data_shape wrong'+str(mag_data.shape)+" != "+str(data_shape))
        if phase_data.shape != data_shape:
            raise Exception('Phase data_shape wrong'+str(mag_data.shape)+" != "+str(data_shape))
        cdata = 10**(mag_data/20)*np.exp(1j*phase_data)
        return ShiftMap(gate1_data, gate2_data, freq_data, np.real(cdata), np.imag(cdata))

    def __init__(self, gate1data, gate2data, freqdata, Idata, Qdata):
        self.gate1data = gate1data
        self.gate2data = gate2data
        self.freqdata = freqdata
        self.Idata = Idata
        self.Qdata = Qdata

        self.cdata = self.Idata + 1j*self.Qdata
        self.ampdata = np.sqrt(self.Idata**2 + self.Qdata**2)
        self.phidata = np.arctan2(self.Qdata, self.Idata)
        
    def make_resonancedata(self, i, j):
        return resonance.ResonanceData(self.freqdata, self.Idata[i][j], self.Qdata[i][j])
    
    def plot_CSD(self, freqindex):
        fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
        im = axs[0, 0].pcolormesh(self.gate1data, self.gate2data, self.Idata.transpose()[freqindex,...])
        cb = fig.colorbar(im, ax=axs[0, 0])
        cb.set_label('I')
        im = axs[0, 1].pcolormesh(self.gate1data, self.gate2data, self.Qdata.transpose()[freqindex,...])
        cb = fig.colorbar(im, ax=axs[0, 1])
        cb.set_label('Q')
        im = axs[1, 0].pcolormesh(self.gate1data, self.gate2data, self.ampdata.transpose()[freqindex,...])
        cb = fig.colorbar(im, ax=axs[1, 0])
        cb.set_label('Magnitude')
        im = axs[1, 1].pcolormesh(self.gate1data, self.gate2data, self.phidata.transpose()[freqindex,...])
        cb = fig.colorbar(im, ax=axs[1, 1])
        cb.set_label('Phase')
        axs[1,0].set_xlabel("Gate 1 Voltage (V)")
        axs[1,1].set_xlabel("Gate 1 Voltage (V)")
        axs[0,0].set_ylabel("Gate 2 Voltage (V)")
        axs[1,0].set_ylabel("Gate 2 Voltage (V)")
        plt.show()
        
    def make_fitCSD(self, plot=True, init_params=None):
        original_shape = self.cdata.shape
        model = resonators.HangerModel_kappa()
        init_params = {} if init_params is None else init_params

        fitresult = fit.array_fit1d(model, 
                                    self.cdata.reshape((original_shape[0]*original_shape[1], original_shape[2])), 
                                    self.freqdata, 
                                    guess_kws=dict(fs=self.freqdata.flatten()),
                                    init_params=init_params)
        reshaped_fitresult = fitresult
        for key in fitresult.keys():
            reshaped_fitresult[key] = np.array(fitresult[key]).reshape(original_shape[0:2])

        if plot:
            plt.pcolormesh(self.gate1data, self.gate2data, reshaped_fitresult['f0'] / 1e9)
            plt.xlabel('Gate Voltage 1 (V)')
            plt.ylabel('Gate Voltage 2 (V)')
            cb = plt.colorbar()
            cb.set_label('Fitted Frequency f0 (GHz)')
        return reshaped_fitresult
