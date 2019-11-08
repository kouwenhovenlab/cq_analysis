import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from plottr.data.datadict import datadict_to_meshgrid
from plottr.data.qcodes_dataset import ds_to_datadict

from cq_analysis.fit.models import dispersive
from cq_analysis import fit


class DispersiveShiftData:
    @classmethod
    def from_dataset(cls, ds, gate_name, freq_name, magnitude_name, phase_name):
        meshgrid = datadict_to_meshgrid(ds_to_datadict(ds))
        return cls.from_meshgrid(meshgrid, gate_name, freq_name, magnitude_name, phase_name)

    @classmethod
    def from_meshgrid(cls, meshgrid, gate_name, freq_name, magnitude_name, phase_name):
        for gateline in meshgrid[gate_name]['values']:
            if not np.all(gateline == gateline[0]):
                raise Exception('gate values are not cubic')
        for freqcolumn in meshgrid[freq_name]['values'].transpose():
            if not np.all(freqcolumn == freqcolumn[0]):
                raise Exception('frequency values are not cubic')

        gate_data = meshgrid[gate_name]['values'].transpose()[0]
        freq_data = meshgrid[freq_name]['values'][0]
        mag_data = meshgrid[magnitude_name]['values']
        phase_data = meshgrid[phase_name]['values']
        if mag_data.shape != (len(gate_data), len(freq_data)):
            raise Exception('magnitude data_shape wrong'+str(mag_data.shape)+" != "+str((len(gate_data), len(freq_data))))
        if phase_data.shape != (len(gate_data), len(freq_data)):
            raise Exception('magnitude data_shape wrong')
        cdata = 10**(mag_data/20)*np.exp(1j*phase_data)
        return DispersiveShiftData(gate_data, freq_data, np.real(cdata), np.imag(cdata))

    def __init__(self, gatedata, freqdata, Idata, Qdata):
        self.gatedata = gatedata
        self.freqdata = freqdata
        self.Idata = Idata
        self.Qdata = Qdata

        self.cdata = self.Idata + 1j*self.Qdata
        self.ampdata = np.sqrt(self.Idata**2 + self.Qdata**2)
        self.phidata = np.arctan2(self.Qdata, self.Idata)

        # data[gate, frequency]

    def plot(self):
        fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
        axs[0, 0].pcolormesh(self.gatedata, self.freqdata, self.Idata.transpose())
        axs[0, 1].pcolormesh(self.gatedata, self.freqdata, self.Qdata.transpose())
        axs[1, 0].pcolormesh(self.gatedata, self.freqdata, self.ampdata.transpose())
        axs[1, 1].pcolormesh(self.gatedata, self.freqdata, self.phidata.transpose())
        plt.show()


def get_shift(dispersivedata: DispersiveShiftData):
    fr = dispersivedata.freqdata[np.argmin(dispersivedata.ampdata, axis=1)]
    return np.max(fr)-np.min(fr)

def make_cutout(dispersivedata: DispersiveShiftData, plot=False):
    # do some magic
    #estimate resonance frequencies
    fr = dispersivedata.freqdata[np.argmin(dispersivedata.ampdata, axis=1)]
    # get charge degeneracies from this
    def f(gate, fc, ampl, phase, freq):
        return fc + np.abs(ampl) * np.sin(phase + 2 * np.pi * freq * gate)

    ig = [np.average(fr),
          (np.max(fr) - np.min(fr)) / 2,
          0,
          2 / (np.max(dispersivedata.gatedata) - np.min(dispersivedata.gatedata))]
    popt, pcov = curve_fit(f, dispersivedata.gatedata, fr, p0=ig)

    # first find all charge degeneragy points
    degenpoints = []
    # these points are when sin is at its 1.5 np.pi + n*2*np.pi point
    g0 = (-2 * popt[2] + 3 * np.pi) / (4 * popt[3] * np.pi)
    gate_delta = 1 / popt[3]
    print(g0, gate_delta)

    g = g0
    while g > np.min(dispersivedata.gatedata):
        g -= gate_delta
    g += gate_delta
    while g < np.max(dispersivedata.gatedata):
        degenpoints.append(g)
        g += gate_delta
    best_window = 0
    best_index = 0
    for i, gate in enumerate(degenpoints):
        # calculate how big the datarange is
        window = min(gate - np.min(dispersivedata.gatedata), gate_delta / 2) + min(np.max(dispersivedata.gatedata) - gate, gate_delta / 2)
        if window > best_window:
            best_window = window
            best_index = i

    # use this info to find the window
    window = [max(degenpoints[best_index] - gate_delta / 2, np.min(dispersivedata.gatedata)),
              min(degenpoints[best_index] + gate_delta / 2, np.max(dispersivedata.gatedata))]
    # and find the index points of these points
    window_index = [np.argmin(np.abs(dispersivedata.gatedata - window[0])), np.argmin(np.abs(dispersivedata.gatedata - window[1]))]
    if plot:
        plt.figure()
        plt.plot(dispersivedata.gatedata, fr)
        plt.plot(dispersivedata.gatedata, f(dispersivedata.gatedata, *popt))
        plt.axvline(x=dispersivedata.gatedata[window_index[0]])
        plt.axvline(x=dispersivedata.gatedata[window_index[1]])
        plt.show()
    gatewindowslicer = slice(window_index[0], window_index[1])
    return DispersiveShiftData(dispersivedata.gatedata[gatewindowslicer],
                               dispersivedata.freqdata,
                               dispersivedata.Idata[gatewindowslicer, :],
                               dispersivedata.Qdata[gatewindowslicer, :])


def fit_dispersiveshift(dispersivedata: DispersiveShiftData, plot=False):
    model = dispersive.DispersiveHangerModel()

    fitresult, _ = fit.fit1d(model, dispersivedata.cdata.ravel(), np.array([dispersivedata.gatedata[:, None], dispersivedata.freqdata[None, :]]),
                             guess_kws=dict(gatedata=dispersivedata.gatedata, freqdata=dispersivedata.freqdata), plot=False, plot_guess=False)

    if plot:
        shape = (len(dispersivedata.gatedata), len(dispersivedata.freqdata))
        fitted = model.func([dispersivedata.gatedata[:, None], dispersivedata.freqdata[None, :]], **fitresult.best_values).reshape(shape)
        fig, axs = plt.subplots(2, 4, sharey=True, sharex=True)
        axs[0, 0].pcolormesh(dispersivedata.gatedata, dispersivedata.freqdata, dispersivedata.Idata.transpose())
        axs[0, 1].pcolormesh(dispersivedata.gatedata, dispersivedata.freqdata, np.real(fitted).transpose())
        axs[0, 2].pcolormesh(dispersivedata.gatedata, dispersivedata.freqdata, dispersivedata.Qdata.transpose())
        axs[0, 3].pcolormesh(dispersivedata.gatedata, dispersivedata.freqdata, np.imag(fitted).transpose())
        axs[1, 0].pcolormesh(dispersivedata.gatedata, dispersivedata.freqdata, dispersivedata.ampdata.transpose())
        axs[1, 1].pcolormesh(dispersivedata.gatedata, dispersivedata.freqdata, np.absolute(fitted).transpose())
        axs[1, 2].pcolormesh(dispersivedata.gatedata, dispersivedata.freqdata, dispersivedata.phidata.transpose())
        axs[1, 3].pcolormesh(dispersivedata.gatedata, dispersivedata.freqdata, np.angle(fitted).transpose())
        plt.show()
    return fitresult



