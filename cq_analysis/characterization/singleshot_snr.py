import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import cmath
from scipy.stats import norm


def fid_from_snr(snr):
    """
    The function that returns fidelity value corresponding to an SNR value.
    :param snr:
    :return fidelity:
    """
    fidelity = 1 - norm.sf(snr)
    return fidelity


def data_to_cdata(I_data, Q_data, I_data_ref=None, Q_data_ref=None):
    """
    This function takes in two arrays of data, I and Q, and combines them into a single complex array cdata
    """

    cdata = I_data + 1j * Q_data
    if I_data_ref is not None:
        cref = (I_data_ref + 1j * Q_data_ref) / np.abs(I_data_ref + 1j * Q_data_ref)
        cdata = cdata / cref
    return cdata


def plothist(cdata, bins=100):
    """
    This function will plot the histogram of the blobs.
    """

    plt.hist2d(cdata.real, cdata.imag, bins=bins)
    plt.show()


def gauss(r, xc, sigma):
    """
    This is a simple, normalised gaussian function we will need for fitting.
    """

    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-np.power(r - xc, 2) / (2 * np.power(sigma, 2)))


def calculate_snr(blob1, blob2, plot=False):
    """
    This function outputs the signal-to-noise ratio of given inputs of the two blobs.
    """

    # Rotate the blobs.
    mean1b = np.average(blob1)
    mean2b = np.average(blob2)
    phase = cmath.phase(mean2b - mean1b)
    blob2rot = blob2 * (cmath.exp(-1j * phase))
    blob1rot = blob1 * (cmath.exp(-1j * phase))

    # Project the rotated blobs and for each compute std. deviation and mean value.
    blob1proj = blob1rot.real
    blob2proj = blob2rot.real

    # Fit the blobs with Gaussian function.
    hist1, xbins1 = np.histogram(blob1proj, bins=100)
    hist2, xbins2 = np.histogram(blob2proj, bins=100)
    x1 = 0.5 * (xbins1[:-1] + xbins1[1:])
    x2 = 0.5 * (xbins2[:-1] + xbins2[1:])

    # Calculate the area. Needed for normalization
    area1 = np.dot((xbins1[1:] - xbins1[:-1]), hist1)
    area2 = np.dot((xbins2[1:] - xbins2[:-1]), hist2)
    if plot:
        plt.figure()
        plothist(np.concatenate((blob1, blob2)), bins=100)
        print('rotation:', str(phase) + ' rad')
        plt.figure()
        plothist(np.concatenate((blob1rot, blob2rot)), bins=100)
    # Obtain ideal parameters through the fitting function - sigma and mu
    popt1, pcov1 = opt.curve_fit(gauss, x1, hist1 / area1, maxfev=14000)
    popt2, pcov2 = opt.curve_fit(gauss, x2, hist2 / area2, maxfev=14000)
    mu_1 = popt1[0]
    mu_2 = popt2[0]
    sigma_1 = popt1[1]
    sigma_2 = popt2[1]

    # Set plot to true to see the plots
    if plot:

        plt.figure()
        plt.plot(x1, gauss(x1, *popt1), 'r-')
        plt.plot(x2, gauss(x2, *popt2), 'b-')
        plt.plot(x1, hist1 / area1, 'r+')
        plt.plot(x2, hist2 / area2, 'b+')
        plt.show()
        print('sigma', sigma_1, sigma_2, abs(sigma_1-sigma_2))
        print('CPH_SNR', abs(mu_1 - mu_2) / np.sqrt(sigma_1**2+sigma_2**2))

    # Calculate SNR using snr = delta/(2*sigma)
    snr = abs(mu_1 - mu_2) / (sigma_1 + sigma_2)
    return snr
