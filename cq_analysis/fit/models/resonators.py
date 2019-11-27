"""lmfit models for resonator responses"""

import numpy as np
import lmfit


twopi = np.pi * 2

class HangerModel(lmfit.Model):
    """
    An implementation of the hanger model from
    Khalil et al. Journal of Applied Physics 111, 054510 (2012)
    Eq. (12)
    Parameters:
    - f0 [Hz] - resonator frequency
    - Q_i [1] - internal quality factor
    - Q_e_mag [1] - magnitude of the external quality factor
    - Q_e_phase [rad] - phase of the external quality factor, indicating
        mismatch between input and output impedance
    - amp_slope [1] - additional slope of the transmission amplitude
    - amp_offset [measurement device units] - amplitude
        corresponding to the full transmission
    - phase_winding [rad/Hz]
    - phase_offset [rad]
    """

    @staticmethod
    def s21_khalil(fs, f0, Q_i, Q_e_mag, Q_e_phase):
        Q_e = Q_e_mag * np.exp(-1j * Q_e_phase)
        Q_c = 1. / ((1. / Q_e).real)
        Q_tot = 1. / (1. / Q_i + 1. / Q_c)
        return 1. - (Q_tot / Q_e_mag * np.exp(1j * Q_e_phase)) / (
                    1 + 2j * Q_tot * (fs - f0) / f0)

    @staticmethod
    def func(fs, f0, Q_i, Q_e_mag, Q_e_phase, amp_slope, amp_offset,
             phase_winding, phase_offset):
        sig = HangerModel.s21_khalil(fs, f0, Q_i, Q_e_mag, Q_e_phase)
        sig *= amp_offset * (1. + amp_slope * (fs - f0) / f0)
        sig *= np.exp(1j * (phase_offset + phase_winding * (fs - fs[0])))
        return sig

    def __init__(self):
        super().__init__(self.func)

    @staticmethod
    def guess_logic(data, fs, edge_fracs=(20, 20)):
        e0, e1 = edge_fracs

        mag, phi = np.abs(data), np.angle(data)
        npts = fs.size

        # some basic metrics of the data
        avg0 = data[:npts // e0].mean()
        avg1 = data[-npts // e1:].mean()
        mag_std = np.abs(data[:npts // e0]).std()
        mag_avg = (np.abs(avg0) + np.abs(avg1)) / 2.
        mag_min = mag.min() / mag_avg

        # guess for resonance is simply the minimal magnitude
        f0 = fs[np.argmin(np.abs(mag - mag_avg))]

        # guess for Q: guess line width from fraction of points that lie a
        # certain distance from the mean. then compute Qs from that.
        noutliers = np.where((mag < (mag_avg - 3. * mag_std)) | (
            mag > (mag_avg + 3. * mag_std)))[0].size
        fracoutliers = 0.5 * noutliers / fs.size
        fwhm = np.abs(fs[-1] - fs[0]) * fracoutliers
        Ql = f0 / fwhm
        Qi = Ql / mag_min
        Qc = Ql / np.abs(1. - mag_min)

        # phase winding:
        # look at the average derivative at the edges, and ignore obvious jumps
        phase_diffs = np.append(np.diff(phi)[:npts // e0],
                                np.diff(phi)[-npts // e1])
        phase_winding = phase_diffs[np.abs(phase_diffs) < np.pi / 8.].mean()

        return dict(
            f0=fs[np.argmin(mag)],
            Q_i=Qi,
            Q_e_mag=Qc,
            Q_e_phase=0,
            amp_slope=0,
            amp_offset=mag_avg,
            phase_winding=phase_winding / (fs[1] - fs[0]),
            phase_offset=phi[0],
        )

    def guess(self, data, fs, edge_fracs=(20, 20)):
        e0, e1 = edge_fracs

        mag, phi = np.abs(data), np.angle(data)
        npts = fs.size

        # some basic metrics of the data
        avg0 = data[:npts // e0].mean()
        avg1 = data[-npts // e1:].mean()
        mag_std = np.abs(data[:npts // e0]).std()
        mag_avg = (np.abs(avg0) + np.abs(avg1)) / 2.
        mag_min = mag.min() / mag_avg

        # guess for resonance is simply the minimal magnitude
        f0 = fs[np.argmin(np.abs(mag - mag_avg))]

        # guess for Q: guess line width from fraction of points that lie a
        # certain distance from the mean. then compute Qs from that.
        noutliers = np.where((mag < (mag_avg - 3. * mag_std)) | (
                    mag > (mag_avg + 3. * mag_std)))[0].size
        fracoutliers = 0.5 * noutliers / fs.size
        fwhm = np.abs(fs[-1] - fs[0]) * fracoutliers
        Ql = f0 / fwhm
        Qi = Ql / mag_min
        Qc = Ql / np.abs(1. - mag_min)

        # phase winding:
        # look at the average derivative at the edges, and ignore obvious jumps
        phase_diffs = np.append(np.diff(phi)[:npts // e0],
                                np.diff(phi)[-npts // e1])
        phase_winding = phase_diffs[np.abs(phase_diffs) < np.pi / 8.].mean()

        p0 = self.make_params(
            f0=fs[np.argmin(mag)],
            Q_i=Qi,
            Q_e_mag=Qc,
            Q_e_phase=0,
            amp_slope=0,
            amp_offset=mag_avg,
            phase_winding=phase_winding / (fs[1] - fs[0]),
            phase_offset=phi[0],
        )
        return p0


class HangerModel_kappa(lmfit.Model):

    @staticmethod
    def func(fs, f0, k_i, k_e_mag, k_e_phase, amp_slope, amp_offset,
             phase_winding, phase_offset):
        Q_i = f0/k_i
        Q_e_mag = f0/k_e_mag
        Q_e_phase = -k_e_phase
        return HangerModel.func(fs, f0, Q_i, Q_e_mag, Q_e_phase, amp_slope, amp_offset, phase_winding, phase_offset)

    def __init__(self):
        super().__init__(self.func)

    def guess(self, data, fs, edge_fracs=(20, 20)):
        p0_q = HangerModel.guess_logic(data, fs, edge_fracs=edge_fracs)

        p0 = self.make_params(
            f0=p0_q['f0'],
            k_i=p0_q['f0']/p0_q['Q_i'],
            k_e_mag=p0_q['f0']/p0_q['Q_e_mag'],
            k_e_phase=-p0_q['Q_e_phase'],
            amp_slope=p0_q['amp_slope'],
            amp_offset=p0_q['amp_offset'],
            phase_winding=p0_q['phase_winding'],
            phase_offset=p0_q['phase_offset'],
        )
        p0['k_e_mag'].min=0
        p0['k_i'].min=0
        return p0



class HangerReflectionModel(lmfit.Model):
    """
    Hanger model from
    Khalil et al. Journal of Applied Physics 111, 054510 (2012)
    modified to measurements in reflection.
    Eq. (12) has a small modification:
    (1+e)(1-BLA) -> (1+e)(1-2*BLA)
    Parameters:
    - f0 [Hz] - resonator frequency
    - Q_i [1] - internal quality factor
    - Q_e_mag [1] - magnitude of the external quality factor
    - Q_e_phase [rad] - phase of the external quality factor, indicating
        mismatch between input and output impedance
    - amp_slope [1] - additional slope of the transmission amplitude
    - amp_offset [measurement device units] - amplitude
        corresponding to the full transmission
    - phase_winding [rad/Hz]
    - phase_offset [rad]
    """

    @staticmethod
    def s21_modified_khalil(fs, f0, Q_i, Q_e_mag, Q_e_phase):
        Q_e = Q_e_mag * np.exp(-1j * Q_e_phase)
        Q_c = 1. / ((1. / Q_e).real)
        Q_tot = 1. / (1. / Q_i + 1. / Q_c)
        return 1. - 2* (Q_tot / Q_e_mag * np.exp(1j * Q_e_phase)) / (
                    1 + 2j * Q_tot * (fs - f0) / f0)

    @staticmethod
    def func(fs, f0, Q_i, Q_e_mag, Q_e_phase, amp_slope, amp_offset,
             phase_winding, phase_offset):
        sig = HangerReflectionModel.s21_modified_khalil(fs, f0, Q_i, Q_e_mag, Q_e_phase)
        sig *= amp_offset * (1. + amp_slope * (fs - f0) / f0)
        sig *= np.exp(1j * (phase_offset + phase_winding * (fs - fs[0])))
        return sig

    def __init__(self):
        super().__init__(self.func)

    @staticmethod
    def guess_logic(data, fs, edge_fracs=(20, 20)):
        e0, e1 = edge_fracs

        mag, phi = np.abs(data), np.angle(data)
        npts = fs.size

        # some basic metrics of the data
        avg0 = data[:npts // e0].mean()
        avg1 = data[-npts // e1:].mean()
        mag_std = np.abs(data[:npts // e0]).std()
        mag_avg = (np.abs(avg0) + np.abs(avg1)) / 2.
        mag_min = mag.min() / mag_avg

        # guess for resonance is simply the minimal magnitude
        f0 = fs[np.argmin(np.abs(mag - mag_avg))]

        # guess for Q: guess line width from fraction of points that lie a
        # certain distance from the mean. then compute Qs from that.
        noutliers = np.where((mag < (mag_avg - 3. * mag_std)) | (
            mag > (mag_avg + 3. * mag_std)))[0].size
        fracoutliers = 0.5 * noutliers / fs.size
        fwhm = np.abs(fs[-1] - fs[0]) * fracoutliers
        Ql = f0 / fwhm
        Qi = Ql / mag_min
        Qc = Ql / np.abs(1. - mag_min)

        # phase winding:
        # look at the average derivative at the edges, and ignore obvious jumps
        phase_diffs = np.append(np.diff(phi)[:npts // e0],
                                np.diff(phi)[-npts // e1])
        phase_winding = phase_diffs[np.abs(phase_diffs) < np.pi / 8.].mean()

        return dict(
            f0=fs[np.argmin(mag)],
            Q_i=Qi,
            Q_e_mag=Qc,
            Q_e_phase=0,
            amp_slope=0,
            amp_offset=mag_avg,
            phase_winding=phase_winding / (fs[1] - fs[0]),
            phase_offset=phi[0],
        )

    def guess(self, data, fs, edge_fracs=(20, 20)):
        e0, e1 = edge_fracs

        mag, phi = np.abs(data), np.angle(data)
        npts = fs.size

        # some basic metrics of the data
        avg0 = data[:npts // e0].mean()
        avg1 = data[-npts // e1:].mean()
        mag_std = np.abs(data[:npts // e0]).std()
        mag_avg = (np.abs(avg0) + np.abs(avg1)) / 2.
        mag_min = mag.min() / mag_avg

        # guess for resonance is simply the minimal magnitude
        f0 = fs[np.argmin(np.abs(mag - mag_avg))]

        # guess for Q: guess line width from fraction of points that lie a
        # certain distance from the mean. then compute Qs from that.
        noutliers = np.where((mag < (mag_avg - 3. * mag_std)) | (
                    mag > (mag_avg + 3. * mag_std)))[0].size
        fracoutliers = 0.5 * noutliers / fs.size
        fwhm = np.abs(fs[-1] - fs[0]) * fracoutliers
        Ql = f0 / fwhm
        Qi = Ql / mag_min
        Qc = Ql / np.abs(1. - mag_min)

        # phase winding:
        # look at the average derivative at the edges, and ignore obvious jumps
        phase_diffs = np.append(np.diff(phi)[:npts // e0],
                                np.diff(phi)[-npts // e1])
        phase_winding = phase_diffs[np.abs(phase_diffs) < np.pi / 8.].mean()

        p0 = self.make_params(
            f0=fs[np.argmin(mag)],
            Q_i=Qi,
            Q_e_mag=Qc,
            Q_e_phase=0,
            amp_slope=0,
            amp_offset=mag_avg,
            phase_winding=phase_winding / (fs[1] - fs[0]),
            phase_offset=phi[0],
        )
        return p0