import numpy as np
import scipy as sp

import scipy.constants as constants
import cq_analysis.fit.models.resonators as resonators

import matplotlib.pyplot as plt

from scipy.interpolate import interp2d, LSQBivariateSpline, SmoothBivariateSpline

from copy import deepcopy


def generate_params(fr, compl, fcav=111.4e+06, phi=3.58):
    reflection_model = resonators.ResonatorReflectionWithAmp()
    pars = reflection_model.make_params()

    pars['f0'].value = np.mean(fr)
    pars['Qi'].value = 2000
    pars['Qi'].min = 3
    pars['Qe'].value = 300
    pars['Qe'].min = 3

    pars['a'].value = np.sqrt(1-0.398**2) # reflection 8 dB
    # pars['a'].value = np.sqrt(1-0.630**2) # reflection 6 dB
    pars['a'].vary = False
    pars['g'].value = 0.178 # couplinf coef 15 dB
    pars['g'].vary = False

    # fcav, phi = 111.4e+06, 3.58
    # fcav, phi = 222.8e+06, 3.58
    pars['fcav'].value = fcav
    pars['fcav'].vary = False
    pars['phi'].value = phi
    # pars['phi'].vary = False

    pars['amp_slope'].value = 0
    pars['amp_offset'].value = np.mean(np.abs(compl))/pars['g'].value
    pars['amp_offset'].min = 0

    phs, wind = reflection_model.guess_phases(compl, fr)
    pars['phase_winding'].value = wind
    pars['phase_offset'].value = 0
    
    return pars

def make_initial_resonator_fit(fr, compl, plot=False, model=None):
    # load data
    #     fr, amp, ph = load_qcodes_data_by_id(N, ['MIDAS_ch1_Amplitude', 'MIDAS_ch1_Phase'])
    #     compl = amp*np.exp(1j*ph)
    
    # perform fit
    if model is None:
        reflection_model = resonators.ResonatorReflectionWithAmp()
        pars = generate_params(fr, compl)
    else:
        reflection_model = model
        pars = reflection_model.guess(compl, fr)
        pars['Qi'].value = 500
        pars['Qi'].min = 5
    init = reflection_model.eval(pars, fs=fr)
    result = reflection_model.fit(compl, pars, fs=fr)

    # plot
    if plot:
        plt.figure(figsize=(7,3))

        plt.subplot(1,2,1)
        plt.plot(fr, np.abs(compl), '.')
        # plt.plot(fr, np.abs(init))
        plt.plot(fr, np.abs(result.best_fit))

        plt.subplot(1,2,2)
        plt.plot(np.real(compl), np.imag(compl), '.')
        # plt.plot(np.real(init), np.imag(init))
        plt.plot(np.real(result.best_fit), np.imag(result.best_fit))

        plt.tight_layout()
        
    return result

def fit_line_by_line_with_fixed_params(fr, compl, params):
    # fix all params except for f0 and Qi
    params['Qe'].vary = False
    params['a'].vary = False
    params['g'].vary = False
    params['fcav'].vary = False
    params['phi'].vary = False
    params['amp_slope'].vary = False
    params['amp_offset'].vary = False
    params['phase_winding'].vary = False
    params['phase_offset'].vary = False
    
    # initialize dictionary with the results
    result_dict = {k: [] for k in params}
    
    reflection_model = resonators.ResonatorReflectionWithAmp()
    for i,c in enumerate(compl):
        if i%10 == 9:
            print('.',end='')
        
        # perform a fit
        result = reflection_model.fit(c, params, fs=fr)
        
        # send the fit resutlt to result dictionary
        for k, v in result_dict.items():
            v.append(result.params[k].value)
    
    
    # convert entries in dictionary to ndarray
    for k, v in result_dict.items():
            v = np.array(v)
            result_dict[k] = v

    return result_dict

def make_IQ_to_f0_Qi_map(params, f_meas, f_min=-0.6e6, f_max=0.6e6, Q_min=0.2, Q_max=5,
                return_grid=False, f0_size=161, Q_size=81, model=None):
    if model is None:
        reflection_model = resonators.ResonatorReflectionWithAmp() 
    else:
        reflection_model = model
    
    p = deepcopy(params)

    f0, Qi = np.meshgrid(np.linspace(params['f0'].value+f_min,params['f0'].value+f_max,f0_size), np.geomspace(params['Qi'].value*Q_min, params['Qi'].value*Q_max, Q_size))
    f0 = f0.flatten()
    Qi = Qi.flatten()
    
    compl = [reflection_model.eval(p, fs=np.array([f_meas]), f0=f, Qi=Q)[0] for f,Q in zip(f0,Qi)]
    I, Q = np.real(compl), np.imag(compl)

    Iknots = np.linspace(np.min(I), np.max(I),81)
    Qknots = np.linspace(np.min(Q), np.max(Q),81)
    f0_estimator = LSQBivariateSpline(I, Q, f0, Iknots, Qknots, kx=1, ky=1)
    Qi_estimator = LSQBivariateSpline(I, Q, Qi, Iknots, Qknots, kx=1, ky=1)
    
    if return_grid:
        return f0_estimator, Qi_estimator, I, Q, f0, Qi, Iknots, Qknots  
    return f0_estimator, Qi_estimator

def calc_f0_Qi_maps(f0_estimator, Qi_estimator, compl):
    f0s, Qis = [], []
    for comp in compl:
        f, Q = [], []
        for c in comp:
            f.append(f0_estimator(np.real(c),np.imag(c))[0,0])
            Q.append(Qi_estimator(np.real(c),np.imag(c))[0,0])
        f0s.append(f)
        Qis.append(Q)
    f0s = np.array(f0s)
    Qis = np.array(Qis)

    return f0s, Qis