"""Module for convience tools for lmfit"""

import logging
import numpy as np
from matplotlib import pyplot as plt


logger = logging.getLogger(__name__)


def fit(model, data, method='leastsq', print_result=True,
        model_kws={}, parameter_init={}, guess_kws={}, fit_kws={}):

    try:
        p0 = model.guess(data, **guess_kws)
    except NotImplementedError:
        logger.warning(f"No guess available for '{model}'.")
        p0 = model.make_params()

    for pname, opts in parameter_init.items():
        p0[pname].value = opts.get('value', p0[pname].value)
        p0[pname].vary = opts.get('vary', True)
        p0[pname].max = opts.get('max', np.inf)
        p0[pname].min = opts.get('min', -np.inf)

    fit_result = model.fit(data, p0, method=method, fit_kws=fit_kws,
                           **model_kws)

    if print_result:
        print(fit_result.fit_report())

    return fit_result


def plot_fit1d(model, data, xvals, fit_result, plot_guess=True):
    model_kws = {model.independent_vars[0]: xvals}
    fit_data = model.eval(fit_result.params, **model_kws)

    if np.iscomplexobj(data):
        complex = True
        naxes = 3
    else:
        complex = False
        naxes = 1

    fig, axes = plt.subplots(1, naxes, figsize=(naxes*4 - 0.5, 3))
    if naxes == 1:
        axes = [axes]

    if complex:
        ys = [np.abs(data), np.angle(data), data.imag]
        xs = [xvals, xvals, data.real]
        fitys = [np.abs(fit_data), np.angle(fit_data), fit_data.imag]
        fitxs = [xvals, xvals, fit_data.imag]
    else:
        ys = [data]
        xs = [xvals]
        fitys = [fit_data]
        fitxs = [xvals]

    for i, ax in enumerate(axes):
        ax.plot(xs[i], ys[i], 'o', ms=5, mfc='w', mew=1.5)
        ax.plot(fitxs[i], fitys[i], '-', lw=2)


    fig.tight_layout()

    return fig, axes


def fit1d(model, data, xvals, plot=True, plot_guess=True, **kw):
    indep = model.independent_vars
    if len(indep) > 1:
        raise ValueError('More than 1 independent for the model.')

    model_kws = {indep[0]: xvals}
    fit_result = fit(model, data, model_kws=model_kws, **kw)

    if plot:
        fig, axes = plot_fit1d(model, data, xvals, fit_result,
                               plot_guess=plot_guess)

    else:
        fig, axes = None, None

    return fit_result, (fig, axes)







