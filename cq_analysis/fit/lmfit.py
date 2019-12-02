"""Module for convience tools for lmfit"""

import logging
import numpy as np
from matplotlib import pyplot as plt


logger = logging.getLogger(__name__)


def fit(model, data, method='leastsq', print_result=True,
        model_kws={}, init_params={}, guess_kws={}, fit_kws={}):

    try:
        p0 = model.guess(data, **guess_kws)
    except NotImplementedError:
        logger.warning(f"No guess available for '{model}'.")
        p0 = model.make_params()

    for pname, opts in init_params.items():
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
        figsize = (9, 3)
    else:
        complex = False
        naxes = 1
        figsize = (4, 3)

    fig, axes = plt.subplots(1, naxes, figsize=figsize)
    if naxes == 1:
        axes = [axes]

    if complex:
        ys = [np.abs(data), np.angle(data), data.imag]
        xs = [xvals, xvals, data.real]
        fitys = [np.abs(fit_data), np.angle(fit_data), fit_data.imag]
        fitxs = [xvals, xvals, fit_data.real]
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


def array_fit1d(model, data, xvals, axis=-1, verbose=False, callback=None, **kw):
    outshp = list(data.shape)
    del outshp[axis]
    if callback is None:
        callback = lambda :None

    results = {}
    for pn in model.param_names:
        results[pn] = np.zeros(outshp) * np.nan
        results[pn + '_std'] = np.zeros(outshp) * np.nan

    if axis != -1:
        order = list(range(len(data.shape)))
        del order[axis]
        order.append(axis)
        data2 = data.transpose(order)
    else:
        data2 = data

    iterator = np.nditer(data2[..., 0], flags=['multi_index'])
    while not iterator.finished:
        idx = iterator.multi_index
        try:
            fitres, _ = fit1d(model, data2[idx], xvals,
                              plot=verbose, print_result=verbose, **kw)
            if fitres.success:
                for pn in model.param_names:
                    results[pn][idx] = fitres.params[pn].value
                    results[pn + '_std'][idx] = fitres.params[pn].stderr
        except:
            logger.warning('could not fit point: ', idx)

        iterator.iternext()
        callback()

    return results

