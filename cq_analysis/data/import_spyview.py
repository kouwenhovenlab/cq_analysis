""" Methods for reading spyview data"""

from typing import Union, Sequence, Any, Dict, Tuple, List
import numpy as np


def read_spyview_meta(path: str) -> Dict[str, Any]:
    """read the spyview meta file from path.

    :return: Dictionary containing the names of the columns in order,
             the shape of the data, and the assumed axes limits
             (from the x/y/z info)
    """
    lines = []
    with open(path) as f:
        for line in f:
            l = line.strip('\n')
            if l == '':
                continue
            lines.append(l)

    meta = {}
    # gather the dimension names
    meta['column names'] = [
        lines[3], lines[7], lines[11]
    ]
    i = 13
    while i < len(lines):
        meta['column names'].append(lines[i])
        i += 2

    # shape of the dataset
    meta['shape'] = (int(lines[0]), int(lines[4]), int(lines[8]))

    # axes values
    meta['axis limits'] = {
        lines[3] : (float(lines[1]), float(lines[2]), int(lines[0])),
        lines[7] : (float(lines[5]), float(lines[6]), int(lines[4])),
        lines[11] : (float(lines[9]), float(lines[10]), int(lines[8])),
    }
    return meta


def read_spyview_data(path: str, usecols: Union[Sequence[int], int] = None,
                      fill: Any = np.nan, verbose: bool = True,
                      **kw) -> Tuple[List[str], Dict[str, Tuple[float, float, int]], Dict[str, np.ndarray]]:
    """read in spyview (.dat + .meta.txt) format data

    All kwargs are forwarded to np.loadtxt.

    :param usecols: see np.loadtxt -- allows selective extraction
    :param fill: fill value for missing data
    :param verbose: if True, print some info about the loaded data.
    :return: list of x/y/z axes (x is the slowest axis!),
             dict containing the axes limits and #points from the meta info,
             and a dictionary containing all data, reshaped according to the
             info in the meta data.
    """
    datpath = path + '.dat'
    metapath = path + '.meta.txt'
    meta = read_spyview_meta(metapath)
    nrecords = np.prod(meta['shape'])

    if usecols is None:
        usecols = tuple(range(len(meta['column names'])))
    elif isinstance(usecols, int):
        usecols = tuple(usecols)

    data = np.ones((nrecords, len(usecols))) * fill
    _data = np.loadtxt(datpath, comments=['#', '\n'], usecols=usecols, **kw)
    data[:_data.size, :] = _data

    retdata = {}
    for idx, col in enumerate(usecols):
        retdata[meta['column names'][col]] = data[:, idx].reshape(meta['shape'], order='F').transpose((2,1,0))
    axnames = tuple(meta['column names'][:3][::-1])
    axlimits = meta['axis limits']

    axinfo = ""
    for iax, ax in enumerate(axnames):
        axinfo += f"\n * {ax}: {axlimits[ax]}"

    if verbose:
        print('Axes: ', axinfo)
        print('Data columns: ', [meta['column names'][c] for c in usecols])

    return axnames, axlimits, retdata
