import qcodes as qc
import pandas as pd
import numpy as np

def load_qcodes_data_by_id(id, variable):
    '''
    Minimal loader of data from Qcodes database. Loads
    a dataset for given id and a single name of 
    measured quantity.

    Loads ONLY data arrays without any metadata.

    Supports 1D sweeps and 2D sweeps on a grid.
    In 1D case returns 1-dimensional array with X values
    and 1-dimensional array with Y values.
    In 2D case returns 1-dimensional arrays of X and Y values
    and 2-dimensional array of Z values.

    If selected measured quantity does not exist prints out the list
    of all possibilities.
    
    Example:
    > gate_voltage, conductance = load_qcodes_data(123, 'Lockin_conductance')
    > plt.plot(gate_voltage, conductance)

    > gate_1, gate_2, conductance = load_qcodes_data(456, 'Lockin_conductance')
    > plt.pcolormesh(gate_1, gate_2, conductance)
    '''
    data = qc.load_by_id(id)
    data_dict = data.get_data_as_pandas_dataframe()

    try:
        d = data_dict[variable]

        # 2D data
        if isinstance(d.index, pd.core.index.MultiIndex):
            d = d.unstack()
            x = d.index.values#[::-1]
            y = d.columns.get_level_values(1).values
            v = d.values.transpose()#[:,::-1]

            x_name = d.index.name
            y_name = d.columns.names[1]
            v_name = d.columns.get_level_values(0)[0]

            return x,y,v#,x_name,y_name,v_name

        # 1D data
        else:
            x = d.index.values
            v = d.values.squeeze()

            x_name = d.index.name
            v_name = d.columns[0]

            return x,v#,x_name,v_name
    except KeyError:
        print(list(data_dict.keys()))

def load_last_qcodes_data(variable, earlier_index=0):
    '''
    Same as load_qcodes_data except loads the latest dataset.
    earlier_index allows to load older data.
    earlier_index=0 is latest dataset,
    earlier_index=1 is 2nd latest dataset etc.
    '''
    e = qc.load_last_experiment()
    ds = e.last_data_set()
    run_id = int(ds.run_id) - earlier_index
    return load_qcodes_data_by_id(run_id, variable)

