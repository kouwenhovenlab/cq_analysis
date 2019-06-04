from qcodes.data.format import Formatter
from qcodes.data.data_array import DataArray

class SpyViewFormat(Formatter):
    def __init__(self):
        pass

    def read_metadata(self, data_set):
        pass

    def read_one_file(self, data_set, f, ids_read):
        # I have written more flexible code in my life
        if len(data_set.arrays) != 0:
            return
        line, data_set.header = self._read_header(f, "# ")
        # interpret metadata (header)

        set_columns = []
        set_arrays = ()
        data_columns = []
        data_arrays = []
        c = [i + 1 for i in range(len(line.split('\t')))]
        print(c)
        for i in c:
            if 'Column ' + str(i) not in data_set.header:
                raise KeyError('File header does not match nr of columns '
                               '(Column' + str(i) + ')')

            info = data_set.header['Column ' + str(i)]
            if info['type'] == 'coordinate':
                set_columns.append(['Column ' + str(i), i-1])
            elif info['type'] == 'value':
                data_columns.append(['Column ' + str(i), i-1])

        set_columns.reverse()
        shape = ()
        for n in set_columns:
            size = data_set.header[n[0]]['size']
            shape += (int(size),)
            name = data_set.header[n[0]]['name']
            set_array = DataArray(label=name,
                                  array_id="DataArray"+str(n[1]),
                                  set_arrays=set_arrays, shape=shape,
                                  is_setpoint=True, snapshot=None)
            set_array.init_data()
            data_set.add_array(set_array)
            set_arrays = set_arrays + (set_array,)

        for n in data_columns:
            name = data_set.header[n[0]]['name']
            data_array = DataArray(label=name,
                                   array_id="DataArray" + str(n[1]),
                                   set_arrays=set_arrays, shape=shape,
                                   snapshot=None)
            data_array.init_data()
            data_set.add_array(data_array)
            data_arrays.append(data_array)

        npoints = 1
        print(shape)
        for n in shape:
            npoints = npoints * n
        print(npoints)

        indices = [0] * len(set_arrays)

        elements = line.split('\t')
        set_values = []
        for i in set_columns:
            set_values.append(float(elements[i[1]]))
        p = 0
        watchdog = 0
        while True:
            # print(indices)
            # handle all elements in this line
            if len(elements) != len(data_set.arrays):
                raise Exception("Wrong number of columns in data row " +
                                str(p))

            # Check hypercubic consistency
            set_value_check = []
            for i in set_columns:
                set_value_check.append(float(elements[i[1]]))

            for a, b in zip(set_value_check, set_values):
                if a != b:
                    raise Exception("data is not hypercubic")

            # add data points to the dataset
            for c in data_columns:
                i = tuple(indices)
                data_set.arrays['DataArray'+str(c[1])][i] = float(elements[c[1]])

            # add set_points to the dataset

            for j, c in enumerate(set_columns):
                i = tuple(indices)[0:j+1]
                data_set.arrays['DataArray' + str(c[1])][i] = float(elements[c[1]])

            p += 1
            if p >= npoints:
                break

            line = ""
            watchdog = 0
            while line is "" and watchdog < 100:
                line = f.readline()
                line = line.rstrip('\n\t\r')
                watchdog += 1
            if watchdog >= 100:
                break


            elements = line.split('\t')
            # update indices en set new set_values to check hypercubicity
            pointer = len(indices) - 1

            indices[pointer] += 1
            # new set value and update it to dataset
            val = float(elements[set_columns[pointer][1]])
            set_values[pointer] = val

            while pointer >= 0 and indices[pointer] >= shape[pointer]:
                # reset and decrease pointer
                indices[pointer] = 0
                pointer -= 1

                indices[pointer] += 1
                # new set value and update it to dataset
                val = float(elements[set_columns[pointer][1]])
                set_values[pointer] = val

        if watchdog >= 100:
            print("unexpected EOF")
            print(p)
            print(npoints)

        line = f.readline()
        line = line.rstrip('\n\t\r')
        if line is not "":
            raise Exception("File is supposed to be empty, but nonempty line "
                            "found (" + line + ")")


    def _read_header(self, f, prefix):
        header = {}
        for line in f:
            newline = line.rstrip('\n\t\r')
            while newline:
                line = newline
                newline = None
                if line.startswith(prefix):
                    components = line.split(': ', 1)
                    if len(components) == 1:
                        newline, subheader = self._read_header(f, "#\t")
                        header.update({components[0][2:-1]: subheader})
                    else:
                        header.update({components[0][2:]: components[1]})
                elif len(line) == 0:
                    pass
                else:
                    return line, header
        return "", header
