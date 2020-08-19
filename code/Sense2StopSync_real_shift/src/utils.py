import os
import csv
import numpy as np
import pandas as pd


def csv_read(path):
    """
    An alternative method to pandas read_csv, to help avoid the case where read_csv method cause error in multithreads.

    Args:
        path: str

    Returns:
        dataframe, data in 'path'

    """
    with open(path) as fd:
        rd = csv.reader(fd)
        header = next(rd)  # initialize column names from first row
        next_key = 0  # additional columns will start at '0'
        data = {k: list() for k in header}  # initialize data list per column
        for row in rd:
            while len(row) > len(header):  # add eventual new columns
                header.append(str(next_key))
                data[header[-1]] = [np.nan] * len(data[header[0]])
                next_key += 1  # increase next column name
            # eventually extend the row up to the header size
            row.extend([np.nan] * (len(header) - len(row)))
            # and add data to the column lists
            for i, k in enumerate(header): data[k].append(row[i])
    # data is now in a dict format, suitable to feed DataFrame
    return pd.DataFrame(data)


def list_files_in_directory(mypath):
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
