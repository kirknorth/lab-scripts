#!/usr/bin/python

import argparse
import numpy as np

from netCDF4 import Dataset

from pyart.io import read_grid
from pyart.retrieve import winds


def parse_grids(scene):
    """
    """
    if scene == 'convective':
        csapri7 = read_grid('')
        xsapri4 = read_grid('')
        xsapri5 = read_grid('')

    elif scene == 'stratiform':
        csapri7 = read_grid('')
        xsapri4 = read_grid('')
        xsapri5 = read_grid('')

    else:
        raise ValueError

    return [csapri7, xsapri4, xsapri5]



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('scene', type=str, help=None)
    parser.add_argument('output', type=str, help=None)
    parser.add_argument('filename', type=str, help=None)
    parser.add_argument('-v', '--verbose', nargs='?', const=True,
                        default=False, type=bool, help=None)
    args = parser.parse_args()
