#!/usr/bin/python

import os
import argparse
import json
import numpy as np

from re import search
from itertools import groupby
from datetime import datetime

### GLOBAL VARIABLES ###
########################

# Common input directories
INPUT_DIRS = ['/storage/kirk/MC3E/processed_data/csapr/',
              '/storage/kirk/MC3E/processed_data/xsapr/',
              '/storage/kirk/MC3E/processed_data/nexrad/']


def sort_by_time(stamps, suffix='cdf', verbose=False):
    """
    """

    # Parse all sources for any specified stamps
    files = []
    for inpdir in INPUT_DIRS:
        for f in sorted(os.listdir(inpdir)):
            if any(stamp in f for stamp in stamps) and f.endswith(suffix):
                files.append(f)
    if verbose:
        print 'Total number of grids = %i' % len(files)

    # Compute the date and time for each file, and determine the indices that
    # would sort the dates and times in ascending order
    matches = [search('[0-9]{8}\.[0-9]{6}', f).group() for f in files]
    times = [datetime.strptime(match, '%Y%m%d.%H%M%S') for match in matches]
    indices = np.argsort(times, kind='mergesort')

    return [files[idx] for idx in indices], [times[idx] for idx in indices]


def group_by_lag(files, times, lag=240, remove_single=False, verbose=False):
    """
    """

    # Compute time offset from first grid time
    tdeltas = [time - times[0] for time in times]
    data = [[f, td] for f, td in zip(files, tdeltas)]
    group_key = lambda (f, td): td.total_seconds() // lag

    groups = []
    for key, group in groupby(data, key=group_key):
        groups.append(list(group))

    # Remove groups with only one radar grid
    if remove_single:
        groups = [group for group in groups if len(group) > 1]

    # Remove timedelta information from groups
    groups = [[grid[0] for grid in group] for group in groups]

    if verbose:
        print ('Total number of grid groups within {} sec of '
               'each other = {}'.format(lag, len(groups)))

    return groups


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('stamps', nargs='+', type=str, help=None)
    parser.add_argument('outdir', type=str, help=None)
    parser.add_argument('filename', type=str, help=None)
    parser.add_argument('--lag', nargs='?', type=int, const=120, default=120,
                        help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, const=True,
                        default=False, help=None)
    parser.add_argument('-db', '--debug', nargs='?', type=bool, const=True,
                        default=False, help=None)
    args = parser.parse_args()

    if args.debug:
        print 'stamps = %s' % args.stamps
        print 'outdir = %s' % args.outdir
        print 'filename = %s' % args.filename
        print 'lag = %s' % args.lag

    # First sort all files by time in ascending order
    files, times = sort_by_time(
        args.stamps, suffix='cdf', verbose=args.verbose)

    # Group files by lag
    groups = group_by_lag(files, times, lag=args.lag, verbose=args.verbose)

    # Save grouped grids to a JSON file
    with open(os.path.join(args.outdir, args.filename), 'w') as fid:
        json.dump(groups, fid, indent=2)
