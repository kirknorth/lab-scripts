#!/usr/bin/python

import os
import argparse
import numpy as np
import collections

from re import search
from datetime import datetime


def sort_by_time(sources, stamps, suffix='cdf', verbose=False):
    """
    """

    # Parse all sources for any specified stamps
    files = []
    for source in sources:
        for f in sorted(os.listdir(source)):
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


def group_by_lag(files, times, stamps, lag=120):
    """
    """

    print np.diff(times, n=1)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--sources', nargs='*', type=str, help=None)
    parser.add_argument('-id', '--stamps', nargs='*', type=str, help=None)
    parser.add_argument('-o', '--output', type=str, help=None)
    parser.add_argument('--lag', nargs='?', type=int, const=120, default=120,
                        help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, const=True,
                        default=False, help=None)
    args = parser.parse_args()

    print args.sources
    print args.stamps

    # First sort all files by time in ascending order
    files, times = sort_by_time(args.sources, args.stamps, suffix='cdf',
                                verbose=args.verbose)

    # Group files by lag
    group_by_lag(files, times, args.stamps, lag=args.lag)
