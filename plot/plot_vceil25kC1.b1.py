#!/usr/bin/python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset, num2date
from matplotlib import rcParams
from matplotlib.colors import BoundaryNorm
from matplotlib.dates import MinuteLocator, HourLocator, DateFormatter
from matplotlib.ticker import MultipleLocator

from pyart.graph import cm

# Define the hours to plot
HOURS = np.arange(0, 24, 1)

# Define radar fields
SCAT_FIELD = 'backscatter'
CBH1_FIELD = 'first_cbh'
CBH2_FIELD = 'second_cbh'
CBH3_FIELD = 'third_cbh'
DETECT_FIELD = 'detection_status'

# Define colour maps
CMAP_SCAT = cm.NWSRef

# Normalize colour maps
NORM_SCAT = BoundaryNorm(np.arange(0, 10.05, 0.05), CMAP_SCAT.N)

# Define colour bar ticks
TICKS_SCAT = np.arange(0, 11, 1)


def multipanel(radar, outdir, dpi=100, verbose=False):
    """
    """

    # Set figure parameters
    rcParams['axes.linewidth'] = 1.5
    rcParams['xtick.major.size'] = 4
    rcParams['xtick.major.width'] = 1
    rcParams['xtick.minor.size'] = 2
    rcParams['xtick.minor.width'] = 1
    rcParams['ytick.major.size'] = 4
    rcParams['ytick.major.width'] = 1
    rcParams['ytick.minor.size'] = 2
    rcParams['ytick.minor.width'] = 1

    for i, hour in enumerate(HOURS):

        if verbose:
            print 'Plotting hour: {:02d}'.format(hour)

        # Bracket time indices for specified hour
        time, t0, tn = _bracket_time(radar, hour=hour)

        # Initialze figure
        subs = {'xlim': (0, 60), 'ylim': (0, 7)}
        figs = {'figsize': (10, 4)}
        fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw=subs, **figs)
        plt.subplots_adjust(wspace=None, hspace=None)

        # (a) Backscatter
        _pcolormesh(
            radar, SCAT_FIELD, hour=hour, scale=1.0e-1, decibel=False,
            cmap=CMAP_SCAT, norm=NORM_SCAT, ticks=TICKS_SCAT, fig=fig,
            ax=ax)

        # (b) First cloud base height
        ax.scatter(
            time, radar.variables[CBH1_FIELD][t0:tn+1] / 1000.0, s=20,
            facecolor='orange', edgecolor='black', marker='o',
            label='CBH$_1$')

        # (c) Second cloud base height
        ax.scatter(
            time, radar.variables[CBH2_FIELD][t0:tn+1] / 1000.0, s=10,
            facecolor='white', edgecolor='black', marker='D',
            label='CBH$_2$')

        # (d) Third cloud base height
        ax.scatter(
            time, radar.variables[CBH3_FIELD][t0:tn+1] / 1000.0, s=10,
            facecolor='pink', edgecolor='black', marker='>',
            label='CBH$_3$')

        # Format axes
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.set_xlabel('Minutes of Hour {:02d} (UTC)'.format(hour))
        ax.set_ylabel('Height (km AGL)')
        ax.grid(which='major', axis='both')

        # Add legend
        ax.legend(loc=[0.21, -0.29], ncol=3)

        # Save image
        date = num2date(
            radar.variables['time'][0], radar.variables['time'].units).date()
        image = '{}.{:02d}0000.png'.format(date.strftime('%Y%m%d'), hour)
        fig.savefig(os.path.join(outdir, image), format='png', dpi=dpi,
                    bbox_inches='tight')
        plt.close(fig)

    return


def _pcolormesh(radar, field, hour=0, scale=1.0, decibel=False, cmap=None,
                norm=None, ticks=None, fig=None, ax=None):
    """
    """

    # Parse figure and axis
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        fig = plt.gca()

    # Parse colour map
    if cmap is None:
        cmap = plt.get_cmap(name='jet')

    # Parse radar coordinates
    # The time variable should be seconds since midnight
    height = radar.variables['range'][:] / 1000.0
    time = radar.variables['time'][:]

    # Bracket time indices for specified hour
    time, t0, tn = _bracket_time(radar, hour=hour)

    # Parse radar data within specified hour
    data = radar.variables[field][t0:tn+1]
    if decibel:
        data = 10.0 * np.ma.log10(data)

    # Create quadmesh
    qm = ax.pcolormesh(time, height, scale * data.T, cmap=cmap, norm=norm)

    # Create colour bar
    fig.colorbar(mappable=qm, ax=ax, extend='neither', orientation='vertical',
                 ticks=ticks)

    # Set title
    title = 'Ceilometer\n{}'.format(radar.variables[field].long_name)
    ax.set_title(title)

    return


def _bracket_time(radar, hour=0):
    """
    """

    # Parse radar time coordinate
    # The time variable should be seconds since midnight
    time = radar.variables['time'][:]

    # Bracket time indices for specified hour
    t0 = np.abs(time - hour * 3600).argmin()
    tn = np.abs(time - (hour + 1) * 3600).argmin()

    # Convert time to minutes of specified hour
    time = (time[t0:tn+1] - 3600 * hour) / 60.0

    return time, t0, tn


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('inpdir', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('outdir', type=str, help=None)
    parser.add_argument('--dpi', nargs='?', type=int, const=50, default=50,
                        help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, const=True,
                        default=False, help=None)
    parser.add_argument('-db', '--debug', nargs='?', type=bool, const=True,
                        default=False, help=None)
    args = parser.parse_args()

    # Parse files to plot
    files = [os.path.join(args.inpdir, f) for
             f in sorted(os.listdir(args.inpdir)) if args.stamp in f]

    if args.verbose:
        print 'Number of files to plot: {}'.format(len(files))

    for f in files:

        if args.verbose:
            print 'Plotting file: {}'.format(os.path.basename(f))

        # Read radar data
        radar = Dataset(f, mode='r')

        # Call desired plotting function
        multipanel(radar, args.outdir, dpi=args.dpi, verbose=args.verbose)

