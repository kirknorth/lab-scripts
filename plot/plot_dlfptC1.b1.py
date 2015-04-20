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


# Define radar fields
SCAT_FIELD = 'attenuated_backscatter'
SNR_FIELD = 'intensity'
VDOP_FIELD = 'radial_velocity'

# Define colour maps
CMAP_SCAT = cm.NWSRef
CMAP_SNR = plt.get_cmap(name='jet')
CMAP_VDOP = plt.get_cmap(name='jet')

# Normalize colour maps
NORM_SCAT = BoundaryNorm(np.arange(0, 10.1, 0.1), CMAP_SCAT.N)
NORM_SNR = BoundaryNorm(np.arange(1, 1.21, 0.01), CMAP_SNR.N)
NORM_VDOP = BoundaryNorm(np.arange(-3, 3.1, 0.1), CMAP_VDOP.N)

# Define colour bar ticks
TICKS_SCAT = np.arange(0, 11, 1)
TICKS_SNR = np.arange(1, 1.22, 0.02)
TICKS_VDOP = np.arange(-3, 3.5, 0.5)


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

    # Compute hour of radar
    hour = num2date(
        radar.variables['time'][0], radar.variables['time'].units).hour

    if verbose:
        print 'Plotting hour: {:02d}'.format(hour)

    # Initialze figure
    subs = {'xlim': (0, 60), 'ylim': (0, 5)}
    figs = {'figsize': (22, 19)}
    fig, ax = plt.subplots(nrows=3, ncols=1, subplot_kw=subs, **figs)
    plt.subplots_adjust(wspace=None, hspace=None)

    # (a) Attenuated backscatter
    _pcolormesh(
        radar, SCAT_FIELD, hour=hour, scale=1.0e6, cmap=CMAP_SCAT,
        norm=NORM_SCAT, ticks=TICKS_SCAT, fig=fig, ax=ax[0])

    # (b) Intensity (signal-to-noise-ratio)
    _pcolormesh(
        radar, SNR_FIELD, hour=hour, decibel=False, cmap=CMAP_SNR,
        norm=NORM_SNR, ticks=TICKS_SNR, fig=fig, ax=ax[1])

    # (c) Doppler (radial) velocity
    _pcolormesh(
        radar, VDOP_FIELD, hour=hour, cmap=CMAP_VDOP, norm=NORM_VDOP,
        ticks=TICKS_VDOP, fig=fig, ax=ax[2])

    # Format axes
    for i in range(ax.size):
        ax[i].xaxis.set_major_locator(MultipleLocator(5))
        ax[i].xaxis.set_minor_locator(MultipleLocator(1))
        ax[i].yaxis.set_major_locator(MultipleLocator(0.5))
        ax[i].yaxis.set_minor_locator(MultipleLocator(0.1))
        ax[i].set_xlabel('Minutes of Hour {:02d} (UTC)'.format(hour))
        ax[i].set_ylabel('Height (km AGL)')
        ax[i].grid(which='major', axis='both')

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

    # Convert time variable to minutes of hour
    time = (time - 3600 * hour) / 60.0

    # Parse radar data
    data = radar.variables[field][:]
    if decibel:
        data = 10.0 * np.log10(data)

    # Create quadmesh
    qm = ax.pcolormesh(time, height, scale * data.T, cmap=cmap, norm=norm)

    # Create colour bar
    fig.colorbar(mappable=qm, ax=ax, extend='neither', orientation='vertical',
                 ticks=ticks)

    # Set title
    title = 'Doppler Lidar\n{}'.format(radar.variables[field].long_name)
    ax.set_title(title)

    return


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

