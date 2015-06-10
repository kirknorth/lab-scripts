#!/usr/bin/python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset, num2date
from matplotlib import rcParams
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import make_axes
from matplotlib.dates import MinuteLocator, HourLocator, DateFormatter
from matplotlib.ticker import MultipleLocator

from pyart.graph import cm


# Define the hours to plot
HOURS = np.arange(0, 24, 1)

# Define radar fields
RWP_REFL_FIELD = 'ReflectivityUAZR'
KAZR_REFL_FIELD = 'ReflectivityKAZR'
REFL_FIELD = 'ReflectivityCombined'
RWP_VDOP_FIELD = 'MeanDopplerVelocityUAZR'
KAZR_VDOP_FIELD = 'MeanDopplerVelocityKAZR'
VDOP_FIELD = 'MeanDopplerVelocityCombined'
RWP_WIDTH_FIELD = 'SpectralWidthUAZR'
KAZR_WIDTH_FIELD = 'SpectralWidthKAZR'
WIDTH_FIELD = 'SpectralWidthCombined'
RWP_SNR_FIELD = 'SignalToNoiseRatioUAZR'
KAZR_SNR_FIELD = 'SignalToNoiseRatioKAZR'
SNR_MASK_FIELD = 'SignalToNoiseRatioFilteredMask'
ECHO_FIELD = 'EchoClassification'
VFALL_FIELD = 'FallSpeed'
WVEL_FIELD = 'VerticalVelocity'

# Define colour maps
CMAP_REFL = cm.NWSRef
CMAP_VDOP = plt.get_cmap(name='jet')
CMAP_SNR = plt.get_cmap(name='jet')
CMAP_WIDTH = plt.get_cmap(name='jet')
CMAP_MASK = plt.get_cmap(name='bone_r')
CMAP_ECHO = plt.get_cmap(name='jet')
CMAP_WVEL = plt.get_cmap(name='jet')
CMAP_VFALL = plt.get_cmap(name='jet')

# Define colour map bins
NORM_REFL = BoundaryNorm(np.arange(-40, 52, 2), CMAP_REFL.N)
NORM_VDOP = BoundaryNorm(np.arange(-6, 6.1, 0.1), CMAP_VDOP.N)
NORM_SNR = BoundaryNorm(np.arange(-40, 41, 1), CMAP_SNR.N)
NORM_WIDTH_KAZR = BoundaryNorm(np.arange(0, 3.05, 0.05), CMAP_WIDTH.N)
NORM_WIDTH_RWP = BoundaryNorm(np.arange(0, 8.1, 0.1), CMAP_WIDTH.N)
NORM_WIDTH = BoundaryNorm(np.arange(0, 8.1, 0.1), CMAP_WIDTH.N)
NORM_MASK = BoundaryNorm(np.arange(0, 3, 1), CMAP_MASK.N)
NORM_ECHO = BoundaryNorm(np.arange(0, 11, 1), CMAP_ECHO.N)
NORM_WVEL = BoundaryNorm(np.arange(-8, 8.2, 0.2), CMAP_WVEL.N)
NORM_VFALL = BoundaryNorm(np.arange(0, 8.1, 0.1), CMAP_VFALL.N)

# Define colour bar ticks
TICKS_REFL = np.arange(-40, 60, 10)
TICKS_VDOP = np.arange(-6, 8, 2)
TICKS_SNR = np.arange(-40, 50, 10)
TICKS_WIDTH_KAZR = np.arange(0, 3.5, 0.5)
TICKS_WIDTH_RWP = np.arange(0, 9, 1)
TICKS_WIDTH = np.arange(0, 9, 1)
TICKS_MASK = np.arange(0, 3, 1)
TICKS_ECHO = np.arange(0, 11, 1)
TICKS_WVEL = np.arange(-8, 10, 2)
TICKS_VFALL = np.arange(0, 9, 1)


def multipanel(radar, outdir, dpi=50, debug=False, verbose=False):
    """
    """

    # Set figure parameters
    rcParams['font.size'] = 14
    rcParams['font.weight'] = 'bold'
    rcParams['axes.titlesize'] = 14
    rcParams['axes.titleweight'] = 'bold'
    rcParams['axes.labelsize'] = 14
    rcParams['axes.labelweight'] = 'bold'
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

        subs = {'xlim': (0, 60), 'ylim': (0, 12)}
        figs = {'figsize': (36, 30)}
        fig, ax = plt.subplots(nrows=5, ncols=3, subplot_kw=subs, **figs)
        plt.subplots_adjust(wspace=0.05, hspace=0.3)

        if verbose:
            print 'Plotting hour: {:02d}'.format(hour)

        # (a) Reflectivity
        _pcolormesh(
            radar, KAZR_REFL_FIELD, hour=hour, cmap=CMAP_REFL, norm=NORM_REFL,
            ticks=TICKS_REFL, fig=fig, ax=ax[0,0])
        _pcolormesh(
            radar, RWP_REFL_FIELD, hour=hour, cmap=CMAP_REFL, norm=NORM_REFL,
            ticks=TICKS_REFL, fig=fig, ax=ax[0,1])
        _pcolormesh(
            radar, REFL_FIELD, hour=hour, cmap=CMAP_REFL, norm=NORM_REFL,
            ticks=TICKS_REFL, fig=fig, ax=ax[0,2])

        # (b) Doppler velocity
        _pcolormesh(
            radar, KAZR_VDOP_FIELD, hour=hour, scale=-1.0, cmap=CMAP_VDOP,
            norm=NORM_VDOP, ticks=TICKS_VDOP, fig=fig, ax=ax[1,0])
        _pcolormesh(
            radar, RWP_VDOP_FIELD, hour=hour, scale=-1.0, cmap=CMAP_VDOP,
            norm=NORM_VDOP, ticks=TICKS_VDOP, fig=fig, ax=ax[1,1])
        _pcolormesh(
            radar, VDOP_FIELD, hour=hour, scale=-1.0, cmap=CMAP_VDOP,
            norm=NORM_VDOP, ticks=TICKS_VDOP, fig=fig, ax=ax[1,2])

        # (c) Signal-to-noise ratio
        _pcolormesh(
            radar, KAZR_SNR_FIELD, hour=hour, cmap=CMAP_SNR, norm=NORM_SNR,
            ticks=TICKS_SNR, fig=fig, ax=ax[2,0])
        _pcolormesh(
            radar, RWP_SNR_FIELD, hour=hour, cmap=CMAP_SNR, norm=NORM_SNR,
            ticks=TICKS_SNR, fig=fig, ax=ax[2,1])
        _pcolormesh(
            radar, SNR_MASK_FIELD, hour=hour, cmap=CMAP_MASK, norm=NORM_MASK,
            ticks=TICKS_MASK, fig=fig, ax=ax[2,2])

        # (d) Spectrum width (co-polar)
        _pcolormesh(
            radar, KAZR_WIDTH_FIELD, hour=hour, cmap=CMAP_WIDTH,
            norm=NORM_WIDTH_KAZR, ticks=TICKS_WIDTH_KAZR, fig=fig, ax=ax[3,0])
        _pcolormesh(
            radar, RWP_WIDTH_FIELD, hour=hour, cmap=CMAP_WIDTH,
            norm=NORM_WIDTH_RWP, ticks=TICKS_WIDTH_RWP, fig=fig, ax=ax[3,1])
        _pcolormesh(
            radar, WIDTH_FIELD, hour=hour, cmap=CMAP_WIDTH, norm=NORM_WIDTH,
            ticks=TICKS_WIDTH, fig=fig, ax=ax[3,2])

        # (e) Echo classification
        _pcolormesh(
            radar, ECHO_FIELD, hour=hour, cmap=CMAP_ECHO, norm=NORM_ECHO,
            ticks=TICKS_ECHO, fig=fig, ax=ax[4,0])

        # (f) Fall speed
        _pcolormesh(
            radar, VFALL_FIELD, hour=hour, cmap=CMAP_VFALL, norm=NORM_VFALL,
            ticks=TICKS_VFALL, fig=fig, ax=ax[4,1])

        # (g) Vertical velocity
        _pcolormesh(
            radar, WVEL_FIELD, hour=hour, scale=-1.0, cmap=CMAP_WVEL,
            norm=NORM_WVEL, ticks=TICKS_WVEL, fig=fig, ax=ax[4,2])

        # Format axes
        for i, j in np.ndindex(ax.shape):
            ax[i,j].xaxis.set_major_locator(MultipleLocator(5))
            ax[i,j].xaxis.set_minor_locator(MultipleLocator(1))
            ax[i,j].yaxis.set_major_locator(MultipleLocator(1))
            ax[i,j].yaxis.set_minor_locator(MultipleLocator(0.5))
            ax[i,j].set_xlabel('Minutes of Hour {:02d} (UTC)'.format(hour))
            ax[i,j].set_ylabel('Height (km AGL)')
            ax[i,j].grid(which='major', axis='both')

        # Define image file name
        date_stamp = _datetimes(radar).min().date().strftime('%Y%m%d')
        fname = '{}.{:02d}0000.png'.format(date_stamp, hour)

        # Save figure
        fig.savefig(os.path.join(outdir, fname), format='png', dpi=dpi,
                    bbox_inches='tight')

        # Close figure to free memory
        plt.close(fig)

    return


def _pcolormesh(radar, field, hour=0, scale=1.0, cmap=None, norm=None,
                ticks=None, fig=None, ax=None):
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
    height = radar.variables['height'][:]
    time = radar.variables['time_offset'][:]

    # Bracket time indices for specified hour and convert time variable to
    # minutes
    t0, tn = _bracket_time(radar, hour=hour)
    time = (time[t0:tn] - 3600 * hour) / 60.0

    # Create quadmesh
    qm = ax.pcolormesh(time, height, scale * radar.variables[field][t0:tn].T,
                       cmap=cmap, norm=norm)

    # Create colour bar
    fig.colorbar(mappable=qm, ax=ax, extend='neither', orientation='vertical',
                 ticks=ticks)

    # Set title
    title = 'RWP/KAZR\n{}'.format(field)
    ax.set_title(title)

    return


def _bracket_time(radar, hour=0):
    """
    """

    # Parse radar time coordinate
    # The time variable should be seconds since midnight
    time = radar.variables['time_offset'][:]

    # Bracket time indices for specified hour
    t0 = np.abs(time - hour * 3600).argmin()
    tn = np.abs(time - (hour + 1) * 3600).argmin()

    return t0, tn


def _datetimes(radar):
    """
    """

    return num2date(radar.variables['time_offset'][:],
                    radar.variables['time_offset'].units)


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
    files = [os.path.join(args.inpdir, f) for f in
             sorted(os.listdir(args.inpdir)) if args.stamp in f]
    if args.verbose:
        print 'Number of files to plot: {}'.format(len(files))

    # Loop over all files
    for filename in files:
        if args.verbose:
            print 'Plotting file: {}'.format(os.path.basename(filename))

        # Read radar data
        radar = Dataset(filename)

        # Call desired plotting function
        multipanel(radar, args.outdir, dpi=args.dpi, debug=args.debug,
                   verbose=args.verbose)

