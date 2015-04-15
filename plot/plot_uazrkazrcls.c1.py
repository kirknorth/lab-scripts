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
REFL_CMAP = cm.NWSRef
VDOP_CMAP = plt.get_cmap(name='jet')
SNR_CMAP = plt.get_cmap(name='jet')
WIDTH_CMAP = plt.get_cmap(name='jet')
MASK_CMAP = plt.get_cmap(name='bone_r')
ECHO_CMAP = plt.get_cmap(name='jet')
WVEL_CMAP = plt.get_cmap(name='jet')
VFALL_CMAP = plt.get_cmap(name='jet')

# Define colour map bins
REFL_NORM = BoundaryNorm(np.arange(-40, 52, 2), REFL_CMAP.N)
VDOP_NORM = BoundaryNorm(np.arange(-6, 6.1, 0.1), VDOP_CMAP.N)
SNR_NORM = BoundaryNorm(np.arange(-40, 41, 1), SNR_CMAP.N)
WIDTH_NORM = BoundaryNorm(np.arange(0, 3.1, 0.1), WIDTH_CMAP.N)
MASK_NORM = BoundaryNorm(np.arange(0, 3, 1), MASK_CMAP.N)
ECHO_NORM = BoundaryNorm(np.arange(0, 11, 1), ECHO_CMAP.N)
WVEL_NORM = BoundaryNorm(np.arange(-8, 8.2, 0.2), WVEL_CMAP.N)
VFALL_NORM = BoundaryNorm(np.arange(0, 8.1, 0.1), VFALL_CMAP.N)

# Define colour bar ticks
REFL_TICKS = np.arange(-40, 60, 10)
VDOP_TICKS = np.arange(-6, 8, 2)
SNR_TICKS = np.arange(-40, 50, 10)
WIDTH_TICKS = np.arange(0, 3.5, 0.5)
MASK_TICKS = np.arange(0, 3, 1)
ECHO_TICKS = np.arange(0, 11, 1)
WVEL_TICKS = np.arange(-8, 10, 2)
VFALL_TICKS = np.arange(0, 9, 1)


def multipanel(filename, outdir, dpi=100, verbose=False):
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

    # Read radar data
    radar = Dataset(filename, mode='r')

    if verbose:
        print 'Plotting file: {}'.format(os.path.basename(filename))

    for i, hour in enumerate(HOURS):

        subs = {'xlim': (0, 60), 'ylim': (0, 14)}
        figs = {'figsize': (36, 30)}
        fig, ax = plt.subplots(nrows=5, ncols=3, subplot_kw=subs, **figs)
        plt.subplots_adjust(wspace=0.05, hspace=0.3)

        if verbose:
            print 'Plotting hour: {:02d}'.format(hour)

        # (a) Reflectivity
        _pcolormesh(
            radar, KAZR_REFL_FIELD, hour=hour, cmap=REFL_CMAP, norm=REFL_NORM,
            ticks=REFL_TICKS, fig=fig, ax=ax[0,0])
        _pcolormesh(
            radar, RWP_REFL_FIELD, hour=hour, cmap=REFL_CMAP, norm=REFL_NORM,
            ticks=REFL_TICKS, fig=fig, ax=ax[0,1])
        _pcolormesh(
            radar, REFL_FIELD, hour=hour, cmap=REFL_CMAP, norm=REFL_NORM,
            ticks=REFL_TICKS, fig=fig, ax=ax[0,2])

        # (b) Doppler velocity
        _pcolormesh(
            radar, KAZR_VDOP_FIELD, hour=hour, scale=-1.0, cmap=VDOP_CMAP,
            norm=VDOP_NORM, ticks=VDOP_TICKS, fig=fig, ax=ax[1,0])
        _pcolormesh(
            radar, RWP_VDOP_FIELD, hour=hour, scale=-1.0, cmap=VDOP_CMAP,
            norm=VDOP_NORM, ticks=VDOP_TICKS, fig=fig, ax=ax[1,1])
        _pcolormesh(
            radar, VDOP_FIELD, hour=hour, scale=-1.0, cmap=VDOP_CMAP,
            norm=VDOP_NORM, ticks=VDOP_TICKS, fig=fig, ax=ax[1,2])

        # (c) Signal-to-noise ratio
        _pcolormesh(
            radar, KAZR_SNR_FIELD, hour=hour, cmap=SNR_CMAP, norm=SNR_NORM,
            ticks=SNR_TICKS, fig=fig, ax=ax[2,0])
        _pcolormesh(
            radar, RWP_SNR_FIELD, hour=hour, cmap=SNR_CMAP, norm=SNR_NORM,
            ticks=SNR_TICKS, fig=fig, ax=ax[2,1])
        _pcolormesh(
            radar, SNR_MASK_FIELD, hour=hour, cmap=MASK_CMAP, norm=MASK_NORM,
            ticks=MASK_TICKS, fig=fig, ax=ax[2,2])

        # (d) Spectrum width (co-polar)
        _pcolormesh(
            radar, KAZR_WIDTH_FIELD, hour=hour, cmap=WIDTH_CMAP,
            norm=WIDTH_NORM, ticks=WIDTH_TICKS, fig=fig, ax=ax[3,0])
        _pcolormesh(
            radar, RWP_WIDTH_FIELD, hour=hour, cmap=WIDTH_CMAP,
            norm=WIDTH_NORM, ticks=WIDTH_TICKS, fig=fig, ax=ax[3,1])
        _pcolormesh(
            radar, WIDTH_FIELD, hour=hour, cmap=WIDTH_CMAP, norm=WIDTH_NORM,
            ticks=WIDTH_TICKS, fig=fig, ax=ax[3,2])

        # (e) Echo classification
        _pcolormesh(
            radar, ECHO_FIELD, hour=hour, cmap=ECHO_CMAP, norm=ECHO_NORM,
            ticks=ECHO_TICKS, fig=fig, ax=ax[4,0])

        # (f) Fall speed
        _pcolormesh(
            radar, VFALL_FIELD, hour=hour, cmap=VFALL_CMAP, norm=VFALL_NORM,
            ticks=VFALL_TICKS, fig=fig, ax=ax[4,1])

        # (g) Vertical velocity
        _pcolormesh(
            radar, WVEL_FIELD, hour=hour, scale=-1.0, cmap=WVEL_CMAP,
            norm=WVEL_NORM, ticks=WVEL_TICKS, fig=fig, ax=ax[4,2])

        # Format axes
        for i, j in np.ndindex(ax.shape):
            ax[i,j].xaxis.set_major_locator(MultipleLocator(5))
            ax[i,j].xaxis.set_minor_locator(MultipleLocator(1))
            ax[i,j].yaxis.set_major_locator(MultipleLocator(1))
            ax[i,j].yaxis.set_minor_locator(MultipleLocator(0.5))
            ax[i,j].set_xlabel('Minutes of Hour {:02d} (UTC)'.format(hour))
            ax[i,j].set_ylabel('Height (km AGL)')
            ax[i,j].grid(which='major', axis='both')

        # Save image
        time_offset = radar.variables['time_offset']
        date = num2date(time_offset[0], time_offset.units).date()
        image = '{}.{:02d}0000.png'.format(date.strftime('%Y%m%d'), hour)
        fig.savefig(os.path.join(outdir, image), format='png', dpi=dpi,
                    bbox_inches='tight')
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
    title = 'RWP/KAZR\n{}'.format(radar.variables[field].long_name)
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


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('filename', type=str, help=None)
    parser.add_argument('outdir', type=str, help=None)
    parser.add_argument('--inpdir', nargs='?', type=str, const=None,
                        default=None, help=None)
    parser.add_argument('--dpi', nargs='?', type=int, const=50, default=50,
                        help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, const=True,
                        default=False, help=None)
    parser.add_argument('-db', '--debug', nargs='?', type=bool, const=True,
                        default=False, help=None)
    args = parser.parse_args()

    # Parse radar file
    if args.inpdir is not None:
        args.filename = os.path.join(args.inpdir, args.filename)

    # Call desired plotting function
    multipanel(args.filename, args.outdir, dpi=args.dpi, verbose=args.verbose)

