#!/usr/bin/python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import num2date
from matplotlib import rcParams
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import make_axes
from matplotlib.dates import MinuteLocator, HourLocator, DateFormatter
from matplotlib.ticker import MultipleLocator

from pyart.graph import cm
from pyart.io import read_cfradial


EXCLUDE_FIELDS = None

# Define the hours to plot
HOURS = np.arange(0, 24, 1)

# Define radar fields
REFL_FIELD = 'reflectivity_copol'
VDOP_FIELD = 'mean_doppler_velocity_copol'
CORR_VDOP_FIELD = 'corrected_mean_doppler_velocity_copol'
SNR_FIELD = 'signal_to_noise_ratio_copol'
WIDTH_FIELD = 'spectral_width_copol'
VDOP_COH_FIELD = 'mean_doppler_velocity_copol_coherency_mask'
NOISE_FIELD = 'radar_noise_floor_mask'
DETECT_FIELD = 'significant_detection_mask'

# Define colour maps
REFL_CMAP = cm.NWSRef
VDOP_CMAP = plt.get_cmap(name='jet')
SNR_CMAP = plt.get_cmap(name='jet')
WIDTH_CMAP = plt.get_cmap(name='jet')
MASK_CMAP = plt.get_cmap(name='bone_r')

# Define colour map bins
REFL_NORM = BoundaryNorm(np.arange(-40, 32, 2), REFL_CMAP.N)
VDOP_NORM = BoundaryNorm(np.arange(-6, 6.1, 0.1), VDOP_CMAP.N)
SNR_NORM = BoundaryNorm(np.arange(-40, 41, 1), SNR_CMAP.N)
WIDTH_NORM = BoundaryNorm(np.arange(0, 1.01, 0.01), WIDTH_CMAP.N)
MASK_NORM = BoundaryNorm(np.arange(0, 3, 1), MASK_CMAP.N)

# Define colour bar ticks
REFL_TICKS = np.arange(-40, 40, 10)
VDOP_TICKS = np.arange(-6, 8, 2)
SNR_TICKS = np.arange(-40, 50, 10)
WIDTH_TICKS = np.arange(0, 1.1, 0.1)
MASK_TICKS = np.arange(0, 3, 1)


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
    radar = read_cfradial(filename, exclude_fields=EXCLUDE_FIELDS)

    if verbose:
        print 'Plotting file: {}'.format(os.path.basename(filename))

    for i, hour in enumerate(HOURS):

        subs = {'xlim': (0, 60), 'ylim': (0, 14)}
        figs = {'figsize': (28, 20)}
        fig, ax = plt.subplots(nrows=4, ncols=2, subplot_kw=subs, **figs)
        plt.subplots_adjust(wspace=0.05, hspace=0.3)

        if verbose:
            print 'Plotting hour: {:02d}'.format(hour)

        # (a) Raw reflectivity (co-polar)
        _pcolormesh(
            radar, REFL_FIELD, hour=hour, cmap=REFL_CMAP, norm=REFL_NORM,
            ticks=REFL_TICKS, fig=fig, ax=ax[0,0])

        # (b) Raw Doppler velocity (co-polar)
        _pcolormesh(
            radar, VDOP_FIELD, hour=hour, cmap=VDOP_CMAP, norm=VDOP_NORM,
            ticks=VDOP_TICKS, fig=fig, ax=ax[0,1])

        # (c) Signal-to-noise ratio (co-polar)
        _pcolormesh(
            radar, SNR_FIELD, hour=hour, cmap=SNR_CMAP, norm=SNR_NORM,
            ticks=SNR_TICKS, fig=fig, ax=ax[1,0])

        # (d) Corrected Doppler velocity (co-polar)
        _pcolormesh(
            radar, CORR_VDOP_FIELD, hour=hour, cmap=VDOP_CMAP, norm=VDOP_NORM,
            ticks=VDOP_TICKS, fig=fig, ax=ax[1,1])

        # (e) Hildebrand noise mask
        _pcolormesh(
            radar, NOISE_FIELD, hour=hour, cmap=MASK_CMAP, norm=MASK_NORM,
            ticks=MASK_TICKS, fig=fig, ax=ax[2,0])

        # (f) Doppler velocity coherency mask
        _pcolormesh(
            radar, VDOP_COH_FIELD, hour=hour, cmap=MASK_CMAP, norm=MASK_NORM,
            ticks=MASK_TICKS, fig=fig, ax=ax[2,1])

        # (g) Radar significant detection mask
        _pcolormesh(
            radar, DETECT_FIELD, hour=hour, cmap=MASK_CMAP, norm=MASK_NORM,
            ticks=MASK_TICKS, fig=fig, ax=ax[3,0])

        # (h) Spectrum width (co-polar)
        _pcolormesh(
            radar, WIDTH_FIELD, hour=hour, cmap=WIDTH_CMAP, norm=WIDTH_NORM,
            ticks=WIDTH_TICKS, fig=fig, ax=ax[3,1])

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
        date = num2date(radar.time['data'][0], radar.time['units']).date()
        image = '{}.{:02d}0000.png'.format(date.strftime('%Y%m%d'), hour)
        fig.savefig(os.path.join(outdir, image), format='png', dpi=dpi,
                    bbox_inches='tight')
        plt.close(fig)

    return


def _pcolormesh(radar, field, hour=0, cmap=None, norm=None, ticks=None,
                fig=None, ax=None):
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
    height = radar.range['data'] / 1000.0
    time = radar.time['data']

    # Bracket time indices for specified hour and convert time variable to
    # minutes
    t0, tn = _bracket_time(radar, hour=hour)
    time = (time[t0:tn] - 3600 * hour) / 60.0

    # Create quadmesh
    qm = ax.pcolormesh(time, height, radar.fields[field]['data'][t0:tn].T,
                       cmap=cmap, norm=norm)

    # Create colour bar
    fig.colorbar(mappable=qm, ax=ax, extend='neither', orientation='vertical',
                 ticks=ticks)

    # Set title
    title = '{}\n{}'.format(
        radar.metadata['instrument_name'], radar.fields[field]['long_name'])
    ax.set_title(title)

    return


def _bracket_time(radar, hour=0):
    """
    """

    # Parse radar time coordinate
    # The time variable should be seconds since midnight
    time = radar.time['data']

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

