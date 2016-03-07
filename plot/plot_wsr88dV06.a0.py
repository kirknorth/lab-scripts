#!/usr/bin/python

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import make_axes
from matplotlib.ticker import MultipleLocator

from pyart.io import read
from pyart.graph import cm
from pyart.config import get_field_name
from pyart.util.datetime_utils import datetimes_from_radar

# Define sweeps to plot
SWEEPS = [0, 2, 4]

# Define max range to plot
MAX_RANGE = 100.0

# Define field names
REFL_FIELD = get_field_name('reflectivity')
VDOP_FIELD = get_field_name('velocity')
SPW_FIELD = get_field_name('spectrum_width')
RHOHV_FIELD = get_field_name('cross_correlation_ratio')
ZDR_FIELD = get_field_name('differential_reflectivity')
PHIDP_FIELD = get_field_name('differential_phase')

# Define colour maps
CMAP_REFL = cm.NWSRef
CMAP_VDOP = cm.NWSVel
CMAP_SPW = cm.NWS_SPW
CMAP_RHOHV = cm.Carbone42
CMAP_ZDR = cm.RefDiff
CMAP_PHIDP = cm.Wild25

# Normalize colour maps
NORM_REFL = BoundaryNorm(np.arange(-10, 65, 5), CMAP_REFL.N)
NORM_VDOP = BoundaryNorm(np.arange(-30, 32, 2), CMAP_VDOP.N)
NORM_SPW = BoundaryNorm(np.arange(0, 8.5, 0.5), CMAP_SPW.N)
NORM_RHOHV = BoundaryNorm(np.arange(0, 1.05, 0.05), CMAP_RHOHV.N)
NORM_ZDR = BoundaryNorm(np.arange(-4, 4.5, 0.5), CMAP_ZDR.N)
NORM_PHIDP = BoundaryNorm(np.arange(0, 365, 5), CMAP_PHIDP.N)

# Define colour bar ticks
TICKS_REFL = np.arange(-10, 70, 10)
TICKS_VDOP = np.arange(-30, 40, 10)
TICKS_SPW = np.arange(0, 10, 2)
TICKS_RHOHV = np.arange(0, 1.2, 0.2)
TICKS_ZDR = np.arange(-4, 6, 2)
TICKS_PHIDP = np.arange(0, 450, 90)
TICKS = [
    TICKS_REFL,
    TICKS_VDOP,
    TICKS_SPW,
    TICKS_RHOHV,
    TICKS_ZDR,
    TICKS_PHIDP,
    ]

# Define figure paramters
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Bitstream Vera Sans'
rcParams['mathtext.default'] = 'regular'
rcParams['text.usetex'] = False
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.size'] = 6
rcParams['xtick.major.width'] = 1
rcParams['xtick.minor.size'] = 3
rcParams['xtick.minor.width'] = 1
rcParams['ytick.major.size'] = 6
rcParams['ytick.major.width'] = 1
rcParams['ytick.minor.size'] = 3
rcParams['ytick.minor.width'] = 1
rcParams['figure.figsize'] = (26, 12)


def multipanel(radar, outdir, dpi=90, debug=False, verbose=False):
    """
    """

    # Create figure instance
    subs = {'xlim': (-100, 100), 'ylim': (-100, 100)}
    fig, axes = plt.subplots(nrows=len(SWEEPS), ncols=6, subplot_kw=subs)
    fig.subplots_adjust(wspace=0.5, hspace=0.6)

    if debug:
        print('Number of sweeps: {}'.format(radar.nsweeps))

    # Loop over specified sweeps
    for i, sweep in enumerate(SWEEPS):

        if verbose:
            print('Plotting sweep index: {}'.format(sweep))

        # (a) Reflectivity
        qma = _pcolormesh(
            radar, REFL_FIELD, sweep=sweep, cmap=CMAP_REFL, norm=NORM_REFL,
            fig=fig, ax=axes[i,0])

        # (b) Doppler velocity
        qmb = _pcolormesh(
            radar, VDOP_FIELD, sweep=sweep+1, cmap=CMAP_VDOP, norm=NORM_VDOP,
            fig=fig, ax=axes[i,1])

        # (c) Spectrum width
        qmc = _pcolormesh(
            radar, SPW_FIELD, sweep=sweep+1, cmap=CMAP_SPW, norm=NORM_SPW,
            fig=fig, ax=axes[i,2])

        # (d) Copolar Correlation coefficient
        qmd = _pcolormesh(
            radar, RHOHV_FIELD, sweep=sweep, cmap=CMAP_RHOHV, norm=NORM_RHOHV,
            fig=fig, ax=axes[i,3])

        # (e) Differential reflectivity
        qme = _pcolormesh(
            radar, ZDR_FIELD, sweep=sweep, cmap=CMAP_ZDR, norm=NORM_ZDR,
            fig=fig, ax=axes[i,4])

        # (f) Differential phase
        qmf = _pcolormesh(
            radar, PHIDP_FIELD, sweep=sweep, cmap=CMAP_PHIDP, norm=NORM_PHIDP,
            fig=fig, ax=axes[i,5])

    # Create color bars
    qm = [qma, qmb, qmc, qmd, qme, qmf]
    _create_colorbars(fig, axes, qm, TICKS)

    # Format plot axes
    for ax in axes.flat:
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.set_xlabel('Eastward Range (km)')
        ax.set_ylabel('Northward Range (km)')
        ax.grid(which='major')

    # Define image file name
    date_stamp = datetimes_from_radar(radar).min().strftime('%Y%m%d.%H%M%S')
    fname = '{}.png'.format(date_stamp)

    # Save figure
    fig.savefig(os.path.join(outdir, fname), format='png', dpi=dpi,
                bbox_inches='tight', frameon=True, transparent=False)

    # Close figure to free memory
    plt.cla()
    plt.clf()
    plt.close(fig)

    return


def _pcolormesh(
        radar, field, sweep=0, cmap=None, norm=None, fig=None, ax=None):
    """
    """

    # Parse axis and figure parameters
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    # Parse radar coordinates
    # Convert angles to radians and range to kilometers
    _range = radar.range['data'] / 1000.0
    azimuth = np.radians(radar.get_azimuth(sweep))

    # Compute index of maximum range
    rf = np.abs(_range - MAX_RANGE).argmin()

    # Compute radar sweep coordinates
    AZI, RNG = np.meshgrid(azimuth, _range[:rf+1], indexing='ij')
    X = RNG * np.sin(AZI)
    Y = RNG * np.cos(AZI)

    # Parse radar data
    data = radar.get_field(sweep, field)[:,:rf+1]

    # Create quadmesh
    qm = ax.pcolormesh(
        X, Y, data, cmap=cmap, norm=norm, shading='flat', alpha=None,
        rasterized=True)

    # Create title
    s0, sf = radar.get_start_end(sweep)
    sweep_time = datetimes_from_radar(radar)[(s0 + sf) / 2]
    title = '{} {:.1f} deg\n{}Z\n{}'.format(
            radar.metadata['instrument_name'],
            radar.fixed_angle['data'][sweep],
            sweep_time.isoformat(), field)
    ax.set_title(title)

    return qm


def _create_colorbars(fig, axes, qm, ticks):
    """
    """

    for i in range(axes.shape[1]):
        parents = [ax for ax in axes[:,i].flat]
        cax, kw = make_axes(parents, location='bottom', pad=0.06, shrink=1.0,
                            fraction=0.01, aspect=20)
        fig.colorbar(mappable=qm[i], cax=cax, orientation='horizontal',
                     ticks=ticks[i], spacing='uniform', drawedges=False)

    return


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('inpdir', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('outdir', type=str, help=None)
    parser.add_argument('--dpi', nargs='?', type=int, const=90, default=90,
                        help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, const=True,
                        default=False, help=None)
    parser.add_argument('-db', '--debug', nargs='?', type=bool, const=True,
                        default=False, help=None)
    args = parser.parse_args()

    if args.debug:
        print('MAX_RANGE -> {}'.format(MAX_RANGE))

    # Parse radar files to plot
    files = [os.path.join(args.inpdir, f) for f
             in sorted(os.listdir(args.inpdir)) if args.stamp in f]

    if args.verbose:
        print('Number of files to plot: {}'.format(len(files)))

    # Loop over all files
    for filename in files:

        if args.verbose:
            print('Plotting file: {}'.format(os.path.basename(filename)))

        # Read radar data
        radar = read(filename, exclude_fields=None)

        start = time.time()

        # Call desired plotting function
        multipanel(radar, args.outdir, dpi=args.dpi, debug=args.debug,
                   verbose=args.verbose)

        if args.verbose:
            elapsed = time.time() - start
            print('Elapsed time to save plot: {:.0f} sec'.format(elapsed))
