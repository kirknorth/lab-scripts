#!/usr/bin/python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import num2date
from matplotlib import rcParams
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import make_axes
from matplotlib.ticker import MultipleLocator

from pyart.graph import cm
from pyart.io import read_mdv
from pyart.config import get_field_name

# Define sweeps to be plotted
SWEEPS = [0, 1, 2]

# Define field names
REFL_FIELD = get_field_name('reflectivity')
VDOP_FIELD = get_field_name('velocity')
WIDTH_FIELD = get_field_name('spectrum_width')
RHOHV_FIELD = get_field_name('cross_correlation_ratio')
ZDR_FIELD = get_field_name('differential_reflectivity')
PHIDP_FIELD = get_field_name('differential_phase')
NCP_FIELD = get_field_name('normalized_coherent_power')

# Define fields to exclude from radar object
EXCLUDE_FIELDS = [
    'corrected_reflectivity',
    'radar_echo_classification',
    'corrected_differential_reflectivity'
    ]

# Define colour maps
CMAP_REFL = cm.NWSRef
CMAP_VDOP = plt.get_cmap('jet')
CMAP_WIDTH = plt.get_cmap('jet')
CMAP_RHOHV = plt.get_cmap('jet')
CMAP_ZDR = plt.get_cmap('jet')
CMAP_PHIDP = plt.get_cmap('jet')
CMAP_NCP = plt.get_cmap('jet')

# Normalize colour maps
NORM_REFL = BoundaryNorm(np.arange(-10, 51, 1), CMAP_REFL.N)
NORM_VDOP = BoundaryNorm(np.arange(-16, 17, 1), CMAP_VDOP.N)
NORM_WIDTH = BoundaryNorm(np.arange(0, 8.5, 0.5), CMAP_WIDTH.N)
NORM_RHOHV = BoundaryNorm(np.arange(0, 1.1, 0.1), CMAP_RHOHV.N)
NORM_ZDR = BoundaryNorm(np.arange(-5, 5.1, 0.1), CMAP_ZDR.N)
NORM_PHIDP = BoundaryNorm(np.arange(-180, 1, 1), CMAP_PHIDP.N)
NORM_NCP = BoundaryNorm(np.arange(0, 1.05, 0.05), CMAP_NCP.N)

# Define colour bar ticks
TICKS_REFL = np.arange(-10, 60, 10)
TICKS_VDOP = np.arange(-16, 20, 4)
TICKS_WIDTH = np.arange(0, 9, 1)
TICKS_RHOHV = np.arange(0, 1.1, 0.1)
TICKS_ZDR = np.arange(-5, 6, 1)
TICKS_PHIDP = np.arange(-180, 20, 20)
TICKS_NCP = np.arange(0, 1.1, 0.1)


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

    # Create figure instance
    subs = {'xlim': (-117, 117), 'ylim': (-117, 117)}
    figs = {'figsize': (55, 25)}
    fig, ax = plt.subplots(nrows=len(SWEEPS), ncols=7, subplot_kw=subs, **figs)

    if debug:
        print 'Number of sweeps: {}'.format(radar.nsweeps)

    # Loop over specified sweeps
    for i, sweep in enumerate(SWEEPS):

        if verbose:
            print 'Plotting sweep index: {}'.format(sweep)

        # (a) Reflectivity
        qma = _pcolormesh(
            radar, REFL_FIELD, sweep=sweep, cmap=CMAP_REFL, norm=NORM_REFL,
            fig=fig, ax=ax[i,0])

        # (b) Doppler velocity
        qmb = _pcolormesh(
            radar, VDOP_FIELD, sweep=sweep, cmap=CMAP_VDOP, norm=NORM_VDOP,
            fig=fig, ax=ax[i,1])

        # (c) Spectrum width
        qmc = _pcolormesh(
            radar, WIDTH_FIELD, sweep=sweep, cmap=CMAP_WIDTH, norm=NORM_WIDTH,
            fig=fig, ax=ax[i,2])

        # (d) Copolar correlation coefficient
        qmd = _pcolormesh(
            radar, RHOHV_FIELD, sweep=sweep, cmap=CMAP_RHOHV, norm=NORM_RHOHV,
            fig=fig, ax=ax[i,3])

        # (e) Differential reflectivity
        qme = _pcolormesh(
            radar, ZDR_FIELD, sweep=sweep, cmap=CMAP_ZDR, norm=NORM_ZDR,
            fig=fig, ax=ax[i,4])

        # (f) Differential phase
        qmf = _pcolormesh(
            radar, PHIDP_FIELD, sweep=sweep, cmap=CMAP_PHIDP, norm=NORM_PHIDP,
            fig=fig, ax=ax[i,5])

        # (g) Normalized coherent power
        qmg = _pcolormesh(
            radar, NCP_FIELD, sweep=sweep, cmap=CMAP_NCP, norm=NORM_NCP,
            fig=fig, ax=ax[i,6])

    # Format plot axes
    for i, j in np.ndindex(ax.shape):
        ax[i,j].xaxis.set_major_locator(MultipleLocator(20))
        ax[i,j].xaxis.set_minor_locator(MultipleLocator(5))
        ax[i,j].yaxis.set_major_locator(MultipleLocator(20))
        ax[i,j].yaxis.set_minor_locator(MultipleLocator(5))
        ax[i,j].set_xlabel('Eastward Range from Radar (km)')
        ax[i,j].set_ylabel('Northward Range from Radar (km)')
        ax[i,j].grid(which='major')

    # Create color bars
    cax = []
    for i in range(ax.shape[1]):
        cax.append(
            make_axes([axis for axis in ax[:,i].flat], location='bottom',
                      pad=0.04, fraction=0.01, shrink=1.0, aspect=20))
    fig.colorbar(mappable=qma, cax=cax[0][0], orientation='horizontal',
                 ticks=TICKS_REFL)
    fig.colorbar(mappable=qmb, cax=cax[1][0], orientation='horizontal',
                 ticks=TICKS_VDOP)
    fig.colorbar(mappable=qmc, cax=cax[2][0], orientation='horizontal',
                 ticks=TICKS_WIDTH)
    fig.colorbar(mappable=qmd, cax=cax[3][0], orientation='horizontal',
                 ticks=TICKS_RHOHV)
    fig.colorbar(mappable=qme, cax=cax[4][0], orientation='horizontal',
                 ticks=TICKS_ZDR)
    fig.colorbar(mappable=qmf, cax=cax[5][0], orientation='horizontal',
                 ticks=TICKS_PHIDP)
    fig.colorbar(mappable=qmg, cax=cax[6][0], orientation='horizontal',
                 ticks=TICKS_NCP)

    # Define image file name
    date_stamp = _datetimes(radar).min().strftime('%Y%m%d.%H%M%S')
    fname = '{}.png'.format(date_stamp)

    # Save figure
    fig.savefig(os.path.join(outdir, fname), format='png', dpi=dpi,
                bbox_inches='tight')

    # Close figure to free memory
    plt.close(fig)

    return


def _pcolormesh(
        radar, field, sweep=0, cmap=None, norm=None, fig=None, ax=None):
    """
    """

    # Parse figure and axis
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    # Parse colour map
    if cmap is None:
        cmap = plt.get_cmap('jet')

    # Parse radar coordinates
    # Convert angles to radians and range to kilometers
    rng = radar.range['data'] / 1000.0
    azi = np.radians(radar.get_azimuth(sweep))

    # Compute radar sweep coordinates
    AZI, RNG = np.meshgrid(azi, rng, indexing='ij')
    X = RNG * np.sin(AZI)
    Y = RNG * np.cos(AZI)

    # Parse radar data
    data = radar.get_field(sweep, field)

    # Create quadmesh
    qm = ax.pcolormesh(
        X, Y, data, cmap=cmap, norm=norm, shading='flat', alpha=None)

    # Create title
    s0, sn = radar.get_start_end(sweep)
    sweep_time = _datetimes(radar)[(s0 + sn) / 2]
    title = '{} {:.1f} deg {}Z\n{}'.format(
            radar.metadata['instrument_name'],
            radar.fixed_angle['data'][sweep],
            sweep_time.isoformat(), field)
    ax.set_title(title)

    return qm


def _datetimes(radar):
    """
    """

    return num2date(radar.time['data'], radar.time['units'])


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
        radar = read_mdv(filename, exclude_fields=EXCLUDE_FIELDS)

        # Call desired plotting function
        multipanel(radar, args.outdir, dpi=args.dpi, debug=args.debug,
                   verbose=args.verbose)
