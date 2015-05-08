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
from pyart.aux_io import read_radx
from pyart.config import get_field_name

# Define sweeps to plot
SWEEPS = [0, 1, 2]

# Define max range to plot
MAX_RANGE = 180.0

# Define radar fields
REFL_FIELD = 'REF'
VDOP_FIELD = 'VEL'
WIDTH_FIELD = 'SW'
RHOHV_FIELD = 'RHO'
ZDR_FIELD = 'ZDR'
PHIDP_FIELD = 'PHI'

# Define colour maps
CMAP_REFL = cm.NWSRef
CMAP_VDOP = plt.get_cmap(name='jet')
CMAP_WIDTH = plt.get_cmap(name='jet')
CMAP_RHOHV = plt.get_cmap(name='jet')
CMAP_ZDR = plt.get_cmap(name='jet')
CMAP_PHIDP = plt.get_cmap(name='jet')

# Normalize colour maps
NORM_REFL = BoundaryNorm(np.arange(-5, 66, 1), CMAP_REFL.N)
NORM_VDOP = BoundaryNorm(np.arange(-25, 26, 1), CMAP_VDOP.N)
NORM_WIDTH = BoundaryNorm(np.arange(0, 8.1, 0.1), CMAP_WIDTH.N)
NORM_RHOHV = BoundaryNorm(np.arange(0, 1.01, 0.01), CMAP_RHOHV.N)
NORM_ZDR = BoundaryNorm(np.arange(-5, 5.1, 0.1), CMAP_ZDR.N)
NORM_PHIDP = BoundaryNorm(np.arange(0, 181, 1), CMAP_PHIDP.N)

# Define colour bar ticks
TICKS_REFL = np.arange(-5, 75, 10)
TICKS_VDOP = np.arange(-25, 30, 5)
TICKS_WIDTH = np.arange(0, 9, 1)
TICKS_RHOHV = np.arange(0, 1.1, 0.1)
TICKS_ZDR = np.arange(-5, 6, 1)
TICKS_PHIDP = np.arange(0, 200, 20)


def multipanel(radar, outdir, dpi=100, debug=False, verbose=False):
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
    subs = {'xlim': (-MAX_RANGE, MAX_RANGE),
            'ylim': (-MAX_RANGE, MAX_RANGE)}
    figs = {'figsize': (50, 25)}
    fig, ax = plt.subplots(nrows=len(SWEEPS), ncols=6, subplot_kw=subs, **figs)

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

        # (d) Copolar Correlation coefficient
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

    # Create colour bars
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

    # Format plot axes
    for i, j in np.ndindex(ax.shape):
        ax[i,j].xaxis.set_major_locator(MultipleLocator(40))
        ax[i,j].xaxis.set_minor_locator(MultipleLocator(10))
        ax[i,j].yaxis.set_major_locator(MultipleLocator(40))
        ax[i,j].yaxis.set_minor_locator(MultipleLocator(10))
        ax[i,j].set_xlabel('Eastward Range from Radar (km)')
        ax[i,j].set_ylabel('Northward Range from Radar (km)')
        ax[i,j].grid(which='major')

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

    # Parse axis and figure parameters
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    # Parse colour map
    if cmap is None:
        cmap = plt.get_cmap(name='jet')

    # Parse radar coordinates
    # Convert angles to radians and range to kilometers
    rng = radar.range['data'] / 1000.0
    azi = np.radians(radar.get_azimuth(sweep))

    # Compute index of maximum range
    rf = np.abs(rng - MAX_RANGE).argmin() + 1

    # Compute radar sweep coordinates
    AZI, RNG = np.meshgrid(azi, rng[:rf], indexing='ij')
    X = RNG * np.sin(AZI)
    Y = RNG * np.cos(AZI)

    # Parse radar data
    data = radar.get_field(sweep, field)[:,:rf]

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

    if args.debug:
        print 'MAX_RANGE -> {}'.format(MAX_RANGE)

    # Parse radar files to plot
    files = [os.path.join(args.inpdir, f) for f
             in sorted(os.listdir(args.inpdir)) if args.stamp in f]

    if args.verbose:
        print 'Number of files to plot: {}'.format(len(files))

    # Loop over all files
    for filename in files:

        if args.verbose:
            print 'Plotting file: {}'.format(os.path.basename(filename))

        # Read radar data
        radar = read_radx(filename)

        # Call desired plotting function
        multipanel(radar, args.outdir, dpi=args.dpi, debug=args.debug,
                   verbose=args.verbose)
