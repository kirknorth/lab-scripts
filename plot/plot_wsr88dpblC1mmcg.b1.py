#!/usr/bin/python

import os
import argparse
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset, num2date
from matplotlib import rcParams
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import make_axes
from matplotlib.ticker import MultipleLocator

from pyart.graph import cm
from pyart.config import get_field_name

# Define heights to plot
HEIGHTS = [0, 5, 10, 15, 20, 25, 30, 40]

# Define radar fields
REFL_FIELD = get_field_name('reflectivity')
VDOP_FIELD = get_field_name('velocity')
CORR_VDOP_FIELD = get_field_name('corrected_velocity')
CORR_VDOP_FIELD = get_field_name('corrected_velocity')
WIDTH_FIELD = get_field_name('spectrum_width')
RHOHV_FIELD = get_field_name('cross_correlation_ratio')
ZDR_FIELD = get_field_name('differential_reflectivity')
PHIDP_FIELD = get_field_name('differential_phase')
DIST_FIELD = 'nearest_neighbor_distance'

# Define colour maps
CMAP_REFL = cm.NWSRef
CMAP_VDOP = plt.get_cmap(name='jet')
CMAP_WIDTH = plt.get_cmap(name='jet')
CMAP_RHOHV = plt.get_cmap(name='jet')
CMAP_ZDR = plt.get_cmap(name='jet')
CMAP_PHIDP = plt.get_cmap(name='jet')
CMAP_DIST = plt.get_cmap(name='jet')

# Normalize colour maps
NORM_REFL = BoundaryNorm(np.arange(-10, 41, 1), CMAP_REFL.N)
NORM_VDOP = BoundaryNorm(np.arange(-25, 26, 1), CMAP_VDOP.N)
NORM_WIDTH = BoundaryNorm(np.arange(0, 8.1, 0.1), CMAP_WIDTH.N)
NORM_RHOHV = BoundaryNorm(np.arange(0.5, 1.01, 0.01), CMAP_RHOHV.N)
NORM_ZDR = BoundaryNorm(np.arange(0, 5.1, 0.1), CMAP_ZDR.N)
NORM_PHIDP = BoundaryNorm(np.arange(0, 181, 1), CMAP_PHIDP.N)
NORM_DIST = BoundaryNorm(np.arange(0, 2.05, 0.05), CMAP_DIST.N)

# Define colour bar ticks
TICKS_REFL = np.arange(-10, 45, 5)
TICKS_VDOP = np.arange(-25, 30, 5)
TICKS_WIDTH = np.arange(0, 9, 1)
TICKS_RHOHV = np.arange(0.5, 1.1, 0.1)
TICKS_ZDR = np.arange(0, 5.5, 0.5)
TICKS_PHIDP = np.arange(0, 200, 20)
TICKS_DIST = np.arange(0, 2.2, 0.2)


def plot_file(filename, outdir, dpi=50, debug=False, verbose=False):
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

    if verbose:
        print 'Plotting file: {}'.format(os.path.basename(filename))

    # Read gridded data
    grid = Dataset(filename, mode='r')

    # Initialize figure and axes
    subs = {'xlim': (-20, 20), 'ylim': (-20, 20)}
    figs = {'figsize': (59, 51)}
    fig, ax = plt.subplots(nrows=8, ncols=8, subplot_kw=subs, **figs)

    # Loop over all heights
    for k, height in enumerate(HEIGHTS):

        if verbose:
            print 'Plotting height index: {}'.format(height)

        # (a) Reflectivity
        qma = _plot_cappi(
            grid, REFL_FIELD, height=height, cmap=CMAP_REFL, norm=NORM_REFL,
            fig=fig, ax=ax[0,k])

        # (b) Uncorrected Doppler velocity
        qmb = _plot_cappi(
            grid, VDOP_FIELD, height=height, cmap=CMAP_VDOP, norm=NORM_VDOP,
            fig=fig, ax=ax[1,k])

        # (c) Corrected Doppler velocity
        qmc = _plot_cappi(
            grid, CORR_VDOP_FIELD, height=height, cmap=CMAP_VDOP,
            norm=NORM_VDOP, fig=fig, ax=ax[2,k])

        # (d) Spectrum width
        qmd = _plot_cappi(
            grid, WIDTH_FIELD, height=height, cmap=CMAP_WIDTH, norm=NORM_WIDTH,
            fig=fig, ax=ax[3,k])

        # (e) Copolar correlation ratio
        qme = _plot_cappi(
            grid, RHOHV_FIELD, height=height, cmap=CMAP_RHOHV, norm=NORM_RHOHV,
            fig=fig, ax=ax[4,k])

        # (f) Differential reflectivity
        qmf = _plot_cappi(
            grid, ZDR_FIELD, height=height, cmap=CMAP_ZDR, norm=NORM_ZDR,
            fig=fig, ax=ax[5,k])

        # (g) Differential phase
        qmg = _plot_cappi(
            grid, PHIDP_FIELD, height=height, cmap=CMAP_PHIDP, norm=NORM_PHIDP,
            fig=fig, ax=ax[6,k])

        # (h) Nearest neighbour distance
        qmh = _plot_cappi(
            grid, DIST_FIELD, height=height, scale=1.0e-3, cmap=CMAP_DIST,
            norm=NORM_DIST, fig=fig, ax=ax[7,k])

    # Create colour bars
    cax = []
    for i in range(ax.shape[0]):
        cax.append(
            make_axes([axis for axis in ax[i].flat], location='right',
                      pad=0.01, fraction=0.01, shrink=1.0, aspect=20))
    fig.colorbar(mappable=qma, cax=cax[0][0], ticks=TICKS_REFL)
    fig.colorbar(mappable=qmb, cax=cax[1][0], ticks=TICKS_VDOP)
    fig.colorbar(mappable=qmc, cax=cax[2][0], ticks=TICKS_VDOP)
    fig.colorbar(mappable=qmd, cax=cax[3][0], ticks=TICKS_WIDTH)
    fig.colorbar(mappable=qme, cax=cax[4][0], ticks=TICKS_RHOHV)
    fig.colorbar(mappable=qmf, cax=cax[5][0], ticks=TICKS_ZDR)
    fig.colorbar(mappable=qmg, cax=cax[6][0], ticks=TICKS_PHIDP)
    fig.colorbar(mappable=qmh, cax=cax[7][0], ticks=TICKS_DIST)

    # Format axes
    for i, j in np.ndindex(ax.shape):
        ax[i,j].xaxis.set_major_locator(MultipleLocator(4))
        ax[i,j].xaxis.set_minor_locator(MultipleLocator(1))
        ax[i,j].yaxis.set_major_locator(MultipleLocator(4))
        ax[i,j].yaxis.set_minor_locator(MultipleLocator(1))
        ax[i,j].grid(which='major')

    # Define image file name
    date = num2date(grid.variables['time'][:].min(),
                    grid.variables['time'].units)
    fname = '{}.png'.format(date.strftime('%Y%m%d.%H%M%S'))

    # Save figure
    fig.savefig(os.path.join(outdir, fname), format='png', dpi=dpi,
                bbox_inches='tight')

    # Close figure to release memory
    plt.close(fig)

    return


def _plot_cappi(grid, field, height=0, scale=1.0, cmap=None, norm=None,
                fig=None, ax=None):
    """
    """
    # Parse figure and axis
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    # Parse colour map
    if cmap is None:
        cmap = 'jet'

    # Parse grid coordinates
    x_disp = grid.variables['x_disp'][:] / 1000.0
    y_disp = grid.variables['y_disp'][:] / 1000.0
    z_disp = grid.variables['z_disp'][:] / 1000.0

    # Create quadmesh
    qm = ax.pcolormesh(
        x_disp, y_disp, scale * grid.variables[field][0,height,:,:], cmap=cmap,
        norm=norm, shading='flat', alpha=None)

    # Set title
    title = '{} {:.1f} km\n{}'.format(
        grid.radar_0_instrument_name, z_disp[height],
        grid.variables[field].long_name)
    ax.set_title(title)

    return qm


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('inpdir', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('outdir', type=str, help=None)
    parser.add_argument('--dpi', nargs='?', type=int, default=50, const=50,
                        help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, default=False,
                        const=True, help=None)
    parser.add_argument('-db', '--debug', nargs='?', type=bool, default=False,
                        const=True, help=None)
    args = parser.parse_args()

    # Parse all files to plot
    files = [os.path.join(args.inpdir, f) for f in
             sorted(os.listdir(args.inpdir)) if args.stamp in f]

    if args.verbose:
        print 'Number of files to plot: {}'.format(len(files))

    # Loop over all files
    for filename in files:

        # Call desired plotting routine
        plot_file(filename, args.outdir, dpi=args.dpi, debug=args.debug,
                  verbose=args.verbose)


