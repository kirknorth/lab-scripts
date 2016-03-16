#!/usr/bin/python

import os
import time
import argparse
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
HEIGHTS = [0, 2, 4, 12, 16, 20]

# Define radar fields
REFL_FIELD = get_field_name('reflectivity')
VDOP_FIELD = get_field_name('velocity')
VDOP_CORR_FIELD = get_field_name('corrected_velocity')
SPW_FIELD = get_field_name('spectrum_width')
RHOHV_FIELD = get_field_name('cross_correlation_ratio')
ZDR_FIELD = get_field_name('differential_reflectivity')
PHIDP_FIELD = get_field_name('differential_phase')
DIST_FIELD = get_field_name('nearest_neighbor_distance')
TIME_FIELD = get_field_name('nearest_neighbor_time')
GQI_FIELD = get_field_name('grid_quality_index')

# Define colour maps
CMAP_REFL = cm.NWSRef
CMAP_VDOP = cm.NWSVel
CMAP_SPW = cm.NWS_SPW
CMAP_RHOHV = cm.Carbone17
CMAP_ZDR = cm.RefDiff
CMAP_PHIDP = cm.Wild25
CMAP_GQI = cm.Carbone17
CMAP_DIST = cm.BlueBrown10

# Normalize colour maps
NORM_REFL = BoundaryNorm(np.arange(-10, 65, 5), CMAP_REFL.N)
NORM_VDOP = BoundaryNorm(np.arange(-30, 32, 2), CMAP_VDOP.N)
NORM_SPW = BoundaryNorm(np.arange(0, 8.5, 0.5), CMAP_SPW.N)
NORM_RHOHV = BoundaryNorm(np.arange(0, 1.05, 0.05), CMAP_RHOHV.N)
NORM_ZDR = BoundaryNorm(np.arange(-6, 6.5, 0.5), CMAP_ZDR.N)
NORM_PHIDP = BoundaryNorm(np.arange(0, 365, 5), CMAP_PHIDP.N)
NORM_GQI = BoundaryNorm(np.arange(0, 1.05, 0.05), CMAP_GQI.N)
NORM_DIST = BoundaryNorm(np.arange(0, 2.1, 0.1), CMAP_DIST.N)

# Define colour bar ticks
TICKS_REFL = np.arange(-10, 70, 10)
TICKS_VDOP = np.arange(-30, 40, 10)
TICKS_SPW = np.arange(0, 9, 1)
TICKS_RHOHV = np.arange(0, 1.2, 0.2)
TICKS_ZDR = np.arange(-6, 8, 2)
TICKS_PHIDP = np.arange(0, 450, 90)
TICKS_GQI = np.arange(0, 1.2, 0.2)
TICKS_DIST = np.arange(0, 2.4, 0.4)
TICKS = [
    TICKS_REFL,
    TICKS_VDOP,
    TICKS_VDOP,
    TICKS_SPW,
    TICKS_RHOHV,
    TICKS_ZDR,
    TICKS_PHIDP,
    TICKS_GQI,
    TICKS_DIST,
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
rcParams['figure.figsize'] = (36, 24)


def plot_file(filename, outdir, dpi=90, debug=False, verbose=False):
    """
    """

    if verbose:
        print 'Plotting file: {}'.format(os.path.basename(filename))

    # Read gridded data
    grid = Dataset(filename, mode='r')

    # Initialize figure and axes
    subs = {'xlim': (-50, 50), 'ylim': (-25, 100)}
    fig, axes = plt.subplots(nrows=len(HEIGHTS), ncols=9, subplot_kw=subs)
    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    # Loop over all heights
    for k, height in enumerate(HEIGHTS):

        if verbose:
            print 'Plotting height index: {}'.format(height)

        # (a) Reflectivity
        qma = _plot_cappi(
            grid, REFL_FIELD, height=height, cmap=CMAP_REFL, norm=NORM_REFL,
            fig=fig, ax=axes[k,0])

        # (b) Uncorrected Doppler velocity
        qmb = _plot_cappi(
            grid, VDOP_FIELD, height=height, cmap=CMAP_VDOP, norm=NORM_VDOP,
            fig=fig, ax=axes[k,1])

        # (c) Corrected Doppler velocity
        qmc = _plot_cappi(
            grid, VDOP_CORR_FIELD, height=height, cmap=CMAP_VDOP,
            norm=NORM_VDOP, fig=fig, ax=axes[k,2])

        # (d) Spectrum width
        qmd = _plot_cappi(
            grid, SPW_FIELD, height=height, cmap=CMAP_SPW, norm=NORM_SPW,
            fig=fig, ax=axes[k,3])

        # (e) Copolar correlation ratio
        qme = _plot_cappi(
            grid, RHOHV_FIELD, height=height, cmap=CMAP_RHOHV, norm=NORM_RHOHV,
            fig=fig, ax=axes[k,4])

        # (f) Differential reflectivity
        qmf = _plot_cappi(
            grid, ZDR_FIELD, height=height, cmap=CMAP_ZDR, norm=NORM_ZDR,
            fig=fig, ax=axes[k,5])

        # (g) Differential phase
        qmg = _plot_cappi(
            grid, PHIDP_FIELD, height=height, cmap=CMAP_PHIDP, norm=NORM_PHIDP,
            fig=fig, ax=axes[k,6])

        # (h) Grid quality index
        qmh = _plot_cappi(
            grid, GQI_FIELD, height=height, cmap=CMAP_GQI, norm=NORM_GQI,
            fig=fig, ax=axes[k,7])

        # (i) Nearest neighbour distance
        qmi = _plot_cappi(
            grid, DIST_FIELD, height=height, scale=1.0e-3, cmap=CMAP_DIST,
            norm=NORM_DIST, fig=fig, ax=axes[k,8])

    # Create colour bars
    qm = [qma, qmb, qmc, qmd, qme, qmf, qmg, qmh, qmi]
    _create_colorbars(fig, axes, qm, TICKS)

    # Format axes
    for ax in axes.flat:
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.set_xlabel('Eastward Distance (km)')
        ax.set_ylabel('Northward Distance (km)')
        ax.grid(which='major')

    # Define image file name
    date_stamp = _datetimes(grid).min().strftime('%Y%m%d.%H%M%S')
    fname = '{}.png'.format(date_stamp)

    # Save figure
    fig.savefig(os.path.join(outdir, fname), format='png', dpi=dpi,
                bbox_inches='tight')

    # Close figure and axes to free memory
    plt.cla()
    plt.clf()
    plt.close(fig)
    grid.close()

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

    # Parse grid coordinates
    x = grid.variables['x'][:] / 1000.0
    y = grid.variables['y'][:] / 1000.0
    z = grid.variables['z'][:] / 1000.0

    # Parse grid data
    data = scale * grid.variables[field][0,height,:,:]

    # Create quadmesh
    qm = ax.pcolormesh(
        x, y, data, cmap=cmap, norm=norm, shading='flat', alpha=None,
        rasterized=True)

    # Create title
    time = grid.variables[TIME_FIELD]
    cappi_time = num2date(np.median(time[0,height,:,:]), time.units)
    title = '{} {:.1f} km\n{}Z\n{}'.format(
        grid.radar_0_instrument_name, z[height],
        cappi_time.replace(microsecond=0).isoformat(), field)
    ax.set_title(title)

    return qm


def _create_colorbars(fig, axes, qm, ticks):
    """
    """

    for i in range(axes.shape[1]):
        parents = [ax for ax in axes[:,i].flat]
        cax, _ = make_axes(parents, location='bottom', pad=0.03, shrink=1.0,
                           fraction=0.01, aspect=20)
        fig.colorbar(mappable=qm[i], cax=cax, orientation='horizontal',
                     ticks=ticks[i], drawedges=False, spacing='uniform')

    return


def _datetimes(grid):
    """
    """
    return num2date(grid.variables['time'][:], grid.variables['time'].units)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('inpdir', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('outdir', type=str, help=None)
    parser.add_argument('--dpi', nargs='?', type=int, default=90, const=90,
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

    for filename in files:

        # Record start time
        start = time.time()

        # Call desired plotting routine
        plot_file(filename, args.outdir, dpi=args.dpi, debug=args.debug,
                  verbose=args.verbose)

        if args.verbose:
            elapsed = time.time() - start
            print('Elapsed time to save plot: {:.0f} sec'.format(elapsed))


