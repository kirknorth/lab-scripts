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

# Define sweeps to plot
SWEEPS = [0, 1, 2, 3, 5, 8, 10, 13]

# Define max range to plot
MAX_RANGE = 180.0

# Define radar fields
REFL_FIELD = get_field_name('reflectivity')
VDOP_FIELD = get_field_name('velocity')
CORR_VDOP_FIELD = get_field_name('corrected_velocity')
WIDTH_FIELD = get_field_name('spectrum_width')
RHOHV_FIELD = get_field_name('cross_correlation_ratio')
ZDR_FIELD = get_field_name('differential_reflectivity')
PHIDP_FIELD = get_field_name('differential_phase')

# Define colour maps
CMAP_REFL = cm.NWSRef
CMAP_VDOP = plt.get_cmap(name='jet')
CMAP_WIDTH = plt.get_cmap(name='jet')
CMAP_RHOHV = plt.get_cmap(name='jet')
CMAP_ZDR = plt.get_cmap(name='jet')
CMAP_PHIDP = plt.get_cmap(name='jet')

# Normalize colour maps
NORM_REFL = BoundaryNorm(np.arange(-10, 62, 2), CMAP_REFL.N)
NORM_VDOP = BoundaryNorm(np.arange(-32, 34, 2), CMAP_VDOP.N)
NORM_WIDTH = BoundaryNorm(np.arange(0, 8.1, 0.1), CMAP_WIDTH.N)
NORM_RHOHV = BoundaryNorm(np.arange(0.7, 1.01, 0.01), CMAP_RHOHV.N)
NORM_ZDR = BoundaryNorm(np.arange(-2, 3.1, 0.1), CMAP_ZDR.N)
NORM_PHIDP = BoundaryNorm(np.arange(0, 181, 1), CMAP_PHIDP.N)

# Define colour bar ticks
TICKS_REFL = np.arange(-10, 70, 10)
TICKS_VDOP = np.arange(-32, 40, 8)
TICKS_WIDTH = np.arange(0, 9, 1)
TICKS_RHOHV = np.arange(0.7, 1.05, 0.05)
TICKS_ZDR = np.arange(-2, 3.5, 0.5)
TICKS_PHIDP = np.arange(0, 200, 20)


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

    # Create figure instance
    subs = {'xlim': (-MAX_RANGE, MAX_RANGE),
            'ylim': (-MAX_RANGE, MAX_RANGE)}
    figs = {'figsize': (59, 49)}
    fig, ax = plt.subplots(nrows=7, ncols=len(SWEEPS), subplot_kw=subs, **figs)

    # Iterate over each sweep
    for j, sweep in enumerate(SWEEPS):

        if verbose:
            print 'Plotting sweep: {}'.format(sweep)

        # (a) Reflectivity
        qma = _pcolormesh(
            radar, REFL_FIELD, sweep=sweep, cmap=CMAP_REFL, norm=NORM_REFL,
            ticks=TICKS_REFL, fig=fig, ax=ax[0,j])

        # (b) Uncorrected Doppler velocity
        qmb = _pcolormesh(
            radar, VDOP_FIELD, sweep=sweep, cmap=CMAP_VDOP, norm=NORM_VDOP,
            ticks=TICKS_VDOP, fig=fig, ax=ax[1,j])

        # (c) Corrected Doppler velocity
        qmc = _pcolormesh(
            radar, CORR_VDOP_FIELD, sweep=sweep, cmap=CMAP_VDOP,
            norm=NORM_VDOP, ticks=TICKS_VDOP, fig=fig, ax=ax[2,j])

        # (d) Spectrum width
        qmd = _pcolormesh(
            radar, WIDTH_FIELD, sweep=sweep, cmap=CMAP_WIDTH, norm=NORM_WIDTH,
            ticks=TICKS_WIDTH, fig=fig, ax=ax[3,j])

        # (e) Copolar Correlation coefficient
        qme = _pcolormesh(
            radar, RHOHV_FIELD, sweep=sweep, cmap=CMAP_RHOHV, norm=NORM_RHOHV,
            ticks=TICKS_RHOHV, fig=fig, ax=ax[4,j])

        # (f) Differential reflectivity
        qmf = _pcolormesh(
            radar, ZDR_FIELD, sweep=sweep, cmap=CMAP_ZDR, norm=NORM_ZDR,
            ticks=TICKS_ZDR, fig=fig, ax=ax[5,j])

        # (g) Differential phase
        qmg = _pcolormesh(
            radar, PHIDP_FIELD, sweep=sweep, cmap=CMAP_PHIDP, norm=NORM_PHIDP,
            ticks=TICKS_PHIDP, fig=fig, ax=ax[6,j])

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

    # Format plot axes
    for i, j in np.ndindex(ax.shape):
        ax[i,j].xaxis.set_major_locator(MultipleLocator(40))
        ax[i,j].xaxis.set_minor_locator(MultipleLocator(10))
        ax[i,j].yaxis.set_major_locator(MultipleLocator(40))
        ax[i,j].yaxis.set_minor_locator(MultipleLocator(10))
        ax[i,j].set_xlabel('Eastward Range from Radar (km)')
        ax[i,j].set_ylabel('Northward Range from Radar (km)')
        ax[i,j].grid(which='major')

    # Save figure
    date_stamp = _datetimes(radar).min().strftime('%Y%m%d.%H%M%S')
    filename = '{}.png'.format(date_stamp)
    fig.savefig(os.path.join(outdir, filename), format='png', dpi=dpi,
                bbox_inches='tight')
    plt.close(fig)

    return


def _pcolormesh(radar, field, sweep=0, cmap=None, norm=None, ticks=None,
                fig=None, ax=None):
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

    # Parse sweep data
    s0 = radar.variables['sweep_start_ray_index'][sweep]
    sn = radar.variables['sweep_end_ray_index'][sweep]

    # Parse radar coordinates
    # Convert angles to radians and range to kilometers
    rng = radar.variables['range'][:] / 1000.0
    azi = np.radians(radar.variables['azimuth'][s0:sn+1])

    # Compute index of maximum range
    rn = np.abs(rng - MAX_RANGE).argmin()

    # Compute radar sweep coordinates
    AZI, RNG = np.meshgrid(azi, rng[:rn+1], indexing='ij')
    X = RNG * np.sin(AZI)
    Y = RNG * np.cos(AZI)

    # Parse radar data
    data = radar.variables[field][s0:sn+1,:rn+1]

    # Create quadmesh
    qm = ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading='flat')

    # Create title
    time_sweep = _datetimes(radar)[(s0 + sn) / 2]
    title = '{} {:.1f} deg {}Z\n{}'.format(
            radar.instrument_name, radar.variables['fixed_angle'][sweep],
            time_sweep.isoformat(), radar.variables[field].long_name)
    ax.set_title(title)

    return qm


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

    if args.debug:
        print 'MAX_RANGE -> {}'.format(MAX_RANGE)

    # Parse radar files to plot
    files = [os.path.join(args.inpdir, f) for f
             in sorted(os.listdir(args.inpdir)) if args.stamp in f]

    if args.verbose:
        print 'Number of files to plot: {}'.format(len(files))

    # Loop over all files
    for f in files:

        if args.verbose:
            print 'Plotting file: {}'.format(os.path.basename(f))

        # Read radar data
        radar = Dataset(f, mode='r')

        # Call desired plotting function
        multipanel(radar, args.outdir, dpi=args.dpi, verbose=args.verbose)
