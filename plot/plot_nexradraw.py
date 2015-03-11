#!/usr/bin/python

import os
import argparse
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import num2date
from matplotlib import rcParams, colors
from matplotlib.ticker import MultipleLocator

from pyart.aux_io import read_radx
from pyart.graph import cm

### GLOBAL VARIABLES ###
########################

# Define the proper number of sweeps --> VCP to plot
VCP_SWEEPS = 14

# Define sweeps to plot
SWEEPS = [0, 1, 2, 3, 5, 8, 10, 13]

# Define maximum range in kilometers to plot
MAX_RANGE = 200.0

### Set figure parameters ###
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.size'] = 4
rcParams['xtick.major.width'] = 1
rcParams['xtick.minor.size'] = 2
rcParams['xtick.minor.width'] = 1
rcParams['ytick.major.size'] = 4
rcParams['ytick.major.width'] = 1
rcParams['ytick.minor.size'] = 2
rcParams['ytick.minor.width'] = 1

### Define colormaps and their boundaries ###
cmap_refl = cm.NWSRef
cmap_vdop = plt.get_cmap('jet')
cmap_sw = plt.get_cmap('jet')
cmap_rhohv = plt.get_cmap('jet')
cmap_zdr = plt.get_cmap('jet')
cmap_phidp = plt.get_cmap('jet')
norm_refl = colors.BoundaryNorm(np.arange(-8, 66, 2), cmap_refl.N)
norm_vdop = colors.BoundaryNorm(np.arange(-32, 34, 2), cmap_vdop.N)
norm_sw = colors.BoundaryNorm(np.arange(0, 10.5, 0.5), cmap_sw.N)
norm_rhohv = colors.BoundaryNorm(np.arange(0, 1.1, 0.1), cmap_rhohv.N)
norm_zdr = colors.BoundaryNorm(np.arange(-10, 10.5, 0.5), cmap_zdr.N)
norm_phidp = colors.BoundaryNorm(np.arange(0, 181, 1), cmap_phidp.N)
ticks_refl = np.arange(-8, 72, 8)
ticks_vdop = np.arange(-32, 40, 8)
ticks_sw = np.arange(0, 11, 1)


def pcolormesh(radar, field, sweep=0, cmap=None, norm=None, ax=None):
    """
    """
    if ax is None:
        ax = plt.gca()

    # Parse sweep data
    s0 = radar.sweep_start_ray_index['data'][sweep]
    sf = radar.sweep_end_ray_index['data'][sweep]

    # Parse radar sweep coordinates
    # Convert angle coordinates into radians and range into kilometers
    rng = radar.range['data'] / 1000.0
    azi = np.radians(radar.azimuth['data'][s0:sf+1])

    # Compute index of maximum range
    i = np.abs(rng - MAX_RANGE).argmin()

    # Compute radar sweep coordinates
    # The units for range are converted to kilometers
    AZI, RNG = np.meshgrid(azi, rng[:i+1], indexing='ij')
    X = RNG * np.sin(AZI)
    Y = RNG * np.cos(AZI)

    # Create plot
    qm = ax.pcolormesh(X, Y, radar.fields[field]['data'][s0:sf+1,:i+1],
                       cmap=cmap, norm=norm, shading='flat', alpha=None)

    # Create title
    time_sweep = num2date(
        radar.time['data'][(s0 + sf) / 2], radar.time['units'])
    title = '{} {:.1f} deg {}Z\n {}'.format(
        radar.metadata['instrument_name'], radar.fixed_angle['data'][sweep],
        time_sweep.isoformat(), radar.fields[field]['long_name'])
    ax.set_title(title)

    return qm


def multipanel_V03(inpdir, outdir, stamp, dpi=100, verbose=False):
    """
    """
    files = [os.path.join(inpdir, f) for f in sorted(os.listdir(inpdir))
             if stamp in f]

    if verbose:
        print 'Number of files to plot = %i' % len(files)

    for f in files:

        # Parse input file
        radar = read_radx(f)

        # Check the number of sweeps are consistent with defined VCP
        if radar.nsweeps != VCP_SWEEPS:
            continue

        if verbose:
            print 'Currently plotting file %s' % os.path.basename(f)

        # Create figure instance
        subs = {'xlim': (-MAX_RANGE, MAX_RANGE),
                'ylim': (-MAX_RANGE, MAX_RANGE)}
        figs = {'figsize': (66, 24)}
        fig, ax = plt.subplots(
            nrows=3, ncols=len(SWEEPS), subplot_kw=subs, **figs)

        # Iterate over each sweep
        for j, sweep in enumerate(SWEEPS):

            # (a) Reflectivity
            qma = pcolormesh(
                radar, 'REF', sweep=sweep, cmap=cmap_refl, norm=norm_refl,
                ax=ax[0,j])

            # (b) Doppler velocity
            qmb = pcolormesh(
                radar, 'VEL', sweep=sweep, cmap=cmap_vdop, norm=norm_vdop,
                ax=ax[1,j])

            # (c) Spectrum width
            qmc = pcolormesh(
                radar, 'SW', sweep=sweep, cmap=cmap_sw, norm=norm_sw,
                ax=ax[2,j])

        # Format plot axes
        for i, j in np.ndindex(ax.shape):
            ax[i,j].xaxis.set_major_locator(MultipleLocator(50))
            ax[i,j].xaxis.set_minor_locator(MultipleLocator(10))
            ax[i,j].yaxis.set_major_locator(MultipleLocator(50))
            ax[i,j].yaxis.set_minor_locator(MultipleLocator(10))
            ax[i,j].set_xlabel('Eastward Range from Radar (km)')
            ax[i,j].set_ylabel('Northward Range from Radar (km)')
            ax[i,j].grid(which='major')

        # Color bars
        plt.colorbar(mappable=qma, cax=fig.add_axes([0.91, 0.68, 0.008, 0.2]),
                     ticks=ticks_refl)
        plt.colorbar(mappable=qmb, cax=fig.add_axes([0.91, 0.40, 0.008, 0.2]),
                     ticks=ticks_vdop)
        plt.colorbar(mappable=qmc, cax=fig.add_axes([0.91, 0.12, 0.008, 0.2]),
                     ticks=ticks_sw)

        # Save figure
        date_stamp = num2date(radar.time['data'][:].min(), radar.time['units'])
        filename = '{}.png'.format(date_stamp.strftime('%Y%m%d.%H%M%S'))
        fig.savefig(os.path.join(outdir, filename), format='png', dpi=dpi,
                    bbox_inches='tight')
        plt.close(fig)

    return


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('inpdir', type=str, help=None)
    parser.add_argument('outdir', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('--dpi', nargs='?', type=int, const=50, default=50,
                        help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, const=True,
                        default=False, help=None)
    parser.add_argument('--debug', nargs='?', type=bool, const=True,
                        default=False, help=None)
    args = parser.parse_args()

    if args.debug:
        print 'source = %s' % args.inpdir
        print 'output = %s' % args.outdir
        print 'stamp = %s' % args.stamp
        print 'dpi = %i' % args.dpi

    if args.verbose:
        print 'VCP_SWEEPS = %i' % VCP_SWEEPS
        print 'MAX_RANGE = %.2f' % MAX_RANGE

    # Call desired plotting function
    multipanel_V03(args.inpdir, args.outdir, args.stamp, dpi=args.dpi,
                   verbose=args.verbose)
