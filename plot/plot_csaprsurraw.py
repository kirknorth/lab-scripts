#!/usr/bin/python

import os
import argparse
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset, num2date
from matplotlib import rcParams, colors
from matplotlib.ticker import MultipleLocator

from pyart.graph import cm
from pyart.io import read_mdv

### GLOBAL VARIABLES ###
########################

# Define the proper number of sweeps --> VCP to plot
VCP_SWEEPS = 17

# Define sweeps to be plotted
SWEEPS = [0, 1, 2, 5, 8, 10, 13, 16]

# Define fields to exclude from radar object
EXCLUDE_FIELDS = ['corrected_reflectivity', 'radar_echo_classification',
                  'corrected_differential_reflectivity']


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

### Define color maps ###
cmap_refl = cm.NWSRef
cmap_vdop = plt.get_cmap('jet')
cmap_ncp = plt.get_cmap('jet')
cmap_rhv = plt.get_cmap('jet')
cmap_sw = plt.get_cmap('jet')
cmap_phi = plt.get_cmap('jet')
norm_refl = colors.BoundaryNorm(np.arange(-8, 66, 2), cmap_refl.N)
norm_vdop = colors.BoundaryNorm(np.arange(-16, 17, 1), cmap_vdop.N)
norm_ncp = colors.BoundaryNorm(np.arange(0, 1.1, 0.1), cmap_ncp.N)
norm_rhv = colors.BoundaryNorm(np.arange(0, 1.1, 0.1), cmap_rhv.N)
norm_sw = colors.BoundaryNorm(np.arange(0, 11, 1), cmap_sw.N)
norm_phi = colors.BoundaryNorm(np.arange(0, 365, 5), cmap_phi.N)
ticks_refl = np.arange(-8, 72, 8)
ticks_vdop = np.arange(-16, 20, 4)
ticks_ncp = np.arange(0, 1.2, 0.2)
ticks_rhv = np.arange(0, 1.2, 0.2)
ticks_sw = np.arange(0, 11, 1)
ticks_phi = np.arange(0, 420, 60)


def _pcolormesh(radar, field, sweep=0, cmap=None, norm=None, ax=None):
    """
    """
    if ax is None:
        ax = plt.gca()

    # Parse sweep data
    s0 = radar.sweep_start_ray_index['data'][sweep]
    sn = radar.sweep_end_ray_index['data'][sweep]
    time_sweep = num2date(
       radar.time['data'][(s0 + sn) / 2], radar.time['units'])

    # Compute radar sweep coordinates
    AZI, RNG = np.meshgrid(np.radians(radar.azimuth['data'][s0:sn]),
                           radar.range['data'] / 1000.0,
                           indexing='ij')
    X = RNG * np.sin(AZI)
    Y = RNG * np.cos(AZI)

    # Create plot
    qm = ax.pcolormesh(X, Y, radar.fields[field]['data'][s0:sn,:], cmap=cmap,
                       norm=norm, shading='flat', alpha=None)

    # Create title
    title = '{} {:.1f} deg {}Z\n {}'.format(
       radar.metadata['instrument_name'], radar.fixed_angle['data'][sweep],
       time_sweep.isoformat(), radar.fields[field]['long_name'])
    ax.set_title(title)

    return qm


def multipanel(inpdir, outdir, stamp, dpi=50, verbose=False):
    """
    """

    files = [os.path.join(inpdir, f) for f in sorted(os.listdir(inpdir))
             if stamp in f]
    if verbose:
        print 'Number of files to plot = %i' % len(files)

    for f in files:

        # Parse input file
        radar = read_mdv(f, exclude_fields=EXCLUDE_FIELDS)

        if radar.nsweeps != VCP_SWEEPS:
            continue

        if verbose:
            print 'Currently plotting file %s' % os.path.basename(f)

        # Create figure instance
        subs = {'xlim': (-117, 117), 'ylim': (-117, 117)}
        figs = {'figsize': (62, 45)}
        fig, ax = plt.subplots(
            nrows=6, ncols=len(SWEEPS), subplot_kw=subs, **figs)

        # Iterate over each sweep
        for j, sweep in enumerate(SWEEPS):

            # (a) Raw reflectivity
            qma = _pcolormesh(
                radar, 'reflectivity', sweep=sweep, cmap=cmap_refl,
                norm=norm_refl, ax=ax[0,j])

            # (b) Raw Doppler velocity
            qmb = _pcolormesh(
                radar, 'velocity', sweep=sweep, cmap=cmap_vdop,
                norm=norm_vdop, ax=ax[1,j])

            # (c) Normalized coherent power
            qmc = _pcolormesh(
                radar, 'normalized_coherent_power', sweep=sweep, cmap=cmap_ncp,
                norm=norm_ncp, ax=ax[2,j])

            # (d) Correlation coefficient
            qmd = _pcolormesh(
                radar, 'cross_correlation_ratio', sweep=sweep, cmap=cmap_rhv,
                norm=norm_rhv, ax=ax[3,j])

            # (e) Spectrum width
	    qme = _pcolormesh(
                radar, 'spectrum_width', sweep=sweep, cmap=cmap_sw,
                norm=norm_sw, ax=ax[4,j])

            # (f) Differential phase
            qmf = _pcolormesh(
                radar, 'differential_phase', sweep=sweep, cmap=cmap_phi,
                norm=norm_phi, ax=ax[5,j])

        # Format plot axes
        for i, j in np.ndindex(ax.shape):
            ax[i,j].xaxis.set_major_locator(MultipleLocator(20))
            ax[i,j].xaxis.set_minor_locator(MultipleLocator(5))
            ax[i,j].yaxis.set_major_locator(MultipleLocator(20))
            ax[i,j].yaxis.set_minor_locator(MultipleLocator(5))
            ax[i,j].set_xlabel('Eastward Range from Radar (km)')
            ax[i,j].set_ylabel('Northward Range from Radar (km)')
            ax[i,j].grid(which='major')

        # Color bars

        # Save figure
        date_stamp = num2date(radar.time['data'].min(), radar.time['units'])
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
    parser.add_argument('-db', '--debug', nargs='?', type=bool, const=True,
                        default=False, help=None)
    args = parser.parse_args()

    if args.debug:
        print 'inpdir = %s' % args.inpdir
        print 'outdir = %s' % args.outdir
        print 'stamp = %s' % args.stamp
        print 'dpi = %i' % args.dpi

    # Call desired plotting function
    multipanel(args.inpdir, args.outdir, args.stamp, dpi=args.dpi,
               verbose=args.verbose)
