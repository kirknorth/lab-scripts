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

### GLOBAL VARIABLES ###
########################
SWEEPS = [0, 1, 2, 5, 8, 13, 17, 21]


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
cmap_ncp = plt.get_cmap('jet')
cmap_rhv = plt.get_cmap('jet')
norm_refl = colors.BoundaryNorm(np.arange(-8, 66, 2), cmap_refl.N)
norm_vdop_raw = colors.BoundaryNorm(np.arange(-17, 18, 1), cmap_vdop.N)
norm_vdop_cor = colors.BoundaryNorm(np.arange(-30, 32, 2), cmap_vdop.N)
norm_ncp = colors.BoundaryNorm(np.arange(0, 1.1, 0.1), cmap_ncp.N)
norm_rhv = colors.BoundaryNorm(np.arange(0, 1.1, 0.1), cmap_rhv.N)


def pcolormesh(radar, field, sweep=0, cmap=None, norm=None, ax=None):
    """
    """
    if ax is None:
        ax = plt.gca()

    # Parse sweep data
    s0 = radar.variables['sweep_start_ray_index'][sweep]
    sf = radar.variables['sweep_end_ray_index'][sweep]
    time_sweep = num2date(radar.variables['time_offset'][(s0 + sf) / 2],
                          radar.variables['time_offset'].units)

    # Compute radar sweep coordinates
    AZI, RNG = np.meshgrid(np.radians(radar.variables['azimuth'][s0:sf]),
                           radar.variables['range'][:] / 1000.0,
                           indexing='ij')
    X = RNG * np.sin(AZI)
    Y = RNG * np.cos(AZI)

    # Create plot
    qm = ax.pcolormesh(X, Y, radar.variables[field][s0:sf,:], cmap=cmap,
                       norm=norm, shading='flat', alpha=None)

    # Create title
    title = '{} {:.1f} deg {}Z\n {}'.format(
            radar.instrument_name, radar.variables['fixed_angle'][sweep],
            time_sweep.isoformat(), radar.variables[field].long_name)
    ax.set_title(title)

    return qm


def multipanel(source, output, stamp, dpi=100, verbose=False):
    """
    """
    files = [source + f for f in sorted(os.listdir(source)) if stamp in f]
    if verbose:
        print 'Number of files to plot = %i' % len(files)

    for f in files:
        if verbose:
            print 'Currently plotting file %s' % os.path.basename(f)

        # Parse input file
        radar = Dataset(f, mode='r')
        time_offset = radar.variables['time_offset']

        # Create figure instance
        subs = {'xlim': (-40, 40), 'ylim': (-40, 40)}
        figs = {'figsize': (62, 45)}
        fig, ax = plt.subplots(nrows=6, ncols=len(SWEEPS),
                               subplot_kw=subs, **figs)

        # Iterate over each sweep
        for j, sweep in enumerate(SWEEPS):

            # (a) Raw reflectivity
            qma = pcolormesh(radar, 'reflectivity', sweep, cmap=cmap_refl,
                             norm=norm_refl, ax=ax[0,j])

            # (b) Corrected reflectivity
            qmb = pcolormesh(radar, 'corrected_reflectivity', sweep,
                             cmap=cmap_refl, norm=norm_refl, ax=ax[1,j])

            # (c) Raw Doppler velocity
            qmc = pcolormesh(radar, 'velocity', sweep,
                             cmap=cmap_vdop, norm=norm_vdop_raw, ax=ax[2,j])

            # (d) Corrected Doppler velocity
            qmd = pcolormesh(radar, 'corrected_velocity', sweep,
                             cmap=cmap_vdop, norm=norm_vdop_cor, ax=ax[3,j])

            # (e) Normalized coherent power
            qme = pcolormesh(radar, 'normalized_coherent_power', sweep,
                             cmap=cmap_ncp, norm=norm_ncp, ax=ax[4,j])

            # (f) Correlation coefficient
            qmf = pcolormesh(radar, 'cross_correlation_ratio', sweep,
                             cmap=cmap_rhv, norm=norm_rhv, ax=ax[5,j])

        # Format plot axes
        for i, j in np.ndindex(ax.shape):
            ax[i,j].xaxis.set_major_locator(MultipleLocator(10))
            ax[i,j].xaxis.set_minor_locator(MultipleLocator(5))
            ax[i,j].yaxis.set_major_locator(MultipleLocator(10))
            ax[i,j].yaxis.set_minor_locator(MultipleLocator(5))
            ax[i,j].set_xlabel('Eastward Range from Radar (km)')
            ax[i,j].set_ylabel('Northward Range from Radar (km)')
            ax[i,j].grid(which='major')


        # Save figure
        date_stamp = num2date(time_offset[:].min(), time_offset.units)
        filename = '{}{}.png'.format(
            output, date_stamp.strftime('%Y%m%d.%H%M%S'))
        fig.savefig(filename, format='png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return


if __name__ == '__main__':
    description = 'Plot multiple X-band PPI files'

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('source', type=str, help=None)
    parser.add_argument('output', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('--dpi', nargs='?', const=50, type=int, help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, const=True,
                        default=False, help=None)
    parser.add_argument('--debug', nargs='?', type=bool, const=True,
                        default=False, help=None)
    args = parser.parse_args()

    if args.debug:
        print 'source = %s' % args.source
        print 'output = %s' % args.output
        print 'stamp = %s' % args.stamp
        print 'dpi = %i' % args.dpi

    # Call desired plotting function
    multipanel(args.source, args.output, args.stamp, dpi=args.dpi,
               verbose=args.verbose)
