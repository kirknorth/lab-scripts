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
SWEEPS = [0, 1, 2, 3, 5, 8, 10, 13]

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
cmap_width = plt.get_cmap('jet')
norm_refl = colors.BoundaryNorm(np.arange(-8, 66, 2), cmap_refl.N)
norm_vdop_raw = colors.BoundaryNorm(np.arange(-33, 35, 2), cmap_vdop.N)
norm_vdop_cor = colors.BoundaryNorm(np.arange(-33, 35, 2), cmap_vdop.N)
norm_width = colors.BoundaryNorm(np.arange(0, 10.5, 0.5), cmap_width.N)


def pcolormesh(radar, field, instrument_name, sweep=0, cmap=None, norm=None,
               ax=None):
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
            instrument_name, radar.variables['fixed_angle'][sweep],
            time_sweep.isoformat(), radar.variables[field].long_name)
    ax.set_title(title)

    return qm


def multipanel(inpdir, outdir, stamp, instrument_name, dpi=100, verbose=False):
    """
    """
    files = [os.path.join(inpdir, f) for f in sorted(os.listdir(inpdir))
             if stamp in f]
    if verbose:
        print 'Number of files to plot = %i' % len(files)

    for f in files:
        if verbose:
            print 'Currently plotting file %s' % os.path.basename(f)

        # Parse input file
        radar = Dataset(f, mode='r')

        # Create figure instance
        subs = {'xlim': (-400, 400), 'ylim': (-400, 400)}
        figs = {'figsize': (8, 6)}
        fig, ax = plt.subplots(
            nrows=5, ncols=len(SWEEPS), subplot_kw=subs, **figs)

        # Iterate over each sweep
        for j, sweep in enumerate(SWEEPS):

            # (a) Raw reflectivity
            qma = pcolormesh(radar, 'reflectivity', instrument_name,
                             sweep=sweep, cmap=cmap_refl, norm=norm_refl,
                             ax=ax[0,j])

            # (b) Corrected reflectivity
            qmb = pcolormesh(radar, 'corrected_reflectivity', instrument_name,
                             sweep=sweep, cmap=cmap_refl, norm=norm_refl,
                             ax=ax[1,j])

            # (c) Raw Doppler velocity
            qmc = pcolormesh(radar, 'velocity', instrument_name, sweep=sweep,
                             cmap=cmap_vdop, norm=norm_vdop_raw, ax=ax[2,j])

            # (d) Corrected Doppler velocity
            qmd = pcolormesh(radar, 'corrected_velocity', instrument_name,
                             sweep=sweep, cmap=cmap_vdop, norm=norm_vdop_cor,
                             ax=ax[3,j])

            # (e) Spectrum width
            qme = pcolormesh(radar, 'spectrum_width', instrument_name,
                             sweep=sweep, cmap=cmap_width, norm=norm_width,
                             ax=ax[4,j])

        # Format plot axes
        for i, j in np.ndindex(ax.shape):
            ax[i,j].xaxis.set_major_locator(MultipleLocator(100))
            ax[i,j].xaxis.set_minor_locator(MultipleLocator(50))
            ax[i,j].yaxis.set_major_locator(MultipleLocator(100))
            ax[i,j].yaxis.set_minor_locator(MultipleLocator(50))
            ax[i,j].set_xlabel('Eastward Range from Radar (km)')
            ax[i,j].set_ylabel('Northward Range from Radar (km)')
            ax[i,j].grid(which='major')

        # Save figure
        time_offset = radar.variables['time_offset']
        date_stamp = num2date(time_offset[:].min(), time_offset.units)
        filename = '{}.png'.format(
            os.path.join(outdir, date_stamp.strftime('%Y%m%d.%H%M%S')))
        fig.savefig(filename, format='png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return


if __name__ == '__main__':
    description = 'Plot multiple X-band PPI files'

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('inpdir', type=str, help=None)
    parser.add_argument('outdir', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('instrument_name', type=str, help=None)
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
        print 'instrument_name = %s' % args.instrument_name
        print 'dpi = %i' % args.dpi

    # Call desired plotting function
    multipanel(args.inpdir, args.outdir, args.stamp, args.instrument_name,
               dpi=args.dpi, verbose=args.verbose)
