#!/usr/bin/python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset, num2date
from matplotlib import rcParams, colors
from matplotlib.ticker import MultipleLocator

from pyart.graph import cm


# Set figure parameters
# Axes line and tick sizes
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.size'] = 4
rcParams['xtick.major.width'] = 1
rcParams['xtick.minor.size'] = 2
rcParams['xtick.minor.width'] = 1
rcParams['ytick.major.size'] = 4
rcParams['ytick.major.width'] = 1
rcParams['ytick.minor.size'] = 2
rcParams['ytick.minor.width'] = 1

# Define the color maps and their boundaries
cmap_refl = cm.NWSRef
cmap_vdop = plt.get_cmap(name='jet')
cmap_ncp = plt.get_cmap(name='jet')
cmap_rhv = plt.get_cmap(name='jet')
norm_refl = colors.BoundaryNorm(np.arange(-8, 65, 2), cmap_refl.N)
norm_vdop = colors.BoundaryNorm(np.arange(-30, 32, 2), cmap_vdop.N)
norm_ncp = colors.BoundaryNorm(np.arange(0, 1.1, 0.1), cmap_ncp.N)
norm_rhv = colors.BoundaryNorm(np.arange(0, 1.1, 0.1), cmap_rhv.N)


def pcolormesh(grid, field, index=0, cmap=None, norm=None, ax=None):
    """
    """
    # Parse axis
    if ax is None:
        ax = plt.gca()

    # Parse grid data
    time_start = num2date(grid.variables['time_start'][0],
                          grid.variables['time_start'].units)

    # Create plot
    qm = ax.pcolormesh(grid.variables['x_disp'][:] / 1000.0,
                       grid.variables['y_disp'][:] / 1000.0,
                       grid.variables[field][0,index,:,:], cmap=cmap,
                       norm=norm, shading='flat', alpha=None)
    ax.grid(which='major')

    # Create title
    title = '{} {:.1f} km {}Z\n {}'.format(
            grid.radar_0_instrument_name,
            grid.variables['z_disp'][index] / 1000.0,
            time_start.isoformat(), grid.variables[field].long_name)
    ax.set_title(title)

    return qm


def multipanel(source, output, stamp, heights, dpi=100, verbose=False):
    """
    """
    files = [source + f for f in sorted(os.listdir(source)) if stamp in f]
    if verbose:
        print 'Number of files to plot = %i' % len(files)

    for f in files:
        if verbose:
            print 'Currently plotting file %s' % os.path.basename(f)

        # Parse input file
        grid = Dataset(f, mode='r')
        time_start = grid.variables['time_start']

        # Create figure instance
        subs = {'xlim': (-10, 10), 'ylim': (-10, 10)}
        figs = {'figsize': (50, 30)}
        fig, ax = plt.subplots(nrows=4, ncols=6, subplot_kw=subs, **figs)

        # Format axes
        for i, j in np.ndindex(ax.shape):
            ax[i,j].xaxis.set_major_locator(MultipleLocator(2))
            ax[i,j].xaxis.set_minor_locator(MultipleLocator(1))
            ax[i,j].yaxis.set_major_locator(MultipleLocator(2))
            ax[i,j].yaxis.set_minor_locator(MultipleLocator(1))
            ax[i,j].set_xlabel('Eastward Distance from Origin (km)')
            ax[i,j].set_ylabel('Northward Distance from Origin (km)')

        # Loop over each height
        for i, k in enumerate(heights):

            # (a) Corrected reflectivity
            qma = pcolormesh(grid, 'corrected_reflectivity', index=k,
                             cmap=cmap_refl, norm=norm_refl, ax=ax[0,i])

            # (b) Corrected radial velocity
            qmb = pcolormesh(grid, 'corrected_velocity', index=k,
                             cmap=cmap_vdop, norm=norm_vdop, ax=ax[1,i])

            # (c) Normalized coherent power
            qmc = pcolormesh(grid, 'normalized_coherent_power', index=k,
                             cmap=cmap_ncp, norm=norm_ncp, ax=ax[2,i])

            # (d) Copolar correlation coefficient
            qmd = pcolormesh(grid, 'cross_correlation_ratio', index=k,
                             cmap=cmap_rhv, norm=norm_rhv, ax=ax[3,i])

        # Color bars
        plt.colorbar(mappable=qma, cax=fig.add_axes([0.91, 0.74, 0.01, 0.15]))
        plt.colorbar(mappable=qmb, cax=fig.add_axes([0.91, 0.53, 0.01, 0.15]))
        plt.colorbar(mappable=qmc, cax=fig.add_axes([0.91, 0.32, 0.01, 0.15]),
                     ticks=np.arange(0, 1.1, 0.1))
        plt.colorbar(mappable=qmd, cax=fig.add_axes([0.91, 0.11, 0.01, 0.15]),
                     ticks=np.arange(0, 1.1, 0.1))

        # Save figure
        date_stamp = num2date(time_start[0], time_start.units)
        filename = '{}{}.png'.format(
            output, date_stamp.strftime('%Y%m%d.%H%M%S'))
        fig.savefig(filename, format='png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    return


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('source', type=str, help=None)
    parser.add_argument('output', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('--heights', nargs='?', const=[0, 8, 16, 24, 32, 40],
                        type=list, help=None)
    parser.add_argument('--dpi', nargs='?', const=50, type=int, help=None)
    parser.add_argument('-v', '--verbose', nargs='?', const=True, type=bool,
                        help=None)
    args = parser.parse_args()

    # Call desired plotting function
    multipanel(args.source, args.output, args.stamp, args.heights,
               dpi=args.dpi, verbose=args.verbose)
