#!/usr/bin/python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset, num2date
from matplotlib import rcParams, colors
from matplotlib.ticker import MultipleLocator

from pyart.graph import cm

### GLOBAL VARIABLES ###
########################

# Height indices for plotting
HEIGHTS = [0, 8, 16, 24, 32, 40]

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

# Define the color maps, their boundaries, and the ticks
bins_refl = np.arange(-8, 66, 2)
bins_wvel = np.arange(-15, 16, 1)
bins_uvel = np.arange(-30, 32, 2)
bins_hdiv = np.arange(-10, 11, 1)
cmap_refl = cm.NWSRef
cmap_wvel = plt.get_cmap(name='jet')
cmap_uvel = plt.get_cmap(name='jet')
cmap_hdiv = plt.get_cmap(name='jet')
norm_refl = colors.BoundaryNorm(bins_refl, cmap_refl.N)
norm_wvel = colors.BoundaryNorm(bins_wvel, cmap_wvel.N)
norm_uvel = colors.BoundaryNorm(bins_uvel, cmap_uvel.N)
norm_hdiv = colors.BoundaryNorm(bins_hdiv, cmap_hdiv.N)
ticks_refl = np.arange(-8, 72, 8)
ticks_wvel = np.arange(-15, 18, 3)
ticks_hdiv = np.arange(-10, 12, 2)
ticks_uvel = np.arange(-30, 40, 10)


def pcolormesh(grid, field, index=0, cmap=None, norm=None, scale=None,
               ax=None):
    """
    """
    # Parse axis
    if ax is None:
        ax = plt.gca()

    # Parse scale
    if scale is None:
        scale = 1.0

    # Parse grid data
    time_start = num2date(grid.variables['time_start'][0],
                          grid.variables['time_start'].units)

    # Create plot
    qm = ax.pcolormesh(grid.variables['x_disp'][:] / 1000.0,
                       grid.variables['y_disp'][:] / 1000.0,
                       scale * grid.variables[field][0,index,:,:],
                       cmap=cmap, norm=norm, shading='flat', alpha=None)

    # Create title
    title = 'ConVV {:.1f} km {}Z\n {}'.format(
            grid.variables['z_disp'][index] / 1000.0,
            time_start.isoformat(), grid.variables[field].long_name)
    ax.set_title(title)

    return qm


def multipanel(inpdir, outdir, stamp, dpi=100, verbose=False):
    """
    """
    files = [os.path.join(inpdir, f) for f in sorted(os.listdir(inpdir))
             if stamp in f]
    if verbose:
        print 'Number of files to plot = %i' % len(files)

    for f in files:
        if verbose:
            print 'Plotting file %s' % os.path.basename(f)

        # Parse input file
        grid = Dataset(f, mode='r')

        # Create figure instance
        subs = {'xlim': (-50, 50), 'ylim': (-50, 50)}
        figs = {'figsize': (50, 40)}
        fig, ax = plt.subplots(nrows=5, ncols=6, subplot_kw=subs, **figs)

        # Loop over each height
        for i, k in enumerate(HEIGHTS):

            # (a) Corrected reflectivity
            qma = pcolormesh(grid, 'network_corrected_reflectivity', index=k,
                             cmap=cmap_refl, norm=norm_refl, ax=ax[0,i])

            # (b) Vertical velocity
            qmb = pcolormesh(grid, 'vertical_wind_component', index=k,
                             cmap=cmap_wvel, norm=norm_wvel, ax=ax[1,i])

            # (c) Horizontal wind divergence
            qmc = pcolormesh(grid, 'horizontal_wind_divergence', index=k,
                             cmap=cmap_hdiv, norm=norm_hdiv, scale=1.0e3,
                             ax=ax[2,i])

            # (d) Eastward velocity
            qmd = pcolormesh(grid, 'eastward_wind_component', index=k,
                             cmap=cmap_uvel, norm=norm_uvel, ax=ax[3,i])

            # (e) Northward velocity
            qme = pcolormesh(grid, 'northward_wind_component', index=k,
                             cmap=cmap_uvel, norm=norm_uvel, ax=ax[4,i])

        # Format axes
        for i, j in np.ndindex(ax.shape):
            ax[i,j].xaxis.set_major_locator(MultipleLocator(10))
            ax[i,j].xaxis.set_minor_locator(MultipleLocator(5))
            ax[i,j].yaxis.set_major_locator(MultipleLocator(10))
            ax[i,j].yaxis.set_minor_locator(MultipleLocator(5))
            ax[i,j].set_xlabel('Eastward Distance from Origin (km)')
            ax[i,j].set_ylabel('Northward Distance from Origin (km)')
            ax[i,j].grid(which='major')

        # Color bars
        plt.colorbar(mappable=qma, cax=fig.add_axes([0.91, 0.78, 0.01, 0.10]),
                     ticks=ticks_refl)
        plt.colorbar(mappable=qmb, cax=fig.add_axes([0.91, 0.61, 0.01, 0.10]),
                     ticks=ticks_wvel)
        plt.colorbar(mappable=qmc, cax=fig.add_axes([0.91, 0.45, 0.01, 0.10]),
                     ticks=ticks_hdiv)
        plt.colorbar(mappable=qmd, cax=fig.add_axes([0.91, 0.28, 0.01, 0.10]),
                     ticks=ticks_uvel)
        plt.colorbar(mappable=qme, cax=fig.add_axes([0.91, 0.12, 0.01, 0.10]),
                     ticks=ticks_uvel)

        # Save figure
	time_start = grid.variables['time_start']
        date_stamp = num2date(time_start[0], time_start.units)
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
    parser.add_argument('--dpi', nargs='?', const=50, default=50, type=int,
                        help=None)
    parser.add_argument('-v', '--verbose', nargs='?', const=True,
                        default=False, type=bool, help=None)
    args = parser.parse_args()

    # Call desired plotting function
    multipanel(args.inpdir, args.outdir, args.stamp, dpi=args.dpi,
               verbose=args.verbose)
