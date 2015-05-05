#!/usr/bin/python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset, num2date
from matplotlib import rcParams
from matplotlib.colors import BoundaryNorm
from matplotlib.dates import MinuteLocator, HourLocator, DateFormatter
from matplotlib.ticker import MultipleLocator

# Define the hours to plot
HOURS = np.arange(0, 24, 1)

# Define radar fields
TEMP_FIELD = 'temp'
PRES_FIELD = 'bar_pres'
RH_FIELD = 'rh'
VAP_PRES_FIELD = 'vap_pres'
WSPD_FIELD = 'wspd'
WDIR_FIELD = 'wdir'
UWIND_FIELD = 'u_wind'
VWIND_FIELD = 'v_wind'
DP_FIELD = 'dp'
THETA_FIELD = 'potential_temp'
SH_FIELD = 'sh'
FRAC_FIELD = 'sonde_fraction'
PRECIP_FIELD = 'precip'

# Define colour maps
CMAP_TEMP = plt.get_cmap('jet')
CMAP_PRES = plt.get_cmap('jet')
CMAP_RH = plt.get_cmap('jet')
CMAP_VAP_PRES = plt.get_cmap('jet')
CMAP_WSPD = plt.get_cmap('jet')
CMAP_WDIR = plt.get_cmap('jet')
CMAP_UWIND = plt.get_cmap('jet')
CMAP_VWIND = plt.get_cmap('jet')
CMAP_DP = plt.get_cmap('jet')
CMAP_THETA = plt.get_cmap('jet')
CMAP_SH = plt.get_cmap('jet')
CMAP_FRAC = plt.get_cmap('jet')

# Normalize colour maps
NORM_TEMP = BoundaryNorm(np.arange(-60, 21, 1), CMAP_TEMP.N)
NORM_PRES = BoundaryNorm(np.arange(0, 101, 1), CMAP_PRES.N)
NORM_RH = BoundaryNorm(np.arange(0, 101, 1), CMAP_RH.N)
NORM_VAP_PRES = BoundaryNorm(np.arange(0, 2.01, 0.01), CMAP_VAP_PRES.N)
NORM_WSPD = BoundaryNorm(np.arange(0, 51, 1), CMAP_WSPD.N)
NORM_WDIR = BoundaryNorm(np.arange(0, 365, 5), CMAP_WDIR.N)
NORM_UWIND = BoundaryNorm(np.arange(-40, 41, 1), CMAP_UWIND.N)
NORM_VWIND = BoundaryNorm(np.arange(-40, 41, 1), CMAP_VWIND.N)
NORM_DP = BoundaryNorm(np.arange(-60, 21, 1), CMAP_DP.N)
NORM_THETA = BoundaryNorm(np.arange(270, 391, 1), CMAP_THETA.N)
NORM_SH = BoundaryNorm(np.arange(0, 16.05, 0.05), CMAP_SH.N)
NORM_FRAC = BoundaryNorm(np.arange(0.9, 1.001, 0.001), CMAP_FRAC.N)

# Define colour bar ticks
TICKS_TEMP = np.arange(-60, 30, 10)
TICKS_PRES = np.arange(0, 110, 10)
TICKS_RH = np.arange(0, 110, 10)
TICKS_VAP_PRES = np.arange(0, 2.2, 0.2)
TICKS_WSPD = np.arange(0, 55, 5)
TICKS_WDIR = np.arange(0, 400, 40)
TICKS_UWIND = np.arange(-40, 50, 10)
TICKS_VWIND = np.arange(-40, 50, 10)
TICKS_DP = np.arange(-60, 30, 10)
TICKS_THETA = np.arange(270, 420, 30)
TICKS_SH = np.arange(0, 18, 2)
TICKS_FRAC = np.arange(0.9, 1.01, 0.01)


def multipanel(sonde, outdir, dpi=100, verbose=False):
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
    rcParams['font.size'] = 16

    # Compute height of 0 deg C isotherm
    height_melt = _melting_height(sonde)

    for i, hour in enumerate(HOURS):

        if verbose:
            print 'Plotting hour: {:02d}'.format(hour)

        # Bracket time indices for specified hour
        time, t0, tn = _bracket_time(sonde, hour=hour)

        # Initialze figure
        subs = {'xlim': (0, 60), 'ylim': (0, 16)}
        figs = {'figsize': (45, 33)}
        fig, ax = plt.subplots(nrows=4, ncols=3, subplot_kw=subs, **figs)
        plt.subplots_adjust(wspace=0.05, hspace=None)

        # (a) Temperature
        _pcolormesh(
            sonde, TEMP_FIELD, hour=hour, cmap=CMAP_TEMP, norm=NORM_TEMP,
            ticks=TICKS_TEMP, label='(deg C)', fig=fig, ax=ax[0,0])

        # (b) Pressure
        _pcolormesh(
            sonde, PRES_FIELD, hour=hour, cmap=CMAP_PRES, norm=NORM_PRES,
            ticks=TICKS_PRES, label='(kPa)', fig=fig, ax=ax[0,1])

        # (c) Relative humidity
        _pcolormesh(
            sonde, RH_FIELD, hour=hour, cmap=CMAP_RH, norm=NORM_RH,
            ticks=TICKS_RH, label='(%)', fig=fig, ax=ax[0,2])

        # (d) Dew point temperature
        _pcolormesh(
            sonde, DP_FIELD, hour=hour, cmap=CMAP_DP, norm=NORM_DP,
            ticks=TICKS_DP, label='(deg C)', fig=fig, ax=ax[1,0])

        # (e) Vapour pressure
        _pcolormesh(
            sonde, VAP_PRES_FIELD, hour=hour, cmap=CMAP_VAP_PRES,
            norm=NORM_VAP_PRES, label='(kPa)', ticks=TICKS_VAP_PRES, fig=fig,
            ax=ax[1,1])

        # (f) Specific humidity
        _pcolormesh(
            sonde, SH_FIELD, hour=hour, scale=1.0e3, cmap=CMAP_SH,
            norm=NORM_SH, label='(g/kg)', ticks=TICKS_SH, fig=fig, ax=ax[1,2])

        # (g) Potential temperature
        _pcolormesh(
            sonde, THETA_FIELD, hour=hour, cmap=CMAP_THETA, norm=NORM_THETA,
            ticks=TICKS_THETA, label='(K)', fig=fig, ax=ax[2,0])

        # (h) Wind speed
        _pcolormesh(
            sonde, WSPD_FIELD, hour=hour, cmap=CMAP_WSPD, norm=NORM_WSPD,
            ticks=TICKS_WSPD, label='(m/s)', fig=fig, ax=ax[2,1])

        # (i) Wind direction
        _pcolormesh(
            sonde, WDIR_FIELD, hour=hour, cmap=CMAP_WDIR, norm=NORM_WDIR,
            ticks=TICKS_WDIR, label='(deg)', fig=fig, ax=ax[2,2])

        # (j) Sounding fraction
        _pcolormesh(
            sonde, FRAC_FIELD, hour=hour, cmap=CMAP_FRAC, norm=NORM_FRAC,
            ticks=TICKS_FRAC, fig=fig, ax=ax[3,0])

        # (k) Eastward wind component
        _pcolormesh(
            sonde, UWIND_FIELD, hour=hour, cmap=CMAP_UWIND, norm=NORM_UWIND,
            ticks=TICKS_UWIND, label='(m/s)', fig=fig, ax=ax[3,1])

        # (l) Northward wind component
        _pcolormesh(
            sonde, VWIND_FIELD, hour=hour, cmap=CMAP_UWIND, norm=NORM_UWIND,
            ticks=TICKS_UWIND, label='(m/s)', fig=fig, ax=ax[3,2])

        # Height of 0 deg C isotherm
        for i, j in np.ndindex(ax.shape):
            ax[i,j].scatter(
                time, height_melt[t0:tn+1], s=60, facecolor='white',
                edgecolor='black', marker='o')

        # Precipitation accumulation
        for i, j in np.ndindex(ax.shape):
            axy = ax[i,j].twinx()
            axy.scatter(
                time, sonde.variables[PRECIP_FIELD][t0:tn+1], s=80,
                facecolor='orange', edgecolor='black', marker='o')
            axy.set_xlim(0, 60)
            axy.set_ylim(0, 5)
            axy.yaxis.set_major_locator(MultipleLocator(1))
            axy.yaxis.set_minor_locator(MultipleLocator(0.1))
            axy.set_ylabel('Precipitation Accumulation (mm)')

        # Format axes
        for i, j in np.ndindex(ax.shape):
            ax[i,j].xaxis.set_major_locator(MultipleLocator(5))
            ax[i,j].xaxis.set_minor_locator(MultipleLocator(1))
            ax[i,j].yaxis.set_major_locator(MultipleLocator(2))
            ax[i,j].yaxis.set_minor_locator(MultipleLocator(0.5))
            ax[i,j].set_xlabel('Minutes of Hour {:02d} (UTC)'.format(hour))
            ax[i,j].set_ylabel('Height (km AGL)')
            ax[i,j].grid(which='major', axis='both')

        # Image file name
        date = num2date(
            sonde.variables['time'][0], sonde.variables['time'].units).date()
        image = '{}.{:02d}0000.png'.format(date.strftime('%Y%m%d'), hour)

        # Save image
        fig.savefig(os.path.join(outdir, image), format='png', dpi=dpi,
                    bbox_inches='tight')

        # Close figure to release memory
        plt.close(fig)

    return


def _pcolormesh(sonde, field, hour=0, scale=1.0, cmap=None, norm=None,
                ticks=None, label=None, fig=None, ax=None):
    """
    """

    # Parse figure and axis
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        fig = plt.gca()

    # Parse colour map
    if cmap is None:
        cmap = plt.get_cmap(name='jet')

    # Parse colour bar label
    if label is None:
        label = ''

    # Parse sounding coordinates
    # The time variable should be seconds since midnight
    height = sonde.variables['height'][:] - sonde.variables['alt'][:] / 1.0e3
    time = sonde.variables['time'][:]

    # Bracket time indices for specified hour
    time, t0, tn = _bracket_time(sonde, hour=hour)

    # Parse sounding data within specified hour
    data = sonde.variables[field][t0:tn+1]

    # Create quadmesh
    qm = ax.pcolormesh(time, height, scale * data.T, cmap=cmap, norm=norm)

    # Create colour bar
    fig.colorbar(mappable=qm, ax=ax, extend='neither', orientation='vertical',
                 ticks=ticks, label=label)

    # Set title
    title = 'Merged Sounding\n{}'.format(sonde.variables[field].long_name)
    ax.set_title(title)

    return


def _bracket_time(sonde, hour=0):
    """
    """

    # Parse sounding time coordinate
    # The time variable should be seconds since midnight
    time = sonde.variables['time'][:]

    # Bracket time indices for specified hour
    t0 = np.abs(time - hour * 3600).argmin()
    tn = np.abs(time - (hour + 1) * 3600).argmin()

    # Convert time to minutes of specified hour
    time = (time[t0:tn+1] - 3600 * hour) / 60.0

    return time, t0, tn


def _melting_height(sonde):
    """
    """

    # Parse sounding coordinates
    time = sonde.variables['time'][:]
    height = sonde.variables['height'][:] - sonde.variables['alt'][:] / 1.0e3

    # Parse sounding temperature data
    temp = sonde.variables[TEMP_FIELD][:]

    # Compute 0 deg C isotherm height
    height_melt = np.zeros_like(time)
    for t in range(time.size):
        height_melt[t] = height[np.abs(temp[t] - 0.0).argmin()]

    return height_melt


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

    # Parse files to plot
    files = [os.path.join(args.inpdir, f) for
             f in sorted(os.listdir(args.inpdir)) if args.stamp in f]

    if args.verbose:
        print 'Number of files to plot: {}'.format(len(files))

    for f in files:

        if args.verbose:
            print 'Plotting file: {}'.format(os.path.basename(f))

        # Read radar data
        sonde = Dataset(f, mode='r')

        # Call desired plotting function
        multipanel(sonde, args.outdir, dpi=args.dpi, verbose=args.verbose)

