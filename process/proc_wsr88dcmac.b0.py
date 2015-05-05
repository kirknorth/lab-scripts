#!/usr/bin/python

import os
import getpass
import platform
import argparse
import numpy as np

from netCDF4 import num2date
from datetime import datetime

from pyart.io import read_cfradial, write_grid
from pyart.config import get_fillvalue, get_field_name
from pyart.correct import dealias_region_based

from grid.interp import mapper


# Radar descriptors
QF = 'V06pblC1mmcg'
FN = 'KVNX'
DL = 'b1'
FACILITY = 'KVNX: Vance AFB, Oklahoma'

# Define k-d tree parameters
NUM_NEIGHBORS = 50
LEAFSIZE = 10

# Radar data objective analysis parameters
CUTOFF_RADIUS = np.inf
KAPPA_STAR = 0.5
DATA_SPACING = 1220.0
MAX_RANGE = None

# Define analysis domain
DOMAIN = 'PBL'
LAT_0 = 36.605
LON_0 = -97.485
ALT_0 = None
TOA = 17000.0

# Define various types of analysis domains of the form (z, y, x)
PBL_DOMAIN = [
    np.arange(0.0, 4100.0, 100.0),
    np.arange(-20000.0, 20250.0, 250.0),
    np.arange(-20000.0, 20250.0, 250.0),
    ]
CRM_DOMAIN = [
    np.arange(0.0, 10250.0, 250.0),
    np.arange(-20000.0, 20250.0, 250.0),
    np.arange(-20000.0, 20250.0, 250.0),
    ]

# Analysis domain dictionary
GRID_COORDS = {
    'PBL': PBL_DOMAIN,
    'CRM': CRM_DOMAIN,
    'RCM': None,
    }

# Define radar field names
REFL_FIELD = get_field_name('reflectivity')
CORR_REFL_FIELD = get_field_name('corrected_reflectivity')
VDOP_FIELD = get_field_name('velocity')
CORR_VDOP_FIELD = get_field_name('corrected_velocity')
WIDTH_FIELD = get_field_name('spectrum_width')
RHOHV_FIELD = get_field_name('cross_correlation_ratio')
ZDR_FIELD = get_field_name('differential_reflectivity')
PHIDP_FIELD = get_field_name('differential_phase')

# Fields to grid
FIELDS = [
    REFL_FIELD,
    CORR_REFL_FIELD,
    VDOP_FIELD,
    CORR_VDOP_FIELD,
    WIDTH_FIELD,
    RHOHV_FIELD,
    ZDR_FIELD,
    PHIDP_FIELD,
    ]

# Fields to exclude from radar object
EXCLUDE_FIELDS = None

# Define output NetCDF format
FORMAT = 'NETCDF4'


def process_file(filename, outdir, debug=False, verbose=False):
    """
    """

    # Read radar data
    radar = read_cfradial(filename, exclude_fields=EXCLUDE_FIELDS)

    if verbose:
        print 'Processing file: {}'.format(os.path.basename(filename))

    # Create radar data objective analysis weight
    weight = mapper.Weight(
        radar, cutoff_radius=CUTOFF_RADIUS, kappa_star=KAPPA_STAR,
        data_spacing=DATA_SPACING)

    # Grid radar data
    grid = mapper.grid_radar(
        radar, GRID_COORDS[DOMAIN], weight=weight, fields=FIELDS, lat_0=LAT_0,
        lon_0=LON_0, alt_0=ALT_0, toa=TOA, max_range=MAX_RANGE,
        k=NUM_NEIGHBORS, leafsize=LEAFSIZE, eps=0.0, debug=debug,
        verbose=verbose)

    # Add new metadata
    _add_metadata(grid, filename)

    # ARM file name protocols
    date_stamp = num2date(radar.time['data'].min(), radar.time['units'])
    fname = 'nexradwsr88d{}{}.{}.{}.cdf'.format(
        QF, FN, DL, date_stamp.strftime('%Y%m%d.%H%M%S'))

    # Write CMAC NetCDF file
    write_grid(os.path.join(outdir, fname), grid, format=FORMAT,
               arm_time_variables=True)

    return


def _add_metadata(grid, filename):
    """
    """

    # Datastreams attributes
    datastream = _parse_datastreams(filename)
    datastream_description = ('A string consisting of the datastream(s), '
                              'datastream version(s), and datastream '
                              'date (range).')

    # History attribute
    user = getpass.getuser()
    host = platform.node()
    time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    history = 'created by user {} on {} at {}'.format(user, host, time)

    # Add new metadata
    grid.metadata['institution'] = 'McGill University'
    grid.metadata['country'] = 'USA'
    grid.metadata['Conventions'] = 'CF/Radial'
    grid.metadata['facility_id'] = FACILITY
    grid.metadata['input_datastreams_description'] = datastream_description
    grid.metadata['input_datastreams_num'] = 1
    grid.metadata['input_datastreams'] = datastream
    grid.metadata['title'] = 'Mapped Moments to Cartesian Grid'
    grid.metadata['field_names'] = ''
    grid.metadata['history'] = history

    return


def _parse_datastreams(filename, version='-9999'):
    """
    """

    split = os.path.basename(filename).split('.')
    stream, dl, date, time = split[:4]
    datastream = '{}.{} : {} : {}.{}'.format(stream, dl, version, date, time)

    return datastream


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('inpdir', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('outdir', type=str, help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, default=False,
                        const=True, help=None)
    parser.add_argument('-db', '--debug', nargs='?', type=bool, default=False,
                        const=True, help=None)
    args = parser.parse_args()

    if args.verbose:
        print 'QF -> {}'.format(QF)
        print 'FN -> {}'.format(FN)
        print 'DL -> {}'.format(DL)
        print 'FACILITY -> {}'.format(FACILITY)
        print 'DOMAIN -> {}'.format(DOMAIN)
        print 'LAT_0 -> {}'.format(LAT_0)
        print 'LON_0 -> {}'.format(LON_0)
        print 'ALT_0 -> {}'.format(ALT_0)
        print 'TOA -> {} m'.format(TOA)
        print 'NUM_NEIGHBORS -> {}'.format(NUM_NEIGHBORS)
        print 'LEAFSIZE -> {}'.format(LEAFSIZE)
        print 'CUTOFF_RADIUS -> {} m'.format(CUTOFF_RADIUS)
        print 'KAPPA_STAR -> {}'.format(KAPPA_STAR)
        print 'DATA_SPACING -> {} m'.format(DATA_SPACING)

    # Parse all radar files to process
    files = [os.path.join(args.inpdir, f) for f in
             sorted(os.listdir(args.inpdir)) if args.stamp in f]
    if args.verbose:
        print 'Number of files to process = {}'.format(len(files))

    for filename in files:
        process_file(
            filename, args.outdir, debug=args.debug, verbose=args.verbose)


