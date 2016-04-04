#!/usr/bin/python

import os
import getpass
import platform
import argparse
import numpy as np
from datetime import datetime

from pyart.io import read, grid_io
from pyart.filters import GateFilter
from pyart.config import get_fillvalue, get_field_name
from pyart.util.datetime_utils import datetimes_from_radar

from grid.interp import grid_radar
from grid.core import Weight, Domain


# Radar descriptors
RADAR = 'KVNX'
RADARS = {
    'KVNX': ['V06innerC1mmcg', 'KVNX', 'b1', 'KVNX: Vance AFB, Oklahoma'],
    'KICT': ['V03pblC1mmcg', 'KICT', 'b1', 'KICT: Wichita, Kansas'],
    'KINX': ['V03crmC1mmcg', 'KINX', 'b1', 'KINX: Tulsa, Oklahoma'],
    }
QF, FN, DL, FACILITY = RADARS[RADAR]

# Define k-d tree parameters
NUM_NEIGHBORS = 200
LEAFSIZE = 10

# Radar data objective analysis parameters
CUTOFF_RADIUS = np.inf
KAPPA_STAR = 0.5
DATA_SPACING = 1000.0
MAX_RANGE = None
TOA = 15000.0

# Define analysis domain location
SITE_ID = 'C1'
LOCATION = {
    'C1': [36.605, -97.485, None],
    'I8': [36.706, -97.388, None],
    'I9': [36.476, -97.422, None],
    'I10': [36.666, -97.624, None],
    }
ORIGIN = LOCATION[SITE_ID]

# Define various types of analysis domains of the form (z, y, x)
DOMAIN = 'INNER'
PBL_DOMAIN = [
    np.arange(0.0, 5250.0, 250.0),
    np.arange(-20000.0, 20250.0, 250.0),
    np.arange(-20000.0, 20250.0, 250.0),
    ]
CRM_DOMAIN = [
    np.arange(0.0, 10250.0, 250.0),
    np.arange(-10000.0, 10250.0, 250.0),
    np.arange(-10000.0, 10250.0, 250.0),
    ]
INNER_DOMAIN = [
    np.arange(0.0, 10500.0, 500.0),
    np.arange(-25000.0, 100500.0, 500.0),
    np.arange(-50000.0, 50500.0, 500.0)
    ]
COORDS = {
    'PBL': PBL_DOMAIN,
    'CRM': CRM_DOMAIN,
    'INNER': INNER_DOMAIN,
    }

# Define radar field names
REFL_FIELD = get_field_name('reflectivity')
REFL_CORR_FIELD = get_field_name('corrected_reflectivity')
VDOP_FIELD = get_field_name('velocity')
VDOP_CORR_FIELD = get_field_name('corrected_velocity')
SPW_FIELD = get_field_name('spectrum_width')
RHOHV_FIELD = get_field_name('cross_correlation_ratio')
ZDR_FIELD = get_field_name('differential_reflectivity')
PHIDP_FIELD = get_field_name('differential_phase')
SD_FIELD = get_field_name('radar_significant_detection')

# Fields to grid
FIELDS = [
    REFL_FIELD,
    VDOP_FIELD,
    SPW_FIELD,
    RHOHV_FIELD,
    ZDR_FIELD,
    PHIDP_FIELD,
    REFL_CORR_FIELD,
    VDOP_CORR_FIELD,
    ]

# Fields to exclude from radar object
EXCLUDE_FIELDS = None

# Define output NetCDF format
FORMAT = 'NETCDF3_CLASSIC'


def process_radar(radar, domain, weight, outdir, gatefilter=None, debug=False,
                  verbose=False):
    """
    """

    if verbose:
        print('Processing file: {}'.format(os.path.basename(filename)))

    # Read radar data
    radar = read(filename, exclude_fields=EXCLUDE_FIELDS)

    # Create gatefilter from significant detection
    gf = GateFilter(radar)
    gf.exclude_below(SD_FIELD, 1, op='or', inclusive=False)

    if debug:
        print('Number of sweeps: {}'.format(radar.nsweeps))

    # Grid radar data
    grid = grid_radar(
        radar, domain, weight=weight, fields=FIELDS, gatefilter=gf, toa=TOA,
        max_range=MAX_RANGE, gqi_field=None, legacy=True, debug=debug,
        verbose=verbose)

    # Add new metadata
    _add_metadata(grid, filename)

    # ARM file name protocols
    date_stamp = datetimes_from_radar(radar).min().strftime('%Y%m%d.%H%M%S')
    fname = 'nexradwsr88d{}{}.{}.{}.cdf'.format(QF, FN, DL, date_stamp)

    # Write MMCG NetCDF file
    grid_io.write_grid(
        os.path.join(outdir, fname), grid, format=FORMAT,
        write_proj_coord_sys=False, proj_coord_sys=None,
        arm_time_variables=True, write_point_x_y_z=False,
        write_point_lon_lat_alt=False)

    return


def _add_metadata(grid, filename):
    """ Add grid metadata. """

    # Datastreams attributes
    datastream = _parse_datastreams(filename)
    datastream_description = ('A string consisting of the datastream(s), '
                              'datastream version(s), and datastream '
                              'date (range).')

    # History attribute
    history = 'created by user {} on {} at {}'.format(
        getpass.getuser(), platform.node(),
        datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))

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
    """ ARM datastreams attribute. """

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
        print 'SITE_ID -> {}'.format(SITE_ID)
        print('LAT_0 -> {}'.format(ORIGIN[0]))
        print('LON_0 -> {}'.format(ORIGIN[1]))
        print('ALT_0 -> {}'.format(ORIGIN[2]))
        print 'TOA -> {} m'.format(TOA)
        print 'NUM_NEIGHBORS -> {}'.format(NUM_NEIGHBORS)
        print 'LEAFSIZE -> {}'.format(LEAFSIZE)
        print 'CUTOFF_RADIUS -> {} m'.format(CUTOFF_RADIUS)
        print 'KAPPA_STAR -> {}'.format(KAPPA_STAR)
        print 'DATA_SPACING -> {} m'.format(DATA_SPACING)
        print 'MAX_RANGE -> {} m'.format(MAX_RANGE)

    # Parse all radar files to process
    files = [os.path.join(args.inpdir, f) for f in
             sorted(os.listdir(args.inpdir)) if args.stamp in f]

    if args.verbose:
        print 'Number of files to process = {}'.format(len(files))

    # Define grid domain
    domain = Domain(COORDS[DOMAIN], ORIGIN, proj='lcca', ellps='WGS84',
                    datum='WGS84', dem=None)

    # Define radar data objective analysis weight
    weight = Weight(
        func=None, cutoff_radius=CUTOFF_RADIUS, kappa_star=KAPPA_STAR,
        data_spacing=DATA_SPACING, k=NUM_NEIGHBORS, leafsize=LEAFSIZE)
    weight.compute_distance_weight_vanishes(verbose=args.verbose)

    for filename in files:

        process_radar(filename, domain, weight, args.outdir, debug=args.debug,
            verbose=args.verbose)
