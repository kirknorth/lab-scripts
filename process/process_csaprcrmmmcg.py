#!/usr/bin/python

import os
import platform
import getpass
import argparse
import numpy as np

from datetime import datetime
from netCDF4 import num2date

from pyart.io import read_cfradial
from pyart.config import get_fillvalue
from pyart.map.grid_test import map_radar_to_grid


### GLOBAL VARIABLES ###
########################

# Define fields to process
FIELDS = ['corrected_reflectivity', 'corrected_velocity',
          'normalized_coherent_power', 'cross_correlation_ratio']

# Define grid coordinates and origin
COORDS = [np.arange(0.0, 15250.0, 250.0),
          np.arange(-10000.0, 10250.0, 250.0),
          np.arange(-10000.0, 10250.0, 250.0)]
ORIGIN = [36.605, -97.485]

# Define gridding parameters
NUM_POINTS = 600
TOA = 17000.0
FUNCTION = 'Barnes'
SMOOTH = 'constant'
CUTOFF = 5000.0
SPACING = 1220.0
KAPPA = 0.5
ROI = None
MIN_RADIUS = 250.0
FORMAT = 'NETCDF4_CLASSIC'


def parse_datastreams(files, version='-9999'):
    """
    """
    streams = []
    for f in files:
        fsplit = os.path.basename(f).split('.')
        stream = '.'.join(fsplit[:2])
        date = '.'.join(fsplit[2:4])
        streams.append('{} : {} : {}'.format(stream, version, date))

    streams = ' ;\n '.join(streams)
    num_streams = len(files)

    return streams, num_streams


def create_metadata(radar, files):
    """
    """

    # Datastreams attributes
    datastreams, num_datastreams = parse_datastreams(files)
    datastream_description = ('A string consisting of the datastream(s), '
                              'datastream version(s), and datastream '
                              'date (range).')

    # History attribute
    user = getpass.getuser()
    host = platform.node()
    time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    history = 'created by user {} on {} at {}'.format(user, host, time)

    metadata = {
        'process_version': '',
        'command_line': '',
        'site_id': 'sgp',
        'facility_id': 'I7: Nardin, Oklahoma',
        'country': 'USA',
        'project': 'MC3E',
        'institution': 'ARM Climate Research Facility',
        'dod_version': '',
        'comment': '',
        'radar_0_instrument_name': radar.metadata['instrument_name'],
        'radar_0_longitude': radar.longitude['data'][0],
        'radar_0_latitude': radar.latitude['data'][0],
        'radar_0_altitude': radar.altitude['data'][0],
        'state': '',
        'Conventions': 'CF/Radial',
        'reference': '',
        'input_datastreams_num': num_datastreams,
        'input_datastreams': datastreams,
        'input_datastreams_description': datastream_description,
        'description': '',
        'title': 'Mapped Moments to Cartesian Grid',
        'field_names': '',
        'history': history}

    return metadata


def process(fcmac, output, dl, debug=False):
    """
    """

    # Read radar file
    radar = read_cfradial(fcmac)

    # Grid radar data
    grid = map_radar_to_grid(
        radar, COORDS, grid_origin=ORIGIN, fields=FIELDS, toa=TOA,
        leafsize=10, k=NUM_POINTS, eps=0.0, weighting_function=FUNCTION,
        smooth_func=SMOOTH, roi_func=ROI, cutoff_radius=CUTOFF,
        data_space=SPACING, kappa_star=KAPPA, h_factor=None, nb=None,
        bsp=None, min_radius=MIN_RADIUS, map_roi=True, map_dist=True,
        proj='lcc', datum='NAD83', ellps='GRS80', debug=debug)

    # Parse metadata
    grid.metadata = create_metadata(radar, [fcmac])

    # ARM file name protocols
    date_stamp = num2date(grid.axes['time_start']['data'][0],
                          grid.axes['time_start']['units'])
    filename = 'sgpcsaprinnermmcgI7.{}.{}.cdf'.format(
        dl, date_stamp.strftime('%Y%m%d.%H%M%S'))

    # Write gridded data to file
    grid.write(os.path.join(output, filename), format=FORMAT)

    return


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('source', type=str, help=None)
    parser.add_argument('output', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('dl', type=str, help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, const=True,
                        default=False, help=None)
    parser.add_argument('--debug', nargs='?', type=bool, const=True,
                        default=False, help=None)
    args = parser.parse_args()

    if args.debug:
        print 'source = %s' % args.source
        print 'output = %s' % args.output
        print 'stamp = %s' % args.stamp
        print 'dl = %s' % args.dl

    # Parse all files to process
    files = [os.path.join(args.source, f) for f in
             sorted(os.listdir(args.source)) if args.stamp in f]
    if args.verbose:
        print 'Number of files to process = %i' % len(files)

    for fcmac in files:
        if args.verbose:
            print 'Processing file %s' % os.path.basename(fcmac)

        process(fcmac, args.output, args.dl, args.debug)
