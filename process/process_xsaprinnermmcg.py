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
COORDS = [np.arange(0, 10250, 250),
          np.arange(-50000, 50500, 500),
          np.arange(-50000, 50500, 500)]
ORIGIN = [36.605, -97.485]


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
    num_streams = str(len(streams))

    return steams, num_streams


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
        'radar_0_lon': radar.longitude['data'][0],
        'radar_0_lat': radar.latitude['data'][0],
        'radar_0_alt': radar.altitude['data'][0],
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


def process(fcmac, grid_coords, Fn, dl, facility_id, grid_origin=None,
            fields=None, toa=17000.0, leafsize=10, k=100, eps=0.0,
            weighting_function='Barnes', smooth_func='constant',
            roi_func='constant', cutoff_radius=5000.0, data_spacing=1220.0,
            kappa_star=0.5, h_factor=3.0, nb=1.5, bsp=2.0, min_radius=250.0,
            map_roi=True, map_dist=True, proj='lcc', datum='NAD83',
            ellps='GRS80', format='NETCDF4_CLASSIC'):
    """
    """

    # Read radar file
    radar = read_cfradial(fcmac)

    # Grid radar data
    grid = map_radar_to_grid(
        radar, grid_coords, grid_origin=grid_origin, fields=fields, toa=toa,
        leafsize=leafsize, k=k, eps=eps, weighting_function=weighting_function,
        smooth_func=smooth_func, roi_func=roi_func,
        cutoff_radius=cutoff_radius, data_spacing=data_spacing,
        kappa_star=kappa_star, h_factor=h_factor, nb=nb, bsp=bsp,
        min_radius=min_radius, map_roi=map_roi, map_dist=map_dist, proj=proj,
        datum=datum, ellps=ellps, debug=False)

    # ARM file name protocols
    date_stamp = num2date(grid.axes['time_start']['data'][0],
                          grid.axes['time_start']['units'])
    filename = 'sgpxsaprinnermmcg{}.{}.{}'.format(
        Fn, dl, date_stamp.strftime('%Y%m%d.%H%M%S'))

    # Write gridded data to file
    grid.write(output + filename, format=format)

    return


if __name__ = '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('source', type=str, help=None)
    parser.add_argument('output', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('Fn', type=str, help=None)
    parser.add_argument('dl', type=str, help=None)
    parser.add_argument('facility_id', type=str, help=None)
    parser.add_argument('-v', '--verbose', nargs='?', const=False, type=bool,
                        help=None)
    args = parser.parse_args()

    files = [args.source + f for f in sorted(os.listdir(args.source)) if
             args.stamp in f]
    if args.verbose:
        print 'Number of files to process = %i' % len(files)

    for fcmac in files:
        if args.verbose:
            print 'Processing file %s' % fcmac

        process(fcmac, grid_coords, args.Fn, args.dl, args.facility_id,
                grid_origin=grid_origin, fields=fields)
