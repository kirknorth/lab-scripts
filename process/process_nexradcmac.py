#!/usr/bin/python

import os
import platform
import getpass
import argparse
import numpy as np

from netCDF4 import Dataset, num2date
from datetime import datetime

from pyart.io import read_nexrad_archive, write_cfradial
from pyart.correct import dealias_unwrap_phase, GateFilter
from pyart.config import get_fillvalue, get_field_name

### GLOBAL VARIABLES ###
########################

# Define number of elevation scans for NEXRAD convective VCP
NUM_TILS = 14

# Define output parameters
FORMAT = 'NETCDF4'


def parse_datastreams(files, dl='00', version='-9999'):
    """
    """
    streams = []
    for f in files:
        fsplit = os.path.basename(f).split('_')
        stream = 'nexrad{}{}.{}'.format(fsplit[2], fsplit[0][:4], dl)
        date = '{}.{}'.format(fsplit[0][4:], fsplit[1])
        streams.append('{} : {} : {}'.format(stream, version, date))

    streams = ' ;\n '.join(streams)
    num_streams = len(files)

    return streams, num_streams


def create_metadata(radar, files, facility_id):
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
        'site_id': '',
        'facility_id': facility_id,
        'country': 'USA',
        'project': '',
        'institution': 'National Weather Service',
        'dod_version': '',
        'comment': '',
        'instrument_name': radar.metadata['instrument_name'],
        'state': '',
        'Conventions': 'CF/Radial',
        'reference': '',
        'input_datastreams_num': num_datastreams,
        'input_datastreams': datastreams,
        'input_datastreams_description': datastream_description,
        'description': '',
        'title': 'Corrected Moments in Antenna Coordinates',
        'field_names': '',
        'history': history}

    return metadata


def process(filename, outdir, qualifier, Fn, dl, facility_id, nyquist=33.0,
            debug=False, verbose=False, vel_field=None, refl_field=None,
            corr_vel_field=None, corr_refl_field=None):
    """
    """

    # Parse field name
    if vel_field is None:
        vel_field = get_field_name('velocity')
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if corr_vel_field is None:
        corr_vel_field = get_field_name('corrected_velocity')
    if corr_refl_field is None:
        corr_refl_field = get_field_name('corrected_reflectivity')

    # Read radar file
    radar = read_nexrad_archive(filename)

    # Check if radar VCP is convective
    if np.unique(radar.fixed_angle['data']).size < NUM_TILS:
        if verbose:
            print 'File %s not convective VCP' % os.path.basename(filename)
        return

    # Mask invalid radar gates
    gate_filter = GateFilter(radar, exclude_based=True)

    # Doppler velocity correction
    vel_corr = dealias_unwrap_phase(
        radar, unwrap_unit='sweep', nyquist_vel=nyquist,
        gatefilter=gate_filter, rays_wrap_around=True, keep_original=True,
        vel_field=vel_field, corr_vel_field=corr_vel_field)
    radar.fields.update({corr_vel_field: vel_corr})

    # Reflectivity correction
    refl_corr = radar.fields[refl_field]
    radar.fields.update({corr_refl_field: refl_corr})

    # Parse metadata
    radar.metadata = create_metadata(radar, [filename], facility_id)

    # ARM file name protocols
    date_stamp = num2date(radar.time['data'].min(), radar.time['units'])
    fname = 'nexradwsr88d{}cmac{}.{}.{}.cdf'.format(
        qualifier, Fn, dl, date_stamp.strftime('%Y%m%d.%H%M%S'))

    # Write CMAC data file
    write_cfradial(os.path.join(outdir, fname), radar, format=FORMAT,
                   arm_time_variables=True)

    return


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('inpdir', type=str, help=None)
    parser.add_argument('outdir', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('qualifier', type=str, help=None)
    parser.add_argument('Fn', type=str, help=None)
    parser.add_argument('dl', type=str, help=None)
    parser.add_argument('facility_id', type=str, help=None)
    parser.add_argument('-v', '--verbose', nargs='?', const=True,
                        default=False, type=bool, help=None)
    parser.add_argument('--debug', nargs='?', type=bool, const=True,
                        default=False, help=None)
    args = parser.parse_args()

    # Parse all radar files to process
    files = [os.path.join(args.inpdir, f) for f in
             sorted(os.listdir(args.inpdir)) if args.stamp in f]
    if args.verbose:
        print 'Number of files to process = %i' % len(files)

    for filename in files:
        if args.verbose:
            print 'Processing file %s' % os.path.basename(filename)

        process(filename, args.outdir, args.qualifier, args.Fn, args.dl,
                args.facility_id, nyquist=33.0, debug=args.debug,
                verbose=args.verbose)
