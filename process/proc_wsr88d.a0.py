#!/usr/bin/python

import os
import platform
import getpass
import argparse
import numpy as np

from netCDF4 import num2date
from datetime import datetime

from pyart.io import write_cfradial
from pyart.aux_io import read_radx
from pyart.util.datetime_utils import datetime_from_radar
from pyart.correct import dealias_region_based, GateFilter
from pyart.config import get_fillvalue, get_field_name, FileMetadata


# Radar descriptors
QF = 'V06'
FN = 'KVNX'
DL = 'b0'
FACILITY = 'KVNX: Vance AFB, Oklahoma'

# Define number of sweeps for VCP
NSWEEPS = 14

# Doppler velocity correction routine and parameters
DEALIAS = 'region'
INTERVAL_SPLITS = 5

# Define output NetCDF format
FORMAT = 'NETCDF4'

# Define radar field names
REFL_FIELD = get_field_name('reflectivity')
CORR_REFL_FIELD = get_field_name('corrected_reflectivity')
CORR_VDOP_FIELD = get_field_name('corrected_velocity')

# Define radar fields to remove from radar object
REMOVE_FIELDS = [
    'REF_s1',
    'REF_s3',
    'REF_s5',
    ]


def process_file(filename, outdir, debug=False, verbose=False):
    """
    """

    # Read radar data
    radar = read_radx(filename)

    # Radar VCP check
    if NSWEEPS is not None and radar.nsweeps != NSWEEPS:
        return

    if verbose:
        print 'Processing file: {}'.format(os.path.basename(filename))

    # Create gate filter
    gf = GateFilter(radar, exclude_based=True)

    # Create file metadata object
    meta = FileMetadata(
        'nexrad_archive', field_names=None, additional_metadata=None,
        file_field_names=False, exclude_fields=None)

    # Step 1: Remove unnecessary fields
    for field in REMOVE_FIELDS:
        radar.fields.pop(field, None)

    # Step 2: Rename fields
    for field in radar.fields.keys():
        default_field = meta.get_field_name(field)
        radar.fields[default_field] = radar.fields.pop(field, None)

    # Step 3: Doppler velocity correction
    if DEALIAS == 'phase':
        vdop_corr = dealias_unwrap_phase(
            radar, gatefilter=gf, unwrap_unit='sweep', nyquist_vel=None,
            rays_wrap_around=True, keep_original=False, vel_field=None,
            corr_vel_field=CORR_VDOP_FIELD)

    elif DEALIAS == 'region':
        vdop_corr = dealias_region_based(
            radar, gatefilter=gf, interval_splits=INTERVAL_SPLITS,
            interval_limits=None, skip_between_rays=2, skip_along_ray=2,
            centered=True, nyquist_vel=None, rays_wrap_around=False,
            keep_original=False, vel_field=None,
            corr_vel_field=CORR_VDOP_FIELD)
    else:
        raise ValueError('Unsupported velocity correction routine')

    radar.add_field(CORR_VDOP_FIELD, vdop_corr, replace_existing=True)

    # Step 4: Reflectivity correction
    # Currently no correction procedures are applied to the reflectivity field
    # due to minimal attenuation at S-band
    refl_corr = radar.fields[REFL_FIELD]
    radar.add_field(CORR_REFL_FIELD, refl_corr, replace_existing=True)

    # Add metadata
    _add_metadata(radar, filename)

    # ARM file name protocols
    date_stamp = num2date(radar.time['data'].min(), radar.time['units'])
    fname = 'nexradwsr88d{}cmac{}.{}.{}.cdf'.format(
        QF, FN, DL, date_stamp.strftime('%Y%m%d.%H%M%S'))

    # Write CMAC NetCDF file
    write_cfradial(os.path.join(outdir, fname), radar, format=FORMAT,
                   arm_time_variables=True)

    return


def _add_metadata(radar, filename):
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
    radar.metadata['process_version'] = '',
    radar.metadata['command_line'] = '',
    radar.metadata['site_id'] = '',
    radar.metadata['facility_id'] = FACILITY,
    radar.metadata['country'] = 'USA',
    radar.metadata['project'] = '',
    radar.metadata['institution'] = 'McGill University',
    radar.metadata['dod_version'] = '',
    radar.metadata['comment'] = '',
    radar.metadata['instrument_name'] = radar.metadata['instrument_name'],
    radar.metadata['state'] = '',
    radar.metadata['Conventions'] = 'CF/Radial',
    radar.metadata['reference'] = '',
    radar.metadata['source'] = radar.metadata['source'],
    radar.metadata['platform_type'] = radar.metadata['platform_type'],
    radar.metadata['scan_id'] = radar.metadata['scan_id'],
    radar.metadata['scan_name'] = radar.metadata['scan_name'],
    radar.metadata['input_datastreams_num'] = 1,
    radar.metadata['input_datastreams'] = datastream,
    radar.metadata['input_datastreams_description'] = datastream_description,
    radar.metadata['description'] = '',
    radar.metadata['title'] = 'Corrected Moments in Antenna Coordinates',
    radar.metadata['field_names'] = '',
    radar.metadata['history'] = history,

    return


def _parse_datastreams(filename, dl='a0', version='-9999'):
    """
    """

    split = os.path.basename(filename).split('_')
    date, time = split[0][4:], split[1]
    stream = 'nexradwsr88d{}{}.{}'.format(QF, FN, dl)
    datastream = '{} : {} : {}.{}'.format(stream, version, date, time)

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
        print 'DEALIAS -> {}'.format(DEALIAS)
        print 'INTERVAL_SPLITS -> {}'.format(INTERVAL_SPLITS)

    # Parse all radar files to process
    files = [os.path.join(args.inpdir, f) for f in
             sorted(os.listdir(args.inpdir)) if args.stamp in f]
    if args.verbose:
        print 'Number of files to process: {}'.format(len(files))

    for filename in files:
        process_file(filename, args.outdir, verbose=args.verbose)
