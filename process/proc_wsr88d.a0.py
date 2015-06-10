#!/usr/bin/python

import os
import platform
import getpass
import argparse
import numpy as np

from netCDF4 import num2date
from datetime import datetime
from scipy import ndimage

from pyart.io import read, write_cfradial
from pyart.aux_io import read_radx
from pyart.util.datetime_utils import datetimes_from_radar
from pyart.correct import dealias_region_based, GateFilter
from pyart.config import get_fillvalue, get_field_name, FileMetadata

from echo.correct import noise, basic_fixes


# Radar descriptors
QF = 'V06'
FN = 'KVNX'
DL = 'b0'
FACILITY = 'KVNX: Vance AFB, Oklahoma'

# Define read function
# If USE_RADX is False, Py-ART's generic read function will be used
USE_RADX = False

# Radar significant detection parameters
SIZE_BINS, SIZE_LIMITS = 75, (0, 300)
FILL_HOLES = False
DILATE = False
ITERATIONS = 1
STRUCTURE = ndimage.generate_binary_structure(2, 1)

# Define radar field names
REFL_FIELD = get_field_name('reflectivity')
REFL_CORR_FIELD = get_field_name('corrected_reflectivity')
VDOP_FIELD = get_field_name('velocity')
VDOP_CORR_FIELD = get_field_name('corrected_velocity')
WIDTH_FIELD = get_field_name('spectrum_width')
RHOHV_FIELD = get_field_name('cross_correlation_ratio')
ZDR_FIELD = get_field_name('differential_reflectivity')
PHIDP_FIELD = get_field_name('differential_phase')

# Define number of sweeps for VCP
CHECK_VCP = False
NSWEEPS = 14

# Doppler velocity correction routine and parameters
DEALIAS = 'region'
INTERVAL_SPLITS = 4

# Define missing gate interpolation parameters
FILL_WINDOW = (3, 3)
FILL_SAMPLE = 7

# Define radar fields to interpolate
FILL_FIELDS = [
    REFL_FIELD,
    WIDTH_FIELD,
    RHOHV_FIELD,
    ZDR_FIELD,
    PHIDP_FIELD,
    REFL_CORR_FIELD,
    VDOP_CORR_FIELD,
    ]

# Define output NetCDF format
FORMAT = 'NETCDF4'

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
    if USE_RADX:
        radar = read_radx(filename)
    else:
        radar = read(filename, exclude_fields=None)

    # Radar VCP check
    if CHECK_VCP:
        if NSWEEPS is not None and radar.nsweeps != NSWEEPS:
            return

    if verbose:
        print 'Processing file: {}'.format(os.path.basename(filename))

    if debug:
        print 'Number of sweeps: {}'.format(radar.nsweeps)

    if USE_RADX:
        # Create file metadata object
        meta = FileMetadata(
            'nexrad_archive', field_names=None, additional_metadata=None,
            file_field_names=False, exclude_fields=None)

        # Remove unnecessary fields
        for field in REMOVE_FIELDS:
            radar.fields.pop(field, None)

        # Rename fields to default Py-ART names
        for field in radar.fields.keys():
            default_field = meta.get_field_name(field)
            radar.fields[default_field] = radar.fields.pop(field, None)

    # Step 1: Determine radar significant detection
    # Since NEXRAD WSR-88D Level II data is already processed to some degree,
    # this amounts to essentially removing salt and pepper noise
    gf = noise._significant_features(
        radar, REFL_FIELD, gatefilter=None, size_bins=SIZE_BINS,
        size_limits=SIZE_LIMITS, structure=STRUCTURE, remove_size_field=False,
        fill_value=None, size_field=None, debug=debug)
    gf = noise.significant_detection(
        radar, gatefilter=gf, remove_small_features=False, size_bins=SIZE_BINS,
        size_limits=SIZE_LIMITS, fill_holes=FILL_HOLES, dilate=DILATE,
        structure=STRUCTURE, iterations=1, rays_wrap_around=False,
        min_ncp=None, detect_field=None, debug=debug, verbose=verbose)

    # Step 2: Doppler velocity correction
    if DEALIAS == 'phase':
        vdop_corr = dealias_unwrap_phase(
            radar, gatefilter=gf, unwrap_unit='sweep', nyquist_vel=None,
            rays_wrap_around=True, keep_original=False, vel_field=None,
            corr_vel_field=VDOP_CORR_FIELD)
    elif DEALIAS == 'region':
        vdop_corr = dealias_region_based(
            radar, gatefilter=gf, interval_splits=INTERVAL_SPLITS,
            interval_limits=None, skip_between_rays=2, skip_along_ray=2,
            centered=True, nyquist_vel=None, rays_wrap_around=True,
            keep_original=False, vel_field=None,
            corr_vel_field=VDOP_CORR_FIELD)
    else:
        raise ValueError('Unsupported velocity correction routine')

    radar.add_field(VDOP_CORR_FIELD, vdop_corr, replace_existing=True)

    # Step 3: Reflectivity correction
    # Currently no correction procedures are applied to the reflectivity field
    # due to minimal attenuation at S-band
    refl_corr = radar.fields[REFL_FIELD].copy()
    radar.add_field(REFL_CORR_FIELD, refl_corr, replace_existing=True)

    # Step 4: Interpolate missing gates
    basic_fixes.interpolate_missing(
        radar, fields=FILL_FIELDS, interp_window=FILL_WINDOW,
        interp_sample=FILL_SAMPLE, kind='mean', rays_wrap_around=False,
        fill_value=None, debug=debug, verbose=verbose)

    # Add metadata
    _add_metadata(radar, filename)

    # ARM file name protocols
    date_stamp = datetimes_from_radar(radar).min().strftime('%Y%m%d.%H%M%S')
    fname = 'nexradwsr88d{}cmac{}.{}.{}.cdf'.format(QF, FN, DL, date_stamp)

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
    radar.metadata['process_version'] = ''
    radar.metadata['command_line'] = ''
    radar.metadata['site_id'] = ''
    radar.metadata['facility_id'] = FACILITY
    radar.metadata['country'] = 'USA'
    radar.metadata['project'] = '',
    radar.metadata['institution'] = 'McGill University'
    radar.metadata['dod_version'] = ''
    radar.metadata['comment'] = ''
    radar.metadata['instrument_name'] = FN
    radar.metadata['state'] = ''
    radar.metadata['Conventions'] = 'CF/Radial'
    radar.metadata['reference'] = ''
    radar.metadata['source'] = radar.metadata['source']
    radar.metadata['input_datastreams_num'] = 1
    radar.metadata['input_datastreams'] = datastream
    radar.metadata['input_datastreams_description'] = datastream_description
    radar.metadata['description'] = ''
    radar.metadata['title'] = 'Corrected Moments in Antenna Coordinates'
    radar.metadata['field_names'] = ''
    radar.metadata['history'] = history

    if USE_RADX:
        radar.metadata['platform_type'] = radar.metadata['platform_type']
        radar.metadata['scan_id'] = radar.metadata['scan_id']
        radar.metadata['scan_name'] = radar.metadata['scan_name']
        radar.metadata['instrument_name'] = radar.metadata['instrument_name']

    return


def _parse_datastreams(filename, dl='a0', version='-9999'):
    """
    """

    split = os.path.basename(filename).split('_')
    date, time = split[0][4:], split[1]
    stream = '{}'.format(os.path.basename(filename))
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
        print 'USE_RADX -> {}'.format(USE_RADX)
        print 'DEALIAS -> {}'.format(DEALIAS)
        print 'INTERVAL_SPLITS -> {}'.format(INTERVAL_SPLITS)
        print 'SIZE_BINS -> {}'.format(SIZE_BINS)
        print 'SIZE_LIMITS -> {}'.format(SIZE_LIMITS)
        print 'FILL_HOLES -> {}'.format(FILL_HOLES)
        print 'DILATE -> {}'.format(DILATE)
        print 'ITERATIONS -> {}'.format(ITERATIONS)
        print 'FILL_SAMPLE -> {}'.format(FILL_SAMPLE)
        print 'FILL_WINDOW -> {}'.format(FILL_WINDOW)

    # Parse all radar files to process
    files = [os.path.join(args.inpdir, f) for f in
             sorted(os.listdir(args.inpdir)) if args.stamp in f]
    if args.verbose:
        print 'Number of files to process: {}'.format(len(files))

    for filename in files:
        process_file(
            filename, args.outdir, debug=args.debug, verbose=args.verbose)
