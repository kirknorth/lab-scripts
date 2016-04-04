#!/usr/bin/python

import os
import getpass
import platform
import argparse
import numpy as np

from datetime import datetime

from pyart.io import write_cfradial
from pyart.aux_io import read_kazr
from pyart.config import get_fillvalue, get_field_name
from pyart.correct import dealias_unwrap_phase, dealias_region_based
from pyart.util.datetime_utils import datetime_from_radar

from echo.correct import noise

# Basic radar thresholds
MIN_NCP = None

# Coherency and texture parameters
TEXTURE_SAMPLE, TEXTURE_WINDOW = 5, (3, 3)
SALT_SAMPLE, SALT_WINDOW = 15, (5, 5)
VDOP_COHER_BINS, VDOP_COHER_LIMITS = 100, (0, 8)

# Doppler velocity correction routine
DEALIAS = 'region'
INTERVAL_SPLITS = 5

# Define output NetCDF format
FORMAT = 'NETCDF4'

# Define radar field names
REFL_FIELD = 'reflectivity_copol'
VDOP_FIELD = 'mean_doppler_velocity_copol'
CORR_VDOP_FIELD = 'corrected_mean_doppler_velocity_copol'
SNR_FIELD = 'signal_to_noise_ratio_copol'
WIDTH_FIELD = 'spectral_width_copol'
POWER_FIELD = 'signal_to_noise_ratio_copol'


def process_file(filename, outdir, dl='b1', verbose=False):
    """
    """

    if verbose:
        print 'Processing file: {}'.format(os.path.basename(filename))

    # Read radar data
    radar = read_kazr(filename, exclude_fields=None)

    # Step 1: Radar significant detection
    # Includes Hildebrand noise floor estimate and Doppler velocity coherency
    gf = noise.velocity_coherency(
        radar, gatefilter=None, num_bins=VDOP_COHER_BINS,
        limits=VDOP_COHER_LIMITS, texture_window=TEXTURE_WINDOW,
        texture_sample=TEXTURE_SAMPLE, min_sigma=None, max_sigma=None,
        nyquist=None, rays_wrap_around=None, remove_salt=False,
        fill_value=None, vdop_field=VDOP_FIELD, vdop_text_field=None,
        cohere_field=None, verbose=verbose)
    gf = noise.hildebrand_noise(
        radar, gatefilter=gf, scale=1.0, remove_salt=False,
        rays_wrap_around=False, fill_value=None, power_field=POWER_FIELD,
        noise_field=None, verbose=verbose)
    gf = noise.significant_detection(
        radar, gatefilter=gf, min_ncp=None, remove_salt=True,
        salt_window=SALT_WINDOW, salt_sample=SALT_SAMPLE, fill_holes=False,
        dilate=False, structure=None, rays_wrap_around=False, ncp_field=None,
        detect_field=None, verbose=verbose)

    # Step 2: Doppler velocity correction
    if DEALIAS == 'phase':
        vdop_corr = dealias_unwrap_phase(
            radar, gatefilter=gf, unwrap_unit='sweep', nyquist_vel=None,
            rays_wrap_around=False, keep_original=False, skip_checks=True,
            vel_field=VDOP_FIELD, corr_vel_field=None)

    elif DEALIAS == 'region':
        vdop_corr = dealias_region_based(
            radar, gatefilter=gf, interval_splits=INTERVAL_SPLITS,
            interval_limits=None, skip_between_rays=2, skip_along_ray=2,
            centered=True, nyquist_vel=None, rays_wrap_around=False,
            keep_original=False, vel_field=VDOP_FIELD, corr_vel_field=None)

    else:
        raise ValueError('Unsupported velocity correction routine')

    radar.add_field(CORR_VDOP_FIELD, vdop_corr, replace_existing=True)

    # TODO
    # Step 3: Reflectivity correction

    # Parse metadata
    radar.metadata = _create_metadata(radar, filename)

    # ARM file name protocols
    date = datetime_from_radar(radar).strftime('%Y%m%d.%H%M%S')
    filename = 'sgpkazrgecmacC1.{}.{}.cdf'.format(dl, date)

    # Write CMAC NetCDF file
    write_cfradial(os.path.join(outdir, filename), radar, format=FORMAT,
                   arm_time_variables=True)

    return


def _create_metadata(radar, filename):
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

    metadata = {
        'process_version': '',
        'command_line': '',
        'site_id': 'sgp',
        'facility_id': 'C1: Lamont, Oklahoma',
        'country': 'USA',
        'project': '',
        'institution': 'McGill University',
        'dod_version': '',
        'comment': '',
        'instrument_name': radar.metadata['comment'],
        'state': '',
        'Conventions': 'CF/Radial',
        'reference': '',
        'input_datastreams_num': 1,
        'input_datastreams': datastream,
        'input_datastreams_description': datastream_description,
        'description': '',
        'title': 'Corrected Moments in Antenna Coordinates',
        'field_names': '',
        'history': history
    }

    return metadata


def _parse_datastreams(filename, version='-9999'):
    """
    """

    stream, dl, date, time = os.path.basename(filename).split('.')[:4]
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

    if args.debug:
        print 'MIN_NCP -> {}'.format(MIN_NCP)
        print 'DEALIAS -> {}'.format(DEALIAS)
        print 'INTERVAL_SPLITS -> {}'.format(INTERVAL_SPLITS)
        print 'TEXTURE_SAMPLE -> {}'.format(TEXTURE_SAMPLE)
        print 'TEXTURE_WINDOW -> {}'.format(TEXTURE_WINDOW)

    # Parse all radar files to process
    files = [os.path.join(args.inpdir, f) for f in
             sorted(os.listdir(args.inpdir)) if args.stamp in f]
    if args.verbose:
        print 'Number of files to process = {}'.format(len(files))

    for filename in files:
        process_file(filename, args.outdir, verbose=args.verbose)
