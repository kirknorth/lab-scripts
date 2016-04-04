#!/usr/bin/python

import os
import getpass
import platform
import argparse
import numpy as np

from datetime import datetime

from pyart.io import read_sigmet, write_cfradial
from pyart.config import get_fillvalue, get_field_name
from pyart.correct import dealias_region_based
from pyart.util.datetime_utils import datetime_from_radar

from echo.correct import noise
from echo.texture import texture_fields
from echo.membership import bayes


# Radar descriptors
FN = 'I4'
DL = 'b1'
FACILITY = 'I4: Billings, Oklahoma'

# Basic radar thresholds
MIN_NCP = 0.3
DILATE = True

# Echo classification parameters
ZERO = 1.0e-5
IGNORE_INPUTS = None
CLOUD_FIELD = 'cloud'
GROUND_FIELD = 'ground'
INSECT_FIELD = 'insect'

# Echo classification calibration data
INPDIR = '~/projects/echo-class/echo/calibration/'
MOMENTS =
TEXTURES =
HEIGHTS = None

# Coherency and texture parameters
TEXTURE_SAMPLE, TEXTURE_WINDOW= 5, (3, 3)
SALT_SAMPLE, SALT_WINDOW = 10, (5, 5)
VDOP_COHER_BINS, VDOP_COHER_LIMITS = 100, (0, 20)
PHASE_COHER_BINS, PHASE_COHER_LIMITS = 100, (0, 0.03)
SW_COHER_BINS, SW_COHER_LIMITS = 50, (0, 5)

# Define output NetCDF format
FORMAT = 'NETCDF4'

# Define radar fields
REFL_FIELD = get_field_name('reflectivity')
VDOP_FIELD = get_field_name('velocity')
CORR_VDOP_FIELD = get_field_name('corrected_velocity')
ZDR_FIELD = get_field_name('differential_reflectivity')
PHIDP_FIELD = get_field_name('differential_phase')
RHOHV_FIELD = get_field_name('cross_correlation_ratio')
NCP_FIELD = get_field_name('normalized_coherent_power')

# Define radar texture fields
TEXTURE_FIELDS = [
    REFL_FIELD,
    VDOP_FIELD,
    WIDTH_FIELD,
    PHIDP_FIELD,
    RHOHV_FIELD,
    NCP_FIELD,
    ]

# Define fields to exclude when reading  radar file
EXLUDE_FIELDS = [
    'radar_echo_classification',
    'corrected_reflectivity',
    'total_power',
    'corrected_differential_reflectivity',
    'unfolded_differential_phase',
    ]


def process_file(filename, outdir, verbose=False):
    """
    """

    if verbose:
        print 'Processing file: {}'.format(os.path.basename(filename))

    # Read radar data
    radar = read_sigmet(filename, exclude_fields=EXLUDE_FIELDS)

    # Radar significant detection
    # Includes Doppler velocity coherency, spectrum width coherency, and
    # minimum normalized coherent power
    gf = noise.velocity_coherency(
        radar, gatefilter=None, num_bins=VDOP_COHER_BINS,
        limits=VDOP_COHER_LIMITS, texture_window=TEXTURE_WINDOW,
        texture_sample=TEXTURE_SAMPLE, min_sigma=None, max_sigma=None,
        rays_wrap_around=False, remove_salt=False, fill_value=None,
        vdop_field=VDOP_FIELD, vdop_text_field=None, cohere_field=None,
        verbose=verbose)
    gf = noise.velocity_phasor_coherency(
        radar, gatefilter=gf, num_bins=PHASE_COHER_BINS,
        limits=PHASE_COHER_LIMITS, texture_window=TEXTURE_WINDOW,
        texture_sample=TEXTURE_SAMPLE, min_sigma=None, max_sigma=None,
        rays_wrap_around=False, remove_salt=False, fill_value=None,
        vdop_field=VDOP_FIELD, vdop_phase_field=None, phase_text_field=None,
        cohere_field=None, verbose=verbose)
    gf = noise.spectrum_width_coherency(
        radar, gatefilter=gf, num_bins=SW_COHER_BINS,
        limits=SW_COHER_LIMITS, texture_window=TEXTURE_WINDOW,
        texture_sample=TEXTURE_SAMPLE, min_sigma=None, max_sigma=None,
        rays_wrap_around=False, remove_salt=False, fill_value=None,
        width_field=SW_FIELD, width_text_field=None, cohere_field=None,
        verbose=verbose)
    gf = noise.significant_detection(
        radar, gatefilter=gf, remove_salt=True, salt_window=SALT_WINDOW,
        salt_sample=SALT_SAMPLE, fill_holes=False, dilate=DILATE,
        structure=None, min_ncp=MIN_NCP, ncp_field=NCP_FIELD,
        detect_field=None, verbose=verbose)

    # Compute radar texture fields
    texture_fields.add_textures(
        radar, fields=TEXTURE_FIELDS, gatefilter=None,
        texture_window=TEXTURE_WINDOW, texture_sample=TEXTURE_SAMPLE,
        min_sweep=None, max_sweep=None, min_range=None, max_range=None,
        min_ncp=None, rays_wrap_around=False, fill_value=None,
        ncp_field=NCP_FIELD)

    # Echo classification
    bayes.classify(
        radar, textures=TEXTURES, moments=MOMENTS, heights=HEIGHTS,
        nonprecip_map=None, gatefilter=gf, weights=1.0, class_prob='equal',
        min_inputs=3, zero=ZERO, ignore_inputs=IGNORE_INPUTS, use_insects=True,
        fill_value=None, cloud_field=CLOUD_FIELD, ground_field=GROUND_FIELD,
        insect_field=INSECT_FIELD, ncp_field=NCP_FIELD, verbose=verbose)

    # Filter ground clutter gates
    gf.exclude_equal(
        'radar_echo_classification', 1, exclude_masked=True, op='or')

    # Doppler velocity correction
    vdop_corr = dealias_region_based(
        radar, gatefilter=gf, interval_splits=3, interval_limits=None,
        skip_between_rays=2, skip_along_ray=2, centered=True, nyquist_vel=None,
        rays_wrap_around=True, keep_original=False, vel_field=VDOP_FIELD,
        corr_vel_field=CORR_VDOP_FIELD)
    radar.add_field(CORR_VDOP_FIELD, vdop_corr, replace_existing=False)

    # TODO: reflectivity correction

    # Parse metadata
    radar.metadata = _create_metadata(radar, filename)

    # ARM file name protocols
    date = datetime_from_radar(radar).strftime('%Y%m%d.%H%M%S')
    filename = 'sgpxsaprppicmac{}.{}.{}.cdf'.format(FN, DL, date)

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
        'facility_id': FACILITY,
        'country': 'USA',
        'project': '',
        'institution': 'McGill University',
        'dod_version': '',
        'comment': '',
        'instrument_name': radar.metadata['instrument_name'],
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

    date = datetime.strptime(os.path.basename(filename)[3:15], '%y%m%d%H%M%S')
    date = date = date.strftime('%Y%m%d.%H%M%S')
    datastream = 'sgpxsaprppi{}.00 : {} : {}'.format(FN, version, date)

    return datastream


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('inpdir', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('outdir', type=str, help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, default=False,
                        const=True, help=None)
    args = parser.parse_args()

    # Parse all radar files to process
    files = [os.path.join(args.inpdir, f) for f in
             sorted(os.listdir(args.inpdir)) if args.stamp in f]
    if args.verbose:
        print 'Number of files to process = {}'.format(len(files))

    for filename in files:
        process_file(filename, args.outdir, verbose=args.verbose)
