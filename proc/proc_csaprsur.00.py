#!/usr/bin/python

import os
import getpass
import platform
import argparse
import numpy as np

from scipy import ndimage
from datetime import datetime

from pyart.io import read_mdv, write_cfradial
from pyart.config import get_fillvalue, get_field_name
from pyart.correct import dealias_region_based
from pyart.util.datetime_utils import datetimes_from_radar

from echo.correct import noise, basic_fixes
from echo.texture import texture_fields


# Radar descriptors
FN = 'I7'
DL = 'b0'
FACILITY = 'I7: Nardin, Oklahoma'

# Radar significant detection parameters
MIN_NCP = 0.3
TEXTURE_SAMPLE, TEXTURE_WINDOW = 5, (3, 3)
SIZE_BINS, SIZE_LIMITS = 75, (0, 300)
VDOP_TEXT_BINS, VDOP_TEXT_LIMITS = 40, (0, 20)
PHASOR_TEXT_BINS, PHASOR_TEXT_LIMITS = 40, (0, 20)
BOUNDS_PERCENTILE = 98.0
FILL_HOLES = False
DILATE = False
ITERATIONS = 1
STRUCTURE = ndimage.generate_binary_structure(2, 1)

# Velocity correction parameters
DEALIAS = 'region'
INTERVAL_SPLITS = 4

# Define output NetCDF format
FORMAT = 'NETCDF4'

# Define radar fields
REFL_FIELD = get_field_name('reflectivity')
CORR_REFL_FIELD = get_field_name('corrected_reflectivity')
VDOP_FIELD = get_field_name('velocity')
CORR_VDOP_FIELD = get_field_name('corrected_velocity')
WIDTH_FIELD = get_field_name('spectrum_width')
ZDR_FIELD = get_field_name('differential_reflectivity')
PHIDP_FIELD = get_field_name('differential_phase')
RHOHV_FIELD = get_field_name('cross_correlation_ratio')
NCP_FIELD = get_field_name('normalized_coherent_power')
PHASOR_FIELD = '{}_phasor_real'.format(VDOP_FIELD)
REFL_TEXT_FIELD = '{}_texture'.format(REFL_FIELD)
VDOP_TEXT_FIELD = '{}_texture'.format(VDOP_FIELD)
WIDTH_TEXT_FIELD = '{}_texture'.format(WIDTH_FIELD)
RHOHV_TEXT_FIELD = '{}_texture'.format(RHOHV_FIELD)
ZDR_TEXT_FIELD = '{}_texture'.format(ZDR_FIELD)
PHIDP_TEXT_FIELD = '{}_texture'.format(PHIDP_FIELD)
NCP_TEXT_FIELD = '{}_texture'.format(NCP_FIELD)
PHASOR_TEXT_FIELD = '{}_texture'.format(PHASOR_FIELD)

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
    NCP_FIELD,
    CORR_VDOP_FIELD,
    CORR_REFL_FIELD,
    ]

# Define radar texture fields to compute
TEXTURE_FIELDS = [
    REFL_FIELD,
    VDOP_FIELD,
    WIDTH_FIELD,
    PHIDP_FIELD,
    RHOHV_FIELD,
    NCP_FIELD,
    ]

# Define fields to exclude when reading radar file
EXLUDE_FIELDS = [
    'radar_echo_classification',
    'corrected_reflectivity',
    'total_power',
    'corrected_differential_reflectivity',
    'unfolded_differential_phase',
    'specific_differential_phase',
    ]

# Define fields to remove before writing CF/Radial file
REMOVE_FIELDS = [
    PHASOR_FIELD,
    REFL_TEXT_FIELD,
    WIDTH_TEXT_FIELD,
    RHOHV_TEXT_FIELD,
    ZDR_TEXT_FIELD,
    PHIDP_TEXT_FIELD,
    NCP_TEXT_FIELD,
    PHASOR_TEXT_FIELD,
    ]


def process_file(filename, outdir, debug=False, verbose=False):
    """
    """

    if verbose:
        print 'Processing file: {}'.format(os.path.basename(filename))

    # Read radar data
    radar = read_mdv(filename, exclude_fields=EXLUDE_FIELDS)

    if debug:
        print 'Number of sweeps: {}'.format(radar.nsweeps)

    # Step 1: Remove last 7 gates from each ray
    # These gates are reserved for signal testing only
    for field in radar.fields.keys():
        if verbose:
            print 'Removing signal testing gates: {}'.format(field)
        radar.fields[field]['data'][:,-7:] = np.ma.masked

    # Step 2: Radar significant detection
    # Includes Doppler velocity coherency, Doppler velocity phasor coherency,
    # and significant echo boundary detection
    gf = noise.velocity_coherency(
        radar, gatefilter=None, text_bins=VDOP_TEXT_BINS,
        text_limits=VDOP_TEXT_LIMITS, texture_window=TEXTURE_WINDOW,
        texture_sample=TEXTURE_SAMPLE, max_texture=None, nyquist=None,
        rays_wrap_around=False, remove_small_features=False, fill_value=None,
        vdop_field=VDOP_FIELD, text_field=None, coherent_field=None,
        debug=debug, verbose=verbose)
    gf = noise.velocity_phasor_coherency(
        radar, gatefilter=gf, text_bins=PHASOR_TEXT_BINS,
        text_limits=PHASOR_TEXT_LIMITS, texture_window=TEXTURE_WINDOW,
        texture_sample=TEXTURE_SAMPLE, max_texture=None,
        rays_wrap_around=False, remove_small_features=False, fill_value=None,
        vdop_field=VDOP_FIELD, phasor_field=None, text_field=None,
        coherent_field=None, debug=debug, verbose=verbose)
    gf = noise.echo_boundaries(
        radar, gatefilter=gf, texture_window=TEXTURE_WINDOW,
        texture_sample=TEXTURE_SAMPLE, min_texture=None,
        bounds_percentile=BOUNDS_PERCENTILE, remove_small_features=False,
        rays_wrap_around=False, fill_value=None, sqi_field=NCP_FIELD,
        text_field=None, bounds_field=None, debug=debug, verbose=verbose)
    gf = noise.significant_detection(
        radar, gatefilter=gf, remove_small_features=True, size_bins=SIZE_BINS,
        size_limits=SIZE_LIMITS, fill_holes=FILL_HOLES, dilate=DILATE,
        structure=STRUCTURE, iterations=ITERATIONS, min_ncp=MIN_NCP,
        ncp_field=NCP_FIELD, detect_field=None, debug=debug, verbose=verbose)

    # Step 3: Compute radar texture fields
    texture_fields.add_textures(
        radar, fields=TEXTURE_FIELDS, gatefilter=None,
        texture_window=TEXTURE_WINDOW, texture_sample=TEXTURE_SAMPLE,
        min_sweep=None, max_sweep=None, min_range=None, max_range=None,
        min_ncp=None, rays_wrap_around=False, fill_value=None,
        ncp_field=NCP_FIELD)

    # Step 4: Doppler velocity correction
    if DEALIAS.upper() == 'REGION':
        vdop_corr = dealias_region_based(
            radar, gatefilter=gf, interval_splits=INTERVAL_SPLITS,
            interval_limits=None, skip_between_rays=2, skip_along_ray=2,
            centered=True, nyquist_vel=None, rays_wrap_around=True,
            keep_original=False, vel_field=VDOP_FIELD,
            corr_vel_field=CORR_VDOP_FIELD)
    else:
        raise ValueError('Unsupported velocity correction routine')

    radar.add_field(CORR_VDOP_FIELD, vdop_corr, replace_existing=False)

    # TODO
    # Step 5: Reflectivity correction
    refl_corr = radar.fields[REFL_FIELD].copy()
    radar.add_field(CORR_REFL_FIELD, refl_corr, replace_existing=False)

    # Step 6: Interpolate missing gates
    basic_fixes.interpolate_missing(
        radar, fields=FILL_FIELDS, interp_window=FILL_WINDOW,
        interp_sample=FILL_SAMPLE, kind='mean', rays_wrap_around=False,
        fill_value=None, debug=debug, verbose=verbose)

    # Step 7: Remove unwanted fields before writing
    for field in REMOVE_FIELDS:
        if verbose:
            print 'Removing radar field before writing: {}'.format(field)
        radar.fields.pop(field, None)

    # Parse metadata
    _add_metadata(radar, filename)

    # ARM file name protocols
    date = datetimes_from_radar(radar).min().strftime('%Y%m%d.%H%M%S')
    fname = 'sgpcsaprsurcmac{}.{}.{}.cdf'.format(FN, DL, date)

    # Write CMAC NetCDF file
    write_cfradial(os.path.join(outdir, fname), radar, format=FORMAT,
                   arm_time_variables=True)

    return



def _add_metadata(radar, filename):
    """
    """

    # Datastreams attributes
    datastream = _parse_datastreams(radar, filename)
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
    radar.metadata['project'] = ''
    radar.metadata['institution'] = 'McGill University'
    radar.metadata['dod_version'] = ''
    radar.metadata['comment'] = ''
    radar.metadata['instrument_name'] = radar.metadata['instrument_name']
    radar.metadata['state'] = ''
    radar.metadata['Conventions'] = 'CF/Radial'
    radar.metadata['reference'] = ''
    radar.metadata['input_datastreams_num'] = 1
    radar.metadata['input_datastreams'] = datastream
    radar.metadata['input_datastreams_description'] = datastream_description
    radar.metadata['description'] = ''
    radar.metadata['title'] = 'Corrected Moments in Antenna Coordinates'
    radar.metadata['field_names'] = ''
    radar.metadata['history'] = history

    return


def _parse_datastreams(radar, filename, version='-9999'):
    """
    """

    time = os.path.basename(filename).split('.')[0]
    date = datetimes_from_radar(radar).min().date().strftime('%Y%m%d')
    datastream = 'sgpcsaprsur{}.00 : {} : {}.{}'.format(
        FN, version, date, time)

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
        print 'FN -> {}'.format(FN)
        print 'DL -> {}'.format(DL)
        print 'FACILITY -> {}'.format(FACILITY)
        print 'DEALIAS -> {}'.format(DEALIAS)
        print 'MIN_NCP -> {}'.format(MIN_NCP)
        print 'INTERVAL_SPLITS -> {}'.format(INTERVAL_SPLITS)
        print 'TEXTURE_SAMPLE -> {}'.format(TEXTURE_SAMPLE)
        print 'TEXTURE_WINDOW -> {}'.format(TEXTURE_WINDOW)
        print 'VDOP_TEXT_BINS -> {}'.format(VDOP_TEXT_BINS)
        print 'VDOP_TEXT_LIMITS -> {}'.format(VDOP_TEXT_LIMITS)
        print 'PHASOR_TEXT_BINS -> {}'.format(PHASOR_TEXT_BINS)
        print 'PHASOR_TEXT_LIMITS -> {}'.format(PHASOR_TEXT_LIMITS)
        print 'BOUNDS_PERCENTILE -> {}'.format(BOUNDS_PERCENTILE)
        print 'SIZE_BINS -> {}'.format(SIZE_BINS)
        print 'SIZE_LIMITS -> {}'.format(SIZE_LIMITS)
        print 'DILATE -> {}'.format(DILATE)
        print 'ITERATIONS -> {}'.format(ITERATIONS)
        print 'FILL_SAMPLE -> {}'.format(FILL_SAMPLE)
        print 'FILL_WINDOW -> {}'.format(FILL_WINDOW)

    # Parse all radar files to process
    files = [os.path.join(args.inpdir, f) for f in
             sorted(os.listdir(args.inpdir)) if args.stamp in f]

    if args.verbose:
        print 'Number of files to process = {}'.format(len(files))

    for filename in files:

        # Process file
        process_file(
            filename, args.outdir, debug=args.debug, verbose=args.verbose)
