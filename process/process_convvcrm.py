#!/usr/bin/python

import os
import platform
import getpass
import argparse
import json
import numpy as np

from datetime import datetime
from netCDF4 import Dataset, num2date

from pyart.io import read_grid
from pyart.retrieve import winds


### GLOBAL VARIABLES ###
########################

# Define common input directories
INPUT_DIRS = ['/storage/kirk/MC3E/processed_data/csapr/',
              '/storage/kirk/MC3E/processed_data/xsapr/',
              '/storage/kirk/MC3E/processed_data/nexrad/']

# Define output paramters
FORMAT = 'NETCDF4_CLASSIC'

# Whether to process single groups
REMOVE_SINGLE = True

# Define 3D-VAR algorithm parameters
CONTINUITY_COST = 'Potvin'
SMOOTH_COST = 'Potvin'
IMPERMEABILITY = 'weak'
WGT_O = 1.0
WGT_O_MIN = 0.01
WGT_C = 500.0
WGT_S1 = 1.0
WGT_S2 = 1.0
WGT_S3 = 1.0
WGT_S4 = 0.1
WGT_UB = 0.01
WGT_VB = 0.01
WGT_WB = 0.0
WGT_W0 = 100.0
LENGTH_SCALE = 250.0
MAXITER = 500
MAXITER_FIRST_PASS = 50
OBS_WGT_FUNC = 'gate_distance'
USE_QC = False

# Computing radar network reflectivity
IGNORE = ['xsapr-sgpr1', 'xsapr-sgpr2', 'xsapr-sgpr3']
MIN_REFL = -10.0
MAX_REFL = 85.0

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


def create_metadata(files, facility_id):
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
        'facility_id': facility_id,
        'country': 'USA',
        'project': 'MC3E',
        'institution': 'ARM Climate Research Facility',
        'dod_version': '',
        'comment': '',
        'state': '',
        'Conventions': 'CF/Radial',
        'reference': '',
        'input_datastreams_num': num_datastreams,
        'input_datastreams': datastreams,
        'input_datastreams_description': datastream_description,
        'description': '',
        'title': 'Convective Vertical Velocity',
        'field_names': '',
        'history': history}

    return metadata


def process(group, fsonde, outdir, qualifier, Fn, dl, facility_id,
            debug=False):
    """
    """

    # Read radar grids
    grids = [read_grid(f) for f in group]

    if debug:
        print 'Number of grids used in retrieval = %i' % len(grids)

    # Read sounding file
    sonde = Dataset(fsonde, mode='r')

    # Compute the radar network reflectivity
    winds._network_reflectivity(
        grids, smooth=False, sigma=0.5, ignore=IGNORE, min_refl=MIN_REFL,
        max_refl=MAX_REFL, refl_field=None)

    # Grid radar data
    conv = winds.solve_wind_field(
        grids, sonde=sonde, target=None, technique='3d-var',
        solver='scipy.fmin_cg', first_guess='zero', background='sounding',
        sounding='ARM interpolated', fall_speed='Caya', finite_scheme='basic',
        continuity_cost=CONTINUITY_COST, smooth_cost=SMOOTH_COST,
        impermeability=IMPERMEABILITY, obs_wgt_func=OBS_WGT_FUNC,
        first_pass=True, sub_beam=False, wgt_o=WGT_O, wgt_o_min=WGT_O_MIN,
        wgt_c=WGT_C, wgt_s1=WGT_S1, wgt_s2=WGT_S2, wgt_s3=WGT_S3,
        wgt_s4=WGT_S4, wgt_ub=WGT_UB, wgt_vb=WGT_VB, wgt_wb=WGT_WB,
        wgt_w0=WGT_W0, length_scale=LENGTH_SCALE, use_qc=USE_QC,
        use_morphology=True, structure=None, mds=0.0, ncp_min=0.2, rhv_min=0.6,
        standard_density=True, maxiter=MAXITER,
        maxiter_first_pass=MAXITER_FIRST_PASS, gtol=1.0e-1, save_refl=True,
        save_hdiv=True, disp=True, refl_field=None, vel_field=None,
        debug=False, verbose=False)

    # Parse metadata
    conv.metadata = create_metadata(group, facility_id)

    # ARM file name protocols
    date_stamp = num2date(conv.axes['time_start']['data'][0],
                          conv.axes['time_start']['units'])
    filename = 'sgp{}convvcrm{}.{}.{}.cdf'.format(
        qualifier, Fn, dl, date_stamp.strftime('%Y%m%d.%H%M%S'))

    # Write gridded data to file
    conv.write(os.path.join(outdir, filename), format=FORMAT)

    return


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('fjson', type=str, help=None)
    parser.add_argument('fsonde', type=str, help=None)
    parser.add_argument('outdir', type=str, help=None)
    parser.add_argument('qualifier', type=str, help=None)
    parser.add_argument('Fn', type=str, help=None)
    parser.add_argument('dl', type=str, help=None)
    parser.add_argument('facility_id', type=str, help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, const=True,
                        default=False, help=None)
    parser.add_argument('-db', '--debug', nargs='?', type=bool, const=True,
                        default=False, help=None)
    args = parser.parse_args()

    if args.debug:
        print 'fjson = %s' % args.fjson
        print 'fsonde = %s' % args.fsonde
        print 'outdir = %s' % args.outdir
        print 'qualifier = %s' % args.qualifier
        print 'Fn = %s' % args.Fn
        print 'dl = %s' % args.dl
        print 'facility_id = %s' % args.facility_id

    # Parse all files to process
    with open(args.fjson, 'r') as fid:
        groups = json.load(fid)
    if args.verbose:
        print 'Number of groups to process = %i' % len(groups)

    if REMOVE_SINGLE:
        groups = [group for group in groups if len(group) > 1]
        if args.verbose:
            print ('Number of groups to process after removing '
                   'singles = %i' % len(groups))

    for group in groups:
        if args.verbose:
            print 'Processing group %s' % group

        # Match directories with files
        for i, filename in enumerate(group):
            for inpdir in INPUT_DIRS:
                if os.path.isfile(os.path.join(inpdir, filename)):
                    group[i] = os.path.join(inpdir, filename)
                    break

        process(group, args.fsonde, args.outdir, args.qualifier, args.Fn,
                args.dl, args.facility_id, debug=args.debug)
