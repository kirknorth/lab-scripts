#!/usr/bin/python

import os
import platform
import getpass
import argparse
import numpy as np

from datetime import datetime
from netCDF4 import Dataset, num2date

from pyart.io import read_grid
from pyart.retrieve import winds


### GLOBAL VARIABLES ###
########################

# Define output paramters
FORMAT = 'NETCDF4_CLASSIC'

# Define 3D-VAR algorithm parameters
WGT_O = 1.0
WGT_C = 500.0
WGT_S1 = 1.0
WGT_S2 = 1.0
WGT_S3 = 1.0
WGT_S4 = 0.1
WGT_UB = 0.05
WGT_VB = 0.05
WGT_WB = 0.0
WGT_W0 = 100.0
LENGTH_SCALE = 250.0
MAXITER = 300
MAXITER_FIRST_PASS = 50


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


def process(fmmcg, fsonde, output, Fn, dl, facility_id, debug=False):
    """
    """

    # Read radar file
    grid = read_grid(fmmcg)

    # Read sounding file
    sonde = Dataset(fsonde, mode='r')

    # Grid radar data
    conv = winds.solve_wind_field(
        [grid], sonde=sonde, target=None, technique='3d-var',
        solver='scipy.fmin_cg', first_guess='zero', background='sounding',
        sounding='ARM interpolated', fall_speed='Caya', finite_scheme='basic',
        continuity_cost='Potvin', smooth_cost='Potvin', impermeability='weak',
        first_pass=True, sub_beam=False, wgt_o=WGT_O, wgt_c=WGT_C,
        wgt_s1=WGT_S1, wgt_s2=WGT_S2, wgt_s3=WGT_S3, wgt_s4=WGT_S4,
        wgt_ub=WGT_UB, wgt_vb=WGT_VB, wgt_wb=WGT_WB, wgt_w0=WGT_W0,
        length_scale=LENGTH_SCALE, use_qc=True, use_morphology=True,
        structure=None, mds=0.0, ncp_min=0.2, rhv_min=0.6,
        standard_density=True, maxiter=MAXITER,
        maxiter_first_pass=MAXITER_FIRST_PASS, gtol=1.0e-1, disp=True,
        debug=False, verbose=False)

    # Parse metadata
    conv.metadata = create_metadata([fmmcg], facility_id)

    # ARM file name protocols
    date_stamp = num2date(conv.axes['time_start']['data'][0],
                          conv.axes['time_start']['units'])
    filename = 'sgpsdconvvcrm{}.{}.{}.cdf'.format(
        Fn, dl, date_stamp.strftime('%Y%m%d.%H%M%S'))

    # Write gridded data to file
    conv.write(os.path.join(output, filename), format=FORMAT)

    return


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('source', type=str, help=None)
    parser.add_argument('output', type=str, help=None)
    parser.add_argument('stamp', type=str, help=None)
    parser.add_argument('sonde', type=str, help=None)
    parser.add_argument('--Fn', nargs='?', type=str, default='C1', help=None)
    parser.add_argument('--dl', nargs='?', type=str, default='c1', help=None)
    parser.add_argument('--facility', nargs='?', type=str,
                        default='C1: Lamont, Oklahoma', help=None)
    parser.add_argument('-v', '--verbose', nargs='?', type=bool, const=True,
                        default=False, help=None)
    parser.add_argument('--debug', nargs='?', type=bool, const=True,
                        default=False, help=None)
    args = parser.parse_args()

    if args.debug:
        print 'source = %s' % args.source
        print 'output = %s' % args.output
        print 'stamp = %s' % args.stamp
        print 'sonde = %s' % args.sonde
        print 'Fn = %s' % args.Fn
        print 'dl = %s' % args.dl
        print 'facility = %s' % args.facility

    # Parse all files to process
    files = [os.path.join(args.source, f) for f in
             sorted(os.listdir(args.source)) if args.stamp in f]
    if args.verbose:
        print 'Number of files to process = %i' % len(files)

    for fmmcg in files:
        if args.verbose:
            print 'Processing file %s' % os.path.basename(fmmcg)

        process(fmmcg, args.sonde, args.output, args.Fn, args.dl,
                args.facility, debug=args.debug)
