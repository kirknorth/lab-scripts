#!/usr/bin/python

import argparse
import numpy as np

from netCDF4 import Dataset

from pyart.io import read_grid
from pyart.retrieve import winds


# Define default constraint weights
lambda_b = np.concatenate((np.arange(0, 1, 0.05),
                          np.arange(1, 10.5, 0.5)), axis=0)
lambda_c = np.concatenate((np.arange(0, 1, 0.2)), axis=0)


def _sensitivity_data(grids, wgt1, wgt2):
    """
    """

    nw1, nw2 = wgt1.size, wgt2.size
    nz, ny, nx = grids[0].fields['corrected_velocity']['data'].shape

    data = {
        'Retrieved u profile': np.ma.empty((nz, nw1, nw2)),
        'Retrieved v profile': np.ma.empty((nz, nw1, nw2)),
        'Continuity residual': np.ma.empty((nw1, nw2)),
        'Continuity residual profile': np.ma.empty((nz, nw1, nw2)),
        ''
    }

    return data



def background_continuity(grids, verbose=False):
    """
    """

    if verbose:
        print 'Testing sensitivity of the continuity-background constraints'

    data = {}
    for i, j in np.ndindex(lambda_b.size, lambda_c.size):
        wgt_b = lambda_b[i]
        wgt_c = lambda_c[j]
        if verbose:
            print 'lambda_b = {:.3f}, lambda_c = {:.3f}'.format(wgt_b, wgt_c)

        conv = winds.solve_wind_field(
            grids, sonde=sonde, target=None, technique='3d-var',
            solver='scipy.fmin_cg', first_guess='zero', background='sounding',
            sounding='ARM interpolated', impermeability='weak',
            fall_speed='Caya', finite_scheme='basic', use_qc=False,
            standard_density=True, continuity_cost='Potvin',
            smooth_cost='Potvin', wgt_o=1.0, wgt_c=wgt_c, wgt_s1=1.0,
            wgt_s2=1.0, wgt_s3=1.0, wgt_s4=0.1, wgt_w0=100.0,
            wgt_ub=wgt_b, wgt_vb=wgt_b, wgt_wb=0.0, length_scale=250.0,
            first_pass=True, save_refl=False, save_hdiv=False, maxiter=200,
            maxiter_first_pass=50, gtol=1.0e-1, disp=False, debug=False,
            verbose=False)

        metrics = winds._check_analysis(
            grids, conv, sonde=sonde, standard_density=True,
            finite_scheme='basic', verbose=False)

        # Parse velocity data
        u = conv.fields['eastward_wind_component']['data']
        v = conv.fields['northward_wind_component']['data']
        w = conv.fields['vertical_wind_component']['data']

        # Compute mean retrieved horizontal wind profile
        data{'Retrieved u profile'}





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-i', '--input', nargs='*', type=str, help=None)
    parser.add_argument('-o', '--output', type=str, help=None)
    parser.add_argument('--sens', type=str, help=None)
    parser.add_argument('--sonde', type=str, help=None)
    parser.add_argument('-v', '--verbose', nargs='?', const=True,
                        default=False, type=bool, help=None)
    args = parser.parse_args()

    # Read input grids
    grids = [read_grid(f) for f in args.input]
    if verbose:
        print 'Number of input grids = %i' % len(grids)
