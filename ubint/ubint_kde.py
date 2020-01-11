#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: shirui <shirui816@gmail.com>

import re
import warnings
import argparse
from scipy.constants import Boltzmann as KB
from scipy.constants import Avogadro as NA
import numpy as np
import pandas as pd
from argparse import RawTextHelpFormatter
from scipy.integrate import simps, quad
from scipy.stats import kstat, gaussian_kde
from scipy.optimize import curve_fit


# TODO: use gaussian_kde and fit exp(-(\sum_i a_i x_i^(i-1)); periodic data
# the numerical stability of exp(-(\sum_i a_i x_i^(i-1)) is the key, make sure
# that fitting range is large enough so that mean force does not diverge.

description = """An Umbrella Integration program.
Written by Shirui shirui816@gmail.com
### metafile format:
/window/data window_center sprint_konst [Temperature]
### window data file format:
time_step coordinate (1-dimentional)
"""

arg_parser = argparse.ArgumentParser(
    description=description, formatter_class=RawTextHelpFormatter)
arg_parser.add_argument('-o', '--output',
                        metavar='Output free energy file',
                        default='free_py.txt', dest='out_put',
                        help="Optional, use 'free_py.txt' as default", )
arg_parser.add_argument('-T', '--temperature',
                        metavar='Temperature',
                        dest='temperature', default=-1, type=float,
                        help="Optional, set a default temperature globally.")
arg_parser.add_argument('-R', '--reduced',
                        default=0, type=int, metavar='0|1', dest='is_reduced',
                        choices=[0, 1],
                        help='Is reduced units being used?')
arg_parser.add_argument('-X', '--range',
                        nargs=2, default=None, metavar='Range of xi',
                        dest='range', type=float,
                        help="Range of reaction coordinate.")
arg_parser.add_argument('-Q', '--order',
                        default=2, metavar='order',
                        dest='order', type=int,
                        help="Order of probability function.")
arg_parser.add_argument('meta_file',
                        nargs=None,
                        help='Meta file name')
arg_parser.add_argument('max_bin',
                        type=int, nargs=None,
                        help='How many bins were used in integration.')

args = arg_parser.parse_args()
alvars = vars(args)

# Utils
p3 = lambda x, a1, a2, a3, c: c * np.exp(-(a1 * x + a2 * x ** 2 + a3 * x ** 3))
p4 = lambda x, a1, a2, a3, a4, c: c * np.exp(-(a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4))

# Variables
_temperature = alvars['temperature']
_is_reduced = alvars['is_reduced']
_max_bin = alvars['max_bin']  # how many bins were used in integration
_xi_range = alvars['range']
_out_put_file = open(alvars['out_put'], 'w')
_out_put_file.write('#r PMF MF\n')
_out_put_file.close()
_out_put_file = open(alvars['out_put'], 'a')
_order = alvars['order']

_kb = _is_reduced or KB * NA
# if _xi_range:
#    raise ValueError(r'Give the rigth range of \xi!')

_meta_file = open(alvars['meta_file'], 'r')
_window_info, _min = [], []


class NoTemperatureError(Exception):
    r"""No temperature error."""

    pass


def pbc(x, d):
    r"""Period boundary condition."""
    return x - d * np.round(x / d)


for _line in _meta_file:
    if not re.search('^#', _line) is None:
        continue
    _line = re.split('\s+', _line.strip())
    if _temperature == -1 and len(_line) != 4:
        raise NoTemperatureError("You have not set temperature for this "
                                 "window or a global temperature!")
    _window_data = pd.read_csv(_line[0], header=None, squeeze=1,
                               delim_whitespace=True, comment='#').values[:, 1]
    center_ = float(_line[1])
    spring_konst = float(_line[2])
    kbT = float(_line[3]) if len(_line) == 4 else _kb * _temperature
    wmean = _window_data.mean()
    _window_data = _window_data - wmean
    kde = gaussian_kde(_window_data)
    x = np.linspace(-20, 20, 1000)
    y = kde(x)
    popt3, pcov3 = curve_fit(p3, x, y, p0=[0]*4)
    popt4, pcov4 = curve_fit(p4, x, y, p0=[0]*5)
    a1, a2, a3, c1 = popt3
    b1, b2, b3, b4, c2 = popt4
    _window_info.append(
        [wmean, _window_data.var(), center_, spring_konst,
         kbT, a1, a2, a3, c1, b1, b2, b3, b4, c2]
    )
    _min.append(_window_data.min() + wmean)
    _min.append(_window_data.max() + wmean)

_window_info = np.array(_window_info)
_window_info = _window_info[np.argsort(_window_info.T[4])]  # sort by center_

if _xi_range:
    if min(_min) > _xi_range[0] or max(_min) < _xi_range[1]:
        warnings.warn("Warning, xi range exceeds the sample range!",
                      UserWarning)

_xi_range = _xi_range or [min(_min), max(_min)]
_xis = np.linspace(_xi_range[0], _xi_range[1], _max_bin)
_xi_mean_w = _window_info.T[0][:, np.newaxis]
_xi_var_w = _window_info.T[1][:, np.newaxis]
_xi_center_w = _window_info.T[2][:, np.newaxis]
_k_w = _window_info.T[3][:, np.newaxis]
_kbT_w = _window_info.T[4][:, np.newaxis]
_a1_w = _window_info.T[5][:, np.newaxis]
_a2_w = _window_info.T[6][:, np.newaxis]
_a3_w = _window_info.T[7][:, np.newaxis]
_c1_w = _window_info.T[8][:, np.newaxis]
_b1_w = _window_info.T[9][:, np.newaxis]
_b2_w = _window_info.T[10][:, np.newaxis]
_b3_w = _window_info.T[11][:, np.newaxis]
_b4_w = _window_info.T[12][:, np.newaxis]
_c2_w = _window_info.T[13][:, np.newaxis]
# \partial A/\partial \xi_{bin} =
# \sum_i^{window} P_i(\xi_{bin})/(\sum_i^{window} P_i(\xi_{bin})) \times
# \partial A_i^u/\partial \xi_{bin}
# \partial A_i^u / \xi_{bin}, with shape (n_window, n_xi)
_delta_xis = _xis - _xi_mean_w
if _order == 2:
    _dAu_dxis = _kbT_w * _delta_xis / _xi_var_w - \
                _k_w * (_xis - _xi_center_w)  # to 2th
    _pb_i = 1 / np.sqrt(2 * np.pi) * 1 / np.sqrt(_xi_var_w) * \
            np.exp(-0.5 * _delta_xis ** 2 / _xi_var_w)

if _order == 3:
    _dAu_dxis = _kbT_w * (_a1_w + 2 * _a2_w * _delta_xis + 3 * _a3_w * _delta_xis ** 2) - _k_w * (_xis - _xi_center_w)
    _pb_i = np.exp(-(_a1_w * _delta_xis + _a2_w * _delta_xis ** 2 + _a3_w * _delta_xis ** 3)) * _c1_w

if _order == 4:
    _dAu_dxis = _kbT_w * (_b1_w + 2 * _b2_w * _delta_xis + 3* _b3_w * _delta_xis ** 2 +4* _b4_w * _delta_xis ** 3) - _k_w * (
                _xis - _xi_center_w)
    _pb_i = np.exp(-(_b1_w * _delta_xis + _b2_w * _delta_xis ** 2 +_b3_w * _delta_xis ** 3 +
                     _b4_w * _delta_xis ** 4)) * _c2_w

# N_iP_i(\xi_{bin}), with shape (n_window, n_xi),
# all Nis are same in this case

#_norm = simps(_pb_i, _xis, axis=1)[:, None]
_norm = np.sum(_pb_i, axis=1)[:, None]
_pb_i = _pb_i / _norm
_dA_dxis = np.sum(_dAu_dxis * _pb_i, axis=0)
_pb_xi = np.sum(_pb_i, axis=0)
_dA_dxis /= _pb_xi

_pmf = np.array([simps(_dA_dxis[_xis <= r], _xis[_xis <= r]) for r in _xis])
np.savetxt(_out_put_file, np.vstack([_xis, _pmf, _dA_dxis]).T, fmt="%.6f")
_out_put_file.close()
