import time
from cmath import exp as cexp
from math import fmod, ceil, gamma
from math import sqrt, floor, pi, atan2

import numba as nb
import numpy as np
from numba import cuda
from scipy.spatial import cKDTree

__doc__ = """Ql calculation, building NN with cKDTree, slower but simpler.
Ql for particle i is all "bond order" of pairs surrounding the particle.
the Q4, Q6 for center particle of FCC structure are 0.1909, 0.5745
0.0972, 0.4848 for hcp
0.0364, 0.5107 for bcc
W4, W6 for fcc are -0.1593, -0.0132
0.1341, -0.0124 for hcp
0.1593, 0.0132 for bcc
"""


@cuda.jit("float64(int64, int64, float64)", device=True)
def legendre(l, m, x):
    pmm = 1.0
    if m > 0:
        somx2 = sqrt((1. - x) * (1. + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    for ll in range(m + 2, l + 1):
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


@cuda.jit("complex128(int64, int64, float64, float64)", device=True)
def spherical_harmonics(l, m, cos_theta, phi):
    m1 = abs(m)
    c = sqrt((2 * l + 1) * gamma(l - m1 + 1.) / (4 * pi * gamma(l + m1 + 1.)))
    c *= legendre(l, m1, cos_theta)
    y = cexp(m * phi * 1j)
    if fmod(m, 2) == -1.:
        y *= -1
    return y * c + 0j


@cuda.jit('float64(float64[:], float64[:], float64[:])', device=True)
def pbc_dist_cu(a, b, box):
    tmp = 0
    for i in range(a.shape[0]):
        d = b[i] - a[i]
        d = d - floor(d / box[i] + 0.5) * box[i]
        tmp += d * d
    return sqrt(tmp)


class Ql(object):
    def __init__(self, ls=np.array([6], dtype=np.int64), gpu=0):
        self.gpu = gpu
        self.ls = ls
        self.dim_local_vec = (ls.shape[0], int(2 * ls.max() + 1))
        self.dim_local_ret_vec = self.dim_local_vec[0]
        print("Building calculator...")
        s = time.time()
        with cuda.gpus[gpu]:
            self.calculator = self.calculator()
        print("Done. Time costing: %.4fs." % (time.time() - s))

    def calculate(self, x, box, rc):
        print("Building neighbour list...")
        s = time.time()
        tree = cKDTree(np.mod(x, box), boxsize=box)
        _nn = tree.query_ball_point(np.mod(x, box), rc)
        nn = np.zeros([len(_nn), len(max(_nn, key=lambda _x: len(_x)))], dtype=np.int64) - 1
        for index, item in enumerate(_nn):
            nn[index][0:len(item)] = item
        print("Done. Time costing: %.4fs." % (time.time() - s))
        print("Start calculating...")
        s = time.time()
        ret = np.zeros((x.shape[0], self.ls.shape[0]), dtype=np.float64)
        with cuda.gpus[self.gpu]:
            d_nn = cuda.to_device(nn)
            device = cuda.get_current_device()
            tpb = device.WARP_SIZE
            # tpb = 32
            bpg = ceil(x.shape[0] / tpb)
            self.calculator[bpg, tpb](
                x, box, rc ** 2, d_nn, self.ls, ret
            )
        print("Done. Time costing: %.4fs." % (time.time() - s))
        return np.hstack([x, ret])

    def calculator(self):
        _dim_local_vec = self.dim_local_vec
        _dim_local_ret_vec = self.dim_local_ret_vec
        with cuda.gpus[self.gpu]:
            @cuda.jit("void(float64[:,:], float64[:], float64, int64[:,:], int64[:], float64[:,:])")
            def _ql(_x, _box, _rc2, _nl, _ls, _ret):
                i = cuda.grid(1)
                if i >= _x.shape[0]:
                    return
                q_vec_i = cuda.local.array(_dim_local_vec, nb.complex128)
                res_i = cuda.local.array(_dim_local_ret_vec, nb.float64)
                for _ in range(_dim_local_vec[0]):
                    res_i[_] = 0
                    for __ in range(_dim_local_vec[1]):
                        q_vec_i[_, __] = 0 + 0j
                nn = 0.0
                for pj in _nl[i]:
                    if pj == -1:  # -1 is guard element
                        continue
                    for pk in _nl[i]:
                        if pk == -1:
                            continue
                        if pk <= pj:
                            continue
                        dx = _x[pk, 0] - _x[pj, 0]
                        dy = _x[pk, 1] - _x[pj, 1]
                        dz = _x[pk, 2] - _x[pj, 2]
                        dx = dx - _box[0] * floor(dx / _box[0] + 0.5)
                        dy = dy - _box[1] * floor(dy / _box[1] + 0.5)
                        dz = dz - _box[2] * floor(dz / _box[2] + 0.5)
                        dr2 = dx * dx + dy * dy + dz * dz
                        if dr2 >= _rc2:
                            continue
                        dr = sqrt(dr2)
                        nn += 1.0
                        phi = atan2(dy, dx)
                        if phi < 0:
                            phi = phi + 2 * pi
                        cos_theta = dz / dr
                        for _l in range(_ls.shape[0]):
                            l = _ls[_l]
                            for m in range(-l, l + 1):
                                q_vec_i[_l, m + l] += spherical_harmonics(l, m, cos_theta, phi)
                if nn < 1.0:
                    nn = 1.0
                for _ in range(_dim_local_vec[0]):
                    for __ in range(_dim_local_vec[1]):
                        res_i[_] += abs(q_vec_i[_, __] / nn) ** 2
                for _ in range(_dim_local_vec[0]):
                    _ret[i, _] = sqrt(res_i[_] * 4 * pi / (2 * _ls[_] + 1))
        return _ql


X = np.loadtxt('xyz.txt')
box = np.array([38., 28, 10], dtype=np.float64)
# X = X - box / 2  # no need for moving box center to 0, using cKDTree building NN
Q46 = Ql(ls=np.array([6], dtype=np.int64), gpu=0)
ret = Q46.calculate(x=X, box=box, rc=0.38 * 2)
np.savetxt('out_ql.txt', ret, fmt='%.4f')
