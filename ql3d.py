import time
from cmath import exp as cexp
from math import fmod, ceil, gamma
from math import sqrt, floor, pi, atan2

import numba as nb
import numpy as np
from numba import cuda, complex128, complex64, float32, float64, int32, void
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


@cuda.jit(device=True)
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


@cuda.jit(device=True)
def spherical_harmonics(l, m, cos_theta, phi):
    m1 = abs(m)
    c = sqrt((2 * l + 1) * gamma(l - m1 + 1.) / (4 * pi * gamma(l + m1 + 1.)))
    c *= legendre(l, m1, cos_theta)
    y = cexp(m * phi * 1j)
    if fmod(m, 2) == -1.:
        y *= -1
    return y * c + 0j



class Ql(object):
    def __init__(self, ls=np.array([6], dtype=np.int32), dtype=np.float32, gpu=0):
        self.gpu = gpu
        self.ls = np.asarray(ls, dtype=np.int32)
        self.dim_local_vec = (ls.shape[0], int(2 * ls.max() + 1))
        self.dim_local_ret_vec = self.dim_local_vec[0]
        self.dtype = dtype
        print("Building calculator...")
        s = time.time()
        with cuda.gpus[gpu]:
            self.calculator = self.build_calculator()
        print("Done. Time costing: %.4fs." % (time.time() - s))

    def calculate(self, x, box, rc):
        x = np.asarray(x, dtype=self.dtype)
        box = np.asarray(box, dtype=self.dtype)
        print("Building neighbour list...")
        s = time.time()
        tree = cKDTree(np.mod(x, box), boxsize=box)
        _nn = tree.query_ball_point(np.mod(x, box), rc)
        nn = np.zeros([len(_nn), len(max(_nn, key=lambda _x: len(_x)))], dtype=np.int32) - 1
        for index, item in enumerate(_nn):
            nn[index][0:len(item)] = item
        print("Done. Time costing: %.4fs." % (time.time() - s))
        print("Start calculating...")
        s = time.time()
        ret = np.zeros((x.shape[0], self.ls.shape[0]), dtype=np.float64)
        with cuda.gpus[self.gpu]:
            d_nn = cuda.to_device(nn)
            d_x = cuda.to_device(x)
            d_box = cuda.to_device(box)
            d_ret = cuda.to_device(ret)
            d_ls = cuda.to_device(self.ls)
            #device = cuda.get_current_device()
            #tpb = device.WARP_SIZE
            tpb = 64
            bpg = int(ceil(x.shape[0] / tpb))
            self.calculator[bpg, tpb](
                d_x, d_box, rc ** 2, d_nn, d_ls, d_ret
            )
            d_ret.copy_to_host(ret)
        print("Done. Time costing: %.4fs." % (time.time() - s))
        return np.hstack([x, ret])

    def build_calculator(self):
        _dim_local_vec = self.dim_local_vec
        _dim_local_ret_vec = self.dim_local_ret_vec
        if self.dtype == np.float32:
            _float = float32
            _complex = complex64
        else:
            _float = float64
            _complex = complex128
        with cuda.gpus[self.gpu]:
            @cuda.jit(void(_float[:, :], _float[:], _float, int32[:, :], int32[:], _float[:, :]))
            def _ql(_x, _box, _rc2, _nl, _ls, _ret):
                i = cuda.grid(1)
                if i >= _x.shape[0]:
                    return
                q_vec_i = cuda.local.array(_dim_local_vec, _complex)
                res_i = cuda.local.array(_dim_local_ret_vec, _float)
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
                        res_i[_] += _float(abs(q_vec_i[_, __] / nn) ** 2)
                for _ in range(_dim_local_vec[0]):
                    _ret[i, _] = _float(sqrt(res_i[_] * 4 * pi / (2 * _ls[_] + 1)))
        return _ql

if __name__ == "__main__":
    X = np.loadtxt('xyz.txt')
    box = np.array([38., 28, 10], dtype=np.float64)
    # X = X - box / 2  # no need for moving box center to 0, using cKDTree building NN
    Q46 = Ql(ls=np.array([4, 6], dtype=np.int32), gpu=0)
    ret = Q46.calculate(x=X, box=box, rc=0.38 * 2)
    np.savetxt('out_ql.txt', ret, fmt='%.4f')
