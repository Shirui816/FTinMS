import numpy as np
import numba as nb
from numba import cuda
from cmath import exp as cexp
from math import sqrt, floor, pi, atan2
from math import fmod, ceil, gamma

__doc__ = """This program is just an example. $w$ and multiple $l$s are not supported yet.
PERIODIC BOUNDARY CONDITION (pbc) is always ON!!! I am planning to add a switch about this.
ALL BONDS AROUND ONE PARTICLE!!!
"""

@cuda.jit("void(int64[:], int64)")
def cu_set_to_int(arr, val):
    i = cuda.grid(1)
    if i >= arr.shape[0]:
        return
    arr[i] = val


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
def sphHar(l, m, cosTheta, phi):
    m1 = abs(m)
    c = sqrt((2 * l + 1) * gamma(l - m1 + 1.) / (4 * pi * gamma(l + m1 + 1.)))
    c *= legendre(l, m1, cosTheta)
    y = cexp(m * phi * 1j)
    if fmod(m, 2) == -1.:
        y *= -1
    return y * c + 0j


@cuda.jit("int64(float64[:], float64[:], int64[:])", device=True)
def cu_cell_id(p, box, ibox):  # In the Fortran way
    ret = floor((p[0] / box[0] + 0.5) * ibox[0])
    tmp = ibox[0]
    for i in range(1, p.shape[0]):
        ret += floor((p[i] / box[i] + 0.5) * ibox[i]) * tmp
        tmp *= ibox[i]
    return ret
    # return floor((p[0] / box[0] + 0.5) * ibox[0]) + \
    # floor((p[1] / box[1] + 0.5) * ibox[1]) * ibox[0] + \
    # floor((p[2] / box[2] + 0.5) * ibox[2]) * ibox[1] * ibox[0]
    # +0.5 for 0 is at center of box.
    # unravel in Fortran way.


@cuda.jit("void(float64[:, :], float64[:], int64[:], int64[:])")
def cu_cell_ind(pos, box, ibox, ret):
    i = cuda.grid(1)
    if i < pos.shape[0]:
        pi = pos[i]
        ic = cu_cell_id(pi, box, ibox)
        ret[i] = ic


@cuda.jit('float64(float64[:], float64[:], float64[:])', device=True)
def pbc_dist_cu(a, b, box):
    tmp = 0
    for i in range(a.shape[0]):
        d = b[i] - a[i]
        d = d - floor(d / box[i] + 0.5) * box[i]
        tmp += d * d
    return sqrt(tmp)


@cuda.jit("void(int64, int64[:], int64[:])", device=True)
def unravel_index_f_cu(i, dim, ret):  # unravel index in Fortran way.
    for k in range(dim.shape[0]):
        ret[k] = int(i % dim[k])
        i = (i - ret[k]) / dim[k]


@cuda.jit("int64(int64[:], int64[:])", device=True)
def ravel_index_f_cu(i, dim):  # ravel index in Fortran way.
    ret = (i[0] + dim[0]) % dim[0]
    tmp = dim[0]
    for k in range(1, dim.shape[0]):
        ret += ((i[k] + dim[k]) % dim[k]) * tmp
        tmp *= dim[k]
    return ret


@cuda.jit("void(int64[:], int64[:])", device=True)
def _add_local_arr_mois_1(a, b):
    for i in range(a.shape[0]):
        a[i] = a[i] + b[i] - 1


@cuda.jit("void(int64[:], int64[:])")
def cu_cell_count(cell_id, ret):
    i = cuda.grid(1)
    if i >= cell_id.shape[0]:
        return
    cuda.atomic.add(ret, cell_id[i] + 1, 1)


def cu_cell_list(pos, box, ibox, gpu=0):
    n = pos.shape[0]
    n_cell = np.multiply.reduce(ibox)
    cell_id = np.zeros(n).astype(np.int64)
    with cuda.gpus[0]:
    #for i in range(1):
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        #tpb = 32
        bpg = ceil(n / tpb)
        cu_cell_ind[bpg, tpb](pos, box, ibox, cell_id)
    cell_list = np.argsort(cell_id)  # pyculib radixsort for cuda acceleration.
    cell_id = cell_id[cell_list]
    cell_counts = np.r_[0, np.cumsum(np.bincount(cell_id, minlength=n_cell))]
    return cell_list.astype(np.int64), cell_counts.astype(np.int64)


def cu_nl(a, box, rc, n_guess=100, gpu=0):
    dim = np.ones(a.shape[1], dtype=np.int64) * 3
    ndim = a.shape[1]
    ibox = np.asarray(np.round(box / rc), dtype=np.int64)
    cl, cc = cu_cell_list(a, box, ibox, gpu=gpu)
    d_nl = cuda.device_array((a.shape[0], n_guess), dtype=np.int64)
    d_nc = cuda.device_array((a.shape[0],), dtype=np.int64)
    d_n_max = cuda.device_array((1,), dtype=np.int64)

    @cuda.jit(
        "void(float64[:,:],float64[:],int64[:],"
        "float64,int64[:],int64[:],int64[:,:],int64[:], int64[:], int64[:])"
    )
    def _nl(_a, _box, _ibox, _rc, _cl, _cc, _ret, _nc, _dim, _n_max):
        r"""
        :param _a: positions of a, (n_pa, n_d)
        :param _b: positions of b, (n_pb, n_d)
        :param _box: box, (n_d,)
        :param _ibox: bins, (n_d,)
        :param _rc: r_cut of rdf, double
        :param _cl: cell-list of b, (n_pb,)
        :param _cc: cell-count-cum, (n_cell + 1,)
        :param _ret: Ql
        :return: None
        """
        i = cuda.grid(1)
        if i >= _a.shape[0]:
            return
        cell_i = cu_cell_id(_a[i], _box, _ibox)  # a[i] in which cell
        cell_vec_i = cuda.local.array(ndim, nb.int64)  # unravel the cell id
        unravel_index_f_cu(cell_i, _ibox, cell_vec_i)  # unravel the cell id
        cell_vec_j = cuda.local.array(ndim, nb.int64)
        nn = 0
        n_needed = 1
        for j in range(3 ** ndim):
            unravel_index_f_cu(j, _dim, cell_vec_j)
            _add_local_arr_mois_1(cell_vec_j, cell_vec_i)
            # cell_vec_i + (-1, -1, -1) to (+1, +1, +1)
            # unraveled results would be (0,0,0) to (2,2,2) for dim=3
            cell_j = ravel_index_f_cu(cell_vec_j, _ibox)  # ravel cell id vector to cell id
            start = _cc[cell_j]  # start pid in the cell_j th cell
            end = _cc[cell_j + 1]  # end pid in the cell_j th cell
            for k in range(start, end):  # particle ids in cell_j
                pid_k = _cl[k]
                dr = pbc_dist_cu(_a[pid_k], _a[i], _box)
                if dr < _rc:
                    if nn < _ret.shape[1]:
                        _ret[i, nn] = pid_k
                    else:
                        n_needed = nn + 1
                    nn += 1
        _nc[i] = nn
        if nn > 0:
            cuda.atomic.max(_n_max, 0, n_needed)

    with cuda.gpus[0]:
    #for i in range(1):
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        #tpb = 32
        bpg = ceil(a.shape[0] / tpb)
        p_n_max = cuda.pinned_array(1, dtype=np.int64)
        while True:
            cu_set_to_int[bpg, tpb](d_nc, 0)
            _nl[bpg, tpb](
                a, box, ibox, rc, cl, cc, d_nl, d_nc, dim, d_n_max
            )
            d_n_max.copy_to_host(p_n_max)
            cuda.synchronize()
            if p_n_max[0] > n_guess:
                n_guess = p_n_max[0]
                n_guess = n_guess + 8 - (n_guess & 7)
                d_nl = cuda.device_array((a.shape[0], n_guess), dtype=np.int64)
            else:
                break
    return d_nl, d_nc, n_guess


def ql(x, box, rc, ls=np.array([4, 6]), n_guess=100):
    d_nl, d_nc, n_guess = cu_nl(x, box, rc, n_guess)
    #print(d_nl.copy_to_host(), d_nc.copy_to_host())
    _d = (ls.shape[0], int(2 * ls.max() + 1))
    _dd = _d[0]
    ret = np.zeros((x.shape[0], ls.shape[0]), dtype=np.float64)

    @cuda.jit("void(float64[:,:], float64[:], float64, int64[:,:], int64[:], int64[:], float64[:,:])")
    def _ql(_x, _box, _rc2, _nl, _nc, _ls, _ret):
        i = cuda.grid(1)
        if i >= _x.shape[0]:
            return
        Qveci = cuda.local.array(_d, nb.complex128)
        resi = cuda.local.array(_dd, nb.float64)
        for _ in range(_d[0]):
            resi[_] = 0
            for __ in range(_d[1]):
                Qveci[_, __] = 0 + 0j
        nn = 0.0
        for j in range(_nc[i] - 1):
            pj = _nl[i, j]
            for k in range(j + 1, _nc[i]):
                pk = _nl[i, k]
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
                cosTheta = dz / dr
                for _l in range(_ls.shape[0]):
                    l = _ls[_l]
                    for m in range(-l, l + 1):
                        Qveci[_l, m + l] += sphHar(l, m, cosTheta, phi)
        #print(i, nn)
        if nn < 1.0:
            nn = 1.0
        for _ in range(_d[0]):
            for __ in range(_d[1]):
                resi[_] += abs(Qveci[_, __] / nn) ** 2
        for _ in range(_d[0]):
            _ret[i, _] = sqrt(resi[_] * 4 * pi / (2 * _ls[_] + 1))

    with cuda.gpus[0]:
    #for i in range(1):
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        #tpb = 32
        bpg = ceil(a.shape[0] / tpb)
        _ql[bpg, tpb](
            a, box, rc**2, d_nl, d_nc, ls, ret
        )
    return ret


a = np.loadtxt('fcc_13.txt')
box = np.array([100., 100, 100])
ret = ql(a, box=box, rc=1.02)
print(ret)
