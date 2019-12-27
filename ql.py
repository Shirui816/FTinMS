import numpy as np
import numba as nb
from numba import cuda
from cmath import exp as cexp
from math import sqrt, floor, pi, atan2
from math import fmod, ceil, gamma

__doc__ = """This program is just an example. $w$ and multiple $l$s are not supported yet.
PERIODIC BOUNDARY CONDITION (pbc) is always ON!!! I am planning to add a switch about this.
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
    with cuda.gpus[gpu]:
        #device = cuda.get_current_device()
        #tpb = device.WARP_SIZE
        tpb = 32
        bpg = ceil(n / tpb)
        cu_cell_ind[bpg, tpb](pos, box, ibox, cell_id)
    cell_list = np.argsort(cell_id)  # pyculib radixsort for cuda acceleration.
    cell_id = cell_id[cell_list]
    cell_counts = np.r_[0, np.cumsum(np.bincount(cell_id, minlength=n_cell))]
    return cell_list.astype(np.int64), cell_counts.astype(np.int64)


def Ql(a, b, l, box, rc, gpu=0):
    ret = np.zeros((a.shape[0],), dtype=np.float64)
    dim = np.ones(a.shape[1], dtype=np.int64) * 3
    ndim = a.shape[1]
    ibox = np.asarray(np.round(box / rc), dtype=np.int64)
    cl, cc = cu_cell_list(b, box, ibox, gpu=gpu)
    _d = int(l * 2 + 1)

    @cuda.jit(
        "void(float64[:,:],float64[:,:], float64[:],int64[:],"
        "float64,int64[:],int64[:],float64[:],int64[:])"
    )
    def _Ql(_a, _b, _box, _ibox, _rc, _cl, _cc, _ret, _dim):
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
        Qveci = cuda.local.array(_d, nb.complex128)
        for _ in range(_d):
            Qveci[_] = 0 + 0j
        nn = 0
        for j in range(_a.shape[1] ** 3):
            unravel_index_f_cu(j, _dim, cell_vec_j)
            _add_local_arr_mois_1(cell_vec_j, cell_vec_i)
            # cell_vec_i + (-1, -1, -1) to (+1, +1, +1)
            # unraveled results would be (0,0,0) to (2,2,2) for dim=3
            cell_j = ravel_index_f_cu(cell_vec_j, _ibox)  # ravel cell id vector to cell id
            start = _cc[cell_j]  # start pid in the cell_j th cell
            end = _cc[cell_j + 1]  # end pid in the cell_j th cell
            for k in range(start, end):  # particle ids in cell_j
                pid_k = _cl[k]
                dx = -_a[i, 0] + _b[pid_k, 0]
                dy = -_a[i, 1] + _b[pid_k, 1]
                dz = -_a[i, 2] + _b[pid_k, 2]
                dx = dx - _box[0] * floor(dx / _box[0] + 0.5)
                dy = dy - _box[1] * floor(dy / _box[1] + 0.5)
                dz = dz - _box[2] * floor(dz / _box[2] + 0.5)
                dr = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                if 1e-5 < dr <= _rc:
                    phi = atan2(dy, dx) + pi
                    cosTheta = dz / dr
                    for m in range(-l, l + 1):
                        Qveci[m + l] += sphHar(l, m, cosTheta, phi)
                    nn += 1.
        if nn == 0: nn = 1.
        resi = 0
        for _ in range(_d):
            resi += abs(Qveci[_] / nn) ** 2
        _ret[i] = sqrt(4 * pi / (_d) * resi)

    with cuda.gpus[0]:
        #device = cuda.get_current_device()
        #tpb = device.WARP_SIZE
        tpb = 32
        bpg = ceil(a.shape[0] / tpb)
        _Ql[bpg, tpb](
            a, b, box, ibox, rc, cl, cc, ret, dim
        )
    np.savetxt('q%d.txt' % (l), ret, fmt='%.6f')
    print(ret.mean())
    return ret

a = np.loadtxt('2.txt')
box = np.array([100., 100, 100])
Ql(a, a, l=6, box=box, rc=1.02, gpu=0)
