import numpy as np
import numba as nb
from numba import cuda
from math import sqrt, floor, ceil


@cuda.jit("void(int64, int64[:], int64[:])", device=True)
def _unravel_index_f_cu(i, dim, ret):  # unravel index in Fortran way.
    for k in range(dim.shape[0]):
        ret[k] = int(i % dim[k])
        i = (i - ret[k]) / dim[k]


@cuda.jit("int64(int64[:], int64[:])", device=True)
def _ravel_index_f_cu(i, dim):  # ravel index in Fortran way.
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


@cuda.jit("void(float64[:,:], float64[:], float64[:], float64[:], float64[:])", device=True)
def _mat_dot_dv_pbc(a, b, c, box, ret):
    for i in range(a.shape[0]):
        tmp = 0
        for j in range(a.shape[1]):
            d = b[j] - c[j]
            d = d - box[j] * floor(d / box[j] + 0.5)
            tmp += a[i, j] * d
        ret[i] = tmp


@cuda.jit("float64(float64[:])", device=True)
def _v_mod(r):
    tmp = 0
    for i in range(r.shape[0]):
        tmp += r[i] ** 2
    return sqrt(tmp)


@cuda.jit("int64(float64[:], float64[:], int64[:])", device=True)
def _cell_id(_a, _box, _ibox):  # In the Fortran way
    ret = floor((_a[0] / _box[0] + 0.5) * _ibox[0])
    tmp = _ibox[0]
    for i in range(1, _a.shape[0]):
        ret += floor((_a[i] / _box[i] + 0.5) * _ibox[i]) * tmp
        tmp *= _ibox[i]
    return ret
    # return floor((p[0] / box[0] + 0.5) * ibox[0]) + \
    # floor((p[1] / box[1] + 0.5) * ibox[1]) * ibox[0] + \
    # floor((p[2] / box[2] + 0.5) * ibox[2]) * ibox[1] * ibox[0]
    # +0.5 for 0 is at center of box.
    # unravel in Fortran way.


@cuda.jit("void(float64[:, :], float64[:], int64[:], int64[:])")
def _cell_ind(_a, _box, _ibox, _ret):
    i = cuda.grid(1)
    if i < _a.shape[0]:
        pi = _a[i]
        ic = _cell_id(pi, _box, _ibox)
        _ret[i] = ic


@cuda.jit("void(int64[:], int64)")
def _init_array(_ary, _value):
    i = cuda.grid(1)
    if i >= _ary.shape[0]:
        return
    _ary[i] = _value


def neighbour_list(a, box, rc, da=None, strain=None, gpu=0):
    ndim = a.shape[1]
    dim = np.ones(ndim, dtype=np.int64) * 3
    n = a.shape[0]
    if da is None:
        da = np.ones(n, dtype=np.float64)
    if strain is None:
        strain = np.eye(ndim)
    ibox = np.asarray(np.round(box / (rc + da.max())), dtype=np.int64)
    n_cell = int(np.multiply.reduce(ibox))

    @cuda.jit(
        "void(float64[:,:],float64[:],int64[:],float64[:],"
        "float64[:,:],float64,int64[:],int64[:],int64[:],int64[:])"
    )
    def _nc(_a, _box, _ibox, _da, _str, _rc, _cl, _cc, _ret, _dim):
        i = cuda.grid(1)
        if i >= _a.shape[0]:
            return
        cell_i = _cell_id(_a[i], _box, _ibox)  # a[i] in which cell
        cell_vec_i = cuda.local.array(ndim, nb.int64)  # unravel the cell id
        _unravel_index_f_cu(cell_i, _ibox, cell_vec_i)  # unravel the cell id
        cell_vec_j = cuda.local.array(ndim, nb.int64)
        dv = cuda.local.array(ndim, nb.float64)
        for j in range(ndim ** 3):
            _unravel_index_f_cu(j, _dim, cell_vec_j)
            _add_local_arr_mois_1(cell_vec_j, cell_vec_i)
            # cell_vec_i + (-1, -1, -1) to (+1, +1, +1)
            # unraveled results would be (0,0,0) to (2,2,2) for dim=3
            cell_j = _ravel_index_f_cu(cell_vec_j, _ibox)  # ravel cell id vector to cell id
            start = _cc[cell_j]  # start pid in the cell_j th cell
            end = _cc[cell_j + 1]  # end pid in the cell_j th cell
            for k in range(start, end):  # particle ids in cell_j
                pid_k = _cl[k]
                if pid_k == i:
                    continue
                _mat_dot_dv_pbc(_str, _a[pid_k], _a[i], _box, dv)
                dr = _v_mod(dv)
                delta = (_da[i] + _da[pid_k]) / 2.0 - 1.0
                if dr - delta < _rc:
                    _ret[i] += 1

    @cuda.jit(
        "void(float64[:,:],float64[:],int64[:],float64[:],float64[:,:],"
        "float64,int64[:],int64[:],int64[:,:],int64[:],int64[:])"
    )
    def _nl(_a, _box, _ibox, _da, _str, _rc, _cl, _cc, _ret, _nc, _dim):
        i = cuda.grid(1)
        if i >= _a.shape[0]:
            return
        cell_i = _cell_id(_a[i], _box, _ibox)  # a[i] in which cell
        cell_vec_i = cuda.local.array(ndim, nb.int64)  # unravel the cell id
        _unravel_index_f_cu(cell_i, _ibox, cell_vec_i)  # unravel the cell id
        cell_vec_j = cuda.local.array(ndim, nb.int64)
        dv = cuda.local.array(ndim, nb.float64)
        for j in range(ndim ** 3):
            _unravel_index_f_cu(j, _dim, cell_vec_j)
            _add_local_arr_mois_1(cell_vec_j, cell_vec_i)
            # cell_vec_i + (-1, -1, -1) to (+1, +1, +1)
            # unraveled results would be (0,0,0) to (2,2,2) for dim=3
            cell_j = _ravel_index_f_cu(cell_vec_j, _ibox)  # ravel cell id vector to cell id
            start = _cc[cell_j]  # start pid in the cell_j th cell
            end = _cc[cell_j + 1]  # end pid in the cell_j th cell
            for k in range(start, end):  # particle ids in cell_j
                pid_k = _cl[k]
                if pid_k == i:
                    continue
                _mat_dot_dv_pbc(_str, _a[pid_k], _a[i], _box, dv)
                dr = _v_mod(dv)
                delta = (_da[i] + _da[pid_k]) / 2.0 - 1.0
                if dr - delta < _rc:
                    _ret[i, _nc[i]] = pid_k
                    _nc[i] += 1

    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = ceil(a.shape[0] / tpb)
        d_a = cuda.to_device(a)
        d_box = cuda.to_device(box)
        d_ibox = cuda.to_device(ibox)
        d_da = cuda.to_device(da)
        d_dim = cuda.to_device(dim)
        d_str = cuda.to_device(strain)
        d_cell_id = cuda.device_array((n,), dtype=np.int64)
        _cell_ind[bpg, tpb](d_a, d_box, d_ibox, d_cell_id)
        cell_id = d_cell_id.copy_to_host()
        cl = np.argsort(cell_id)
        cell_id = cell_id[cl]
        cc = np.r_[0, np.cumsum(np.bincount(cell_id, minlength=n_cell))]
        cc = cc.astype(np.int64)
        d_cl = cuda.to_device(cl)
        d_cc = cuda.to_device(cc)
        d_nc = cuda.device_array((n,), dtype=np.int64)
        _init_array[bpg, tpb](d_nc, 0)
        _nc[bpg, tpb](
            d_a, d_box, d_ibox, d_da, d_str, rc, d_cl, d_cc, d_nc, d_dim
        )
        nc = d_nc.copy_to_host()
        d_nl = cuda.device_array((n, nc.max()), dtype=np.int64)
        _init_array[bpg, tpb](d_nc, 0)
        _nl[bpg, tpb](
            d_a, d_box, d_ibox, d_da, d_str, rc, d_cl, d_cc, d_nl, d_nc, d_dim
        )
    return d_nl.copy_to_host(), nc
