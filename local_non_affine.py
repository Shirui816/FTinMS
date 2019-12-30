import numba as nb
import numpy as np
from math import ceil
from math import floor
from math import sqrt
from numba import cuda


@cuda.jit("void(float64[:,:], float64[:], float64[:], float64[:], float64[:])", device=True)
def cu_mat_dot_v_pbc(a, b, c, box, ret):
    for i in range(a.shape[0]):
        tmp = 0
        for j in range(a.shape[1]):
            dc = b[j] - c[j]
            dc = dc - box[j] * floor(dc / box[j] + 0.5)
            tmp += a[i, j] * dc
        ret[i] = tmp


@cuda.jit("void(float64[:,:], float64[:], float64[:])", device=True)
def cu_mat_dot_v(a, b, ret):
    for i in range(a.shape[0]):
        tmp = 0
        for j in range(a.shape[1]):
            tmp += a[i, j] * b[j]
        ret[i] = tmp


@cuda.jit("float64(float64[:])", device=True)
def cu_v_mod(r):
    tmp = 0
    for i in range(r.shape[0]):
        tmp += r[i] ** 2
    return sqrt(tmp)


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
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = ceil(n / tpb)
        cu_cell_ind[bpg, tpb](pos, box, ibox, cell_id)
    cell_list = np.argsort(cell_id)  # pyculib radixsort for cuda acceleration.
    cell_id = cell_id[cell_list]
    cell_counts = np.r_[0, np.cumsum(np.bincount(cell_id, minlength=n_cell))]
    return cell_list.astype(np.int64), cell_counts.astype(np.int64)


def cu_nl(a, box, rc, da, l0, gpu=0):
    # l0 is the strain tensor of the reference frame, box is always cubic of 0 strain.
    # l0 = np.array([[1., xy0, 0], [0, 1., 0], [0, 0, 1]])
    dim = np.ones(a.shape[1], dtype=np.int64) * 3
    ndim = a.shape[1]
    ibox = np.asarray(np.round(box / (rc + da.max())), dtype=np.int64)
    cl, cc = cu_cell_list(a, box, ibox, gpu=gpu)
    nc = np.zeros((a.shape[0],), dtype=np.int64)

    @cuda.jit(
        "void(float64[:,:],float64[:],int64[:],float64[:],"
        "float64[:,:],float64,int64[:],int64[:],int64[:], int64[:])"
    )
    def cu_nc(_a, _box, _ibox, _da, _l0, _rc, _cl, _cc, _nc, _dim):
        i = cuda.grid(1)
        if i >= _a.shape[0]:
            return
        cell_i = cu_cell_id(_a[i], _box, _ibox)  # a[i] in which cell
        cell_vec_i = cuda.local.array(ndim, nb.int64)  # unravel the cell id
        unravel_index_f_cu(cell_i, _ibox, cell_vec_i)  # unravel the cell id
        cell_vec_j = cuda.local.array(ndim, nb.int64)
        _dv = cuda.local.array(ndim, nb.float64)
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
                if pid_k == i:
                    continue
                cu_mat_dot_v_pbc(_l0, _a[pid_k], _a[i], _box, _dv)
                dr = cu_v_mod(_dv)
                delta = (_da[i] + _da[pid_k]) / 2 - 1
                if dr - delta < _rc:
                    _nc[i] += 1

    @cuda.jit(
        "void(float64[:,:],float64[:],int64[:],float64[:], float64[:,:],"
        "float64,int64[:],int64[:],int64[:,:],int64[:], int64[:])"
    )
    def _nl(_a, _box, _ibox, _da, _l0, _rc, _cl, _cc, _ret, _nc, _dim):
        i = cuda.grid(1)
        if i >= _a.shape[0]:
            return
        cell_i = cu_cell_id(_a[i], _box, _ibox)  # a[i] in which cell
        cell_vec_i = cuda.local.array(ndim, nb.int64)  # unravel the cell id
        unravel_index_f_cu(cell_i, _ibox, cell_vec_i)  # unravel the cell id
        cell_vec_j = cuda.local.array(ndim, nb.int64)
        _dv = cuda.local.array(ndim, nb.float64)
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
                if pid_k == i:
                    continue
                cu_mat_dot_v_pbc(_l0, _a[pid_k], _a[i], _box, _dv)
                dr = cu_v_mod(_dv)
                delta = (_da[i] + _da[pid_k]) / 2 - 1
                if dr - delta < _rc:
                    _ret[i, _nc[i]] = pid_k
                    _nc[i] += 1

    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = ceil(a.shape[0] / tpb)
        cu_nc[bpg, tpb](
            a, box, ibox, da, l0, rc, cl, cc, nc, dim
        )
        ret = np.zeros((a.shape[0], nc.max()), dtype=np.int64)
        nc = np.zeros((a.shape[0],), dtype=np.int64)
        _nl[bpg, tpb](
            a, box, ibox, da, l0, rc, cl, cc, ret, nc, dim
        )
    return ret, nc


def local_non_affine_of_ab(a, b, box, l0, lt, da, rc, gpu=0):
    ndim = a.shape[1]
    # l0 = np.array([[1, xy0, 0], [0, 1., 0], [0, 0, 1]])
    # lt = np.array([[1, xyt, 0], [0, 1., 0], [0, 0, 1]])
    invl0 = np.linalg.inv(l0)
    invlt = np.linalg.inv(lt)
    a0 = invl0.dot(a.T).T
    b0 = invlt.dot(b.T).T
    nl, nc = cu_nl(a0, box, rc, da, l0)
    Xij = np.zeros((a.shape[0], ndim, ndim), dtype=np.float64)
    Yij = np.zeros((a.shape[0], ndim, ndim), dtype=np.float64)
    DIV = np.zeros((a.shape[0],), dtype=np.float64)

    # ndim: dimentsion, a.shape[1], jit does not support
    # using a.shape[1] directly in cuda.local.array creation,
    # numba==0.44.1
    # dim = np.ones(ndim, dtype=np.int64) * ndim
    # dimension of (-1, 0, 1) vector in calculating adjacent cells
    # a 3-D case would be (3, 3, 3), simply (3,) * n_d
    # _dim is required for in nonpython mode, sig=array(int64, 1d, A)
    # and using dim directly would by `readonly array(int64, 1d, C)'

    @cuda.jit(
        "void(float64[:,:],float64[:,:],float64[:],int64[:,:],int64[:],"
        "float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:])"
    )
    def _cu_XY(_a, _b, _box, _nl, _nc, _Xij, _Yij, _l0, _lt):
        i = cuda.grid(1)
        if i >= _a.shape[0]:
            return
        _dr0 = cuda.local.array(ndim, nb.float64)
        _drt = cuda.local.array(ndim, nb.float64)
        for j in range(_nc[i]):
            pj = _nl[i, j]
            cu_mat_dot_v_pbc(_l0, _a[pj], _a[i], _box, _dr0)
            cu_mat_dot_v_pbc(_lt, _b[pj], _b[i], _box, _drt)
            for k in range(ndim):
                for l in range(ndim):
                    _Xij[i, k, l] += _drt[k] * _dr0[l]
                    _Yij[i, k, l] += _dr0[k] * _dr0[l]

    @cuda.jit(
        "void(float64[:,:],float64[:,:],float64[:], int64[:,:],"
        "int64[:],float64[:,:,:], float64[:,:], float64[:,:], float64[:])"
    )
    def _cu_DIV(_a, _b, _box, _nl, _nc, _XIY, _l0, _lt, _ret):
        i = cuda.grid(1)
        if i >= _a.shape[0]:
            return
        _dr0 = cuda.local.array(ndim, nb.float64)
        _drt = cuda.local.array(ndim, nb.float64)
        _dr = cuda.local.array(ndim, nb.float64)
        for j in range(_nc[i]):
            pj = _nl[i, j]
            cu_mat_dot_v_pbc(_l0, _a[pj], _a[i], _box, _dr0)
            cu_mat_dot_v_pbc(_lt, _b[pj], _b[i], _box, _drt)
            cu_mat_dot_v(_XIY[i], _dr0, _dr)
            for k in range(ndim):
                _ret[i] += (_drt[k] - _dr[k]) ** 2
        if _nc[i] != 0:
            _ret[i] = _ret[i] / _nc[i]

    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = ceil(a.shape[0] / tpb)
        _cu_XY[bpg, tpb](
            a0, b0, box, nl, nc, Xij, Yij, l0, lt
        )
        # XIY = np.matmul(Xij, np.linalg.inv(Yij))
        XIY = np.matmul(Xij, np.linalg.pinv(Yij, hermitian=True))
        # Moore-Penrose inverse, for nc[i] < ndim
        _cu_DIV[bpg, tpb](
            a0, b0, box, nl, nc, XIY, l0, lt, DIV
        )
    return XIY, DIV


if __name__ == "__main__":
    from sys import argv

    a, b = np.loadtxt(argv[1], dtype=np.float), np.loadtxt(argv[2], dtype=np.float64)
    box = np.array([47.56874465942, 47.56874465942, 47.56874465942], dtype=np.float64)
    rc = 3.0
    da = np.ones(a.shape[0], dtype=np.float)
    xy0 = 0
    xyt = 0.09999900311232
    l0 = np.array([[1, xy0, 0], [0, 1., 0], [0, 0, 1]])
    lt = np.array([[1, xyt, 0], [0, 1., 0], [0, 0, 1]])
    XIY, DIV = local_non_affine_of_ab(a, b, box, l0, lt, da, rc, gpu=0)
    XIY = XIY - np.eye(a.shape[1])
    np.savetxt('local_div.txt', np.hstack([a, DIV.reshape(-1, 1)]), fmt='%.6f')
    np.savetxt('local_tensor.txt', np.hstack([a, XIY.reshape(a.shape[0], -1)]), fmt='%.6f')
