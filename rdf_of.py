import numba as nb
import numpy as np
from math import ceil
from math import floor
from math import sqrt
from numba import cuda


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


def rdf_of_ab(a, b, box, da, db, bs, rc, gpu=0):
    ret = np.zeros((a.shape[0], int(rc / bs)), dtype=np.int64)
    dim = np.ones(a.shape[1], dtype=np.int64) * 3
    ndim = a.shape[1]
    ibox = np.asarray(np.round(_box / _rc), dtype=np.int64)
    cl, cc = cu_cell_list(b, box, ibox, gpu=gpu)

    # ndim: dimentsion, a.shape[1], jit does not support
    # using a.shape[1] directly in cuda.local.array creation,
    # numba==0.44.1
    # dim = np.ones(ndim, dtype=np.int64) * ndim
    # dimension of (-1, 0, 1) vector in calculating adjacent cells
    # a 3-D case would be (3, 3, 3), simply (3,) * n_d
    # _dim is required for in nonpython mode, sig=array(int64, 1d, A)
    # and using dim directly would by `readonly array(int64, 1d, C)'

    @cuda.jit(
        "void(float64[:,:],float64[:,:],float64[:],int64[:],float64[:],"
        "float64[:],float64,float64,int64[:],int64[:],int64[:,:], int64[:])"
    )
    def _rdf(_a, _b, _box, _ibox, _da, _db, _bs, _rc, _cl, _cc, _ret, _dim):
        r"""
        :param _a: positions of a, (n_pa, n_d)
        :param _b: positions of b, (n_pb, n_d)
        :param _box: box, (n_d,)
        :param _ibox: bins, (n_d,)
        :param _da: diameters of a, (n_pa,)
        :param _db: diameters of b, (n_pb,)
        :param _bs: binsize of rdf, double
        :param _rc: r_cut of rdf, double
        :param _cl: cell-list of b, (n_pb,)
        :param _cc: cell-count-cum, (n_cell + 1,)
        :param _ret: rdfs of (n_pa, n_bin)
        :return: None
        """
        i = cuda.grid(1)
        if i >= _a.shape[0]:
            return
        cell_i = cu_cell_id(_a[i], _box, _ibox)  # a[i] in which cell
        cell_vec_i = cuda.local.array(ndim, nb.int64)  # unravel the cell id
        unravel_index_f_cu(cell_i, _ibox, cell_vec_i)  # unravel the cell id
        cell_vec_j = cuda.local.array(ndim, nb.int64)
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
                dij = pbc_dist_cu(_a[i], _b[pid_k], _box) - (_da[i] + _db[pid_k]) / 2 + 1
                if dij < rc:
                    cuda.atomic.add(_ret[i], int(dij / _bs), 1)

    with cuda.gpus[0]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = ceil(a.shape[0] / tpb)
        _rdf[bpg, tpb](
            a, b, box, ibox, da, db, bs, rc, cl, cc, ret, dim
        )
    rho_b = b.shape[0] / np.multiply.reduce(box)
    r = (np.arange(ret.shape[1] + 1) + 0.5) * bs
    dV = 4 / 3 * np.pi * np.diff(r ** 3)
    ret = ret / np.expand_dims(dV, 0) / rho_b
    np.savetxt('rdf.txt', np.vstack([r[:-1], ret.mean(axis=0)]).T, fmt='%.6f')
    return r, ret


if __name__ == "__main__":
    from sys import argv

    _a, _b = np.loadtxt(argv[1], dtype=np.float), np.loadtxt(argv[2], dtype=np.float)
    _box = np.array([50, 50, 50], dtype=np.float)
    _rc = 3.0
    _ibox = np.asarray(np.round(_box / _rc), dtype=np.int64)
    _da = np.ones(_a.shape[0], dtype=np.float)
    _db = np.ones(_b.shape[0], dtype=np.float)
    _bs = 0.1
    _r, _ret = rdf_of_ab(_a, _b, _box, _da, _db, _bs, _rc, gpu=0)
