import time
from math import pow, sqrt, floor, ceil
from sys import argv

import numpy as np
from Xml import Xml
from molecule import molecules
from numba import cuda, jit, float64


@jit(nopython=True, nogil=True)
def pbc(r, d):
    return r - d * np.rint(r / d)


@jit(nopython=True, nogil=True)
def build_mol_table(mol_info, n_atoms):
    ret = np.zeros(n_atoms, dtype=np.int64) - 1
    for i, mol in enumerate(mol_info):
        for j in mol:
            if j != -1:
                ret[j] = i
    return ret


@jit(nopython=True, nogil=True)
def unravel_index_f(i, dim):  # unraval index in the fortran way
    d = dim.shape[0]
    ret = np.zeros(d, dtype=np.int64)
    for k in range(d):
        ret[k] = i % dim[k]
        i = (i - ret[k]) / dim[k]
    return ret


# jth neighbour cell, for 2d of 9 cells, j -> (0, 8)
# and neighbours are cell_vec_i + (-1, 1) to (1, 1)
# 1. evaluate cell id of cell_i
# 2. unravel cell_i to cell_vec_i, using ibox (dimension of cells)
# 3. neighbours cell_vec_j = cell_vec_i + (-1, -1) (left, down) to (1, 1) (right, up)
#    the vectors (-1, -1) to (1, 1) are unraveled from 0~8 (3^2 for 2d) with shape (3, 3) (2d)
# 4. ravel cell_vec_j to cell id, cell_j, using ibox
@jit(nopython=True, nogil=True)
def jth_neighbour(vec_i, vec_j, dim):  # vecj is the shifting vector from (0,0) to (2,2) (2d)
    d = dim.shape[0]
    ret = (vec_i[0] + vec_j[0] - 1 + dim[0]) % dim[0]  # -1 to ensure moving from (-1, -1) to (1, 1) (2d)
    tmp = dim[0]
    for k in range(1, d):  # ravel index in the fortran way
        ret += ((vec_i[k] + vec_j[k] - 1 + dim[k]) % dim[k]) * tmp
        tmp *= dim[k]
    return ret


@jit(nopython=True, nogil=True)
def build_cell_struct(ib, n_cell):
    n_d = ib.shape[0]
    ret = np.zeros((n_cell, 3 ** n_d), dtype=np.int64)
    dim = np.zeros((n_d,), dtype=np.int64) + 3
    # j_vecs are shifting vector, from (0, 0) to (2, 2) (2d)
    j_vecs = np.empty((3 ** n_d, n_d), dtype=np.int64)
    for i in range(3 ** n_d):
        j_vecs[i] = unravel_index_f(i, dim)
    for ic in range(n_cell):
        vec_i = unravel_index_f(ic, ib)  # cell vector for ith cell
        for j in range(3 ** n_d):  # 9 neighbours (2d)
            # ravel (cell_vec_i + shifting vector - 1) using ibox
            jc = jth_neighbour(vec_i, j_vecs[j], ib)
            ret[ic, j] = jc
    return ret


class PPA_energy_minimizer(object):
    def __init__(self, n_dim=3, gpu=0):
        self.gpu = gpu
        self.build_nn = None
        self.non_bond_nn = None
        self.fire = None
        self.bond_force = None
        self.n_dim = n_dim
        self.n_atoms = 0
        self.rebuild = None
        self.cell_dim = None
        self.cell_map = None
        self.mol_info = None
        self.mol_length = None
        self.mol_table = None
        self.chain_ends = None
        self.is_end = None
        self.bond_hash = None
        self.lj_epsilon = 1.0
        self.lj_sigma = 1.0
        self.n_cell = None
        self.set_float_1d = None
        self.set_float_2d = None
        self.set_int_1d = None
        self.build_cell_list = None
        self.x = None
        self.box = None
        self.bond_info = None
        print("Start building calculators...")
        s = time.time()
        self.build_calculators()
        print("Done. Building time: %.4fs." % (time.time() - s))

    def build_calculators(self):
        _n_dim = self.n_dim
        with cuda.gpus[self.gpu]:
            @cuda.jit("int64(float64[:], float64[:], int64[:])", device=True)
            def _cell_id(_x, _box, _cell_dim):
                ret = 0
                n_cell = 1
                for i in range(0, _x.shape[0]):
                    tmp = _x[i] / box[i]
                    # if tmp < -0.5 or tmp > 0.5:
                    # return -1
                    ret = ret + floor((tmp + 0.5) * _cell_dim[i]) * n_cell
                    n_cell = n_cell * _cell_dim[i]
                return ret

            @cuda.jit(
                "void(float64[:, :], float64[:], int64[:], int64[:, :], int64[:], int64[:], int64[:])")
            def _build_cell_list(_x, _box, _cell_dim, _cell_list, _cell_count, _cell_index, _cell_max):
                pi = cuda.grid(1)
                if pi >= _x.shape[0]:
                    return
                # xi = cuda.local.array(ndim, dtype=float64)
                # for k in range(ndim):
                # xi[k] = x[pi, k]
                xi = _x[pi]
                ic = _cell_id(xi, _box, _cell_dim)
                _cell_index[pi] = ic
                index = cuda.atomic.add(_cell_count, ic, 1)
                if index < _cell_list.shape[1]:
                    _cell_list[ic, index] = pi
                else:
                    cuda.atomic.max(_cell_max, 0, index + 1)

            @cuda.jit("void(float64[:], float64[:], float64[:], float64[:])", device=True)
            def _pbc_vec(_x, _y, _box, _vec):
                for i in range(_x.shape[0]):
                    dx = _y[i] - _x[i]
                    dx = dx - _box[i] * floor(dx / _box[i] + 0.5)
                    _vec[i] = dx

            @cuda.jit("int64(int64, int64, int64[:], int64[:,:])", device=True)
            def _exclude(i, j, _mol_table, _mol_info):
                m_i = _mol_table[i]
                for k in _mol_info[m_i]:
                    if k == j:
                        return 1
                return 0

            @cuda.jit(
                "void(float64[:, :], float64, float64[:], int64[:], int64[:], int64[:],"
                "int64[:,:], int64[:, :], int64[:], int64[:, :], int64[:], int64[:,:], int64[:],"
                "float64[:,:])"
            )
            def _build_nn(_x, _rc2, _box, _cell_index, _cell_dim, _mol_table, _mol_info, _cell_list, _cell_count,
                          _cell_map, _n_count, _nn, _s, _last_x):
                i = cuda.grid(1)
                if i >= _x.shape[0]:
                    return
                pi = _x[i]
                _dr = cuda.local.array(_n_dim, float64)
                cell_i = _cell_index[i]
                num_n_i = 0
                for _j in range(3 ** _x.shape[1]):
                    cell_j = _cell_map[cell_i, _j]
                    for k in range(_cell_count[cell_j]):
                        j = _cell_list[cell_j, k]
                        if j <= i:
                            continue
                        ex_p = _exclude(i, j, _mol_table, _mol_info)
                        if ex_p == 1:
                            continue
                        pj = _x[j]
                        _pbc_vec(pi, pj, _box, _dr)
                        r2 = 0
                        for d in range(_x.shape[1]):
                            r2 += _dr[d] ** 2
                        if r2 < _rc2:
                            # print(i, nei, nn.shape[0])
                            if num_n_i < _nn.shape[1]:
                                _nn[i, num_n_i] = j
                            num_n_i += 1
                _n_count[i] = num_n_i
                cuda.atomic.max(_s, 0, num_n_i)
                for d in range(_x.shape[1]):
                    _last_x[i, d] = _x[i, d]

            @cuda.jit("void(float64, float64, float64, float64[:])", device=True)
            def _lj(_r2, _epsilon, _sigma, ret):
                sigma6 = pow(_sigma, 6.0)
                r6 = pow(_r2, 3.0)
                r2inv = 1.0 / _r2
                r6inv = 1.0 / r6
                r14inv = r6inv * r6inv * r2inv
                ret[1] = -24.0 * _epsilon * sigma6 * r14inv * (r6 - 2.0 * sigma6)
                ret[0] = 4 * _epsilon * (sigma6 * sigma6 * r6inv * r6inv - sigma6 * r6inv)

            @cuda.jit("void(float64[:, :], float64, float64[:], float64, float64,"
                      "int64[:,:], int64[:], float64[:, :], float64[:])"
                      )
            def _non_bond_nn(_x, _rc2, _box, _sigma, _epsilon, _nn, _n_count, _force, _energy):
                i = cuda.grid(1)
                if i >= _x.shape[0]:
                    return
                pi = _x[i]
                _dr = cuda.local.array(_n_dim, float64)
                _en = cuda.local.array(2, float64)
                for _j in range(_n_count[i]):
                    j = _nn[i, _j]
                    pj = _x[j]
                    _pbc_vec(pi, pj, _box, _dr)
                    r2 = 0
                    for d in range(_x.shape[1]):
                        r2 += _dr[d] ** 2
                    if r2 < _rc2:
                        _lj(r2, _epsilon, _sigma, _en)
                        cuda.atomic.add(_energy, 0, _en[0])
                        for d in range(_x.shape[1]):
                            f = - _en[1] * _dr[d]
                            cuda.atomic.add(_force[i], d, f)
                            cuda.atomic.add(_force[j], d, -f)

            @cuda.jit("void(float64[:, :], float64[:], int64[:, :],float64[:], float64[:, :])")
            def _bond_force(_x, _box, _bonds, _energy, _forces):
                i = cuda.grid(1)
                if i >= _bonds.shape[0]:
                    return
                a = _x[_bonds[i, 1]]
                b = _x[_bonds[i, 2]]
                _dr = cuda.local.array(_n_dim, float64)
                _pbc_vec(a, b, _box, _dr)
                r2 = 0.0
                for d in range(_x.shape[1]):
                    r2 += _dr[d] ** 2
                r = sqrt(r2)
                _f = -1000.0 * (r - 0.0) / r
                e = 500 * (r - 0.0) ** 2
                cuda.atomic.add(_energy, 0, e)
                # cuda.atomic.add(energy, bonds[i, 2], e/2)
                for d in range(_x.shape[1]):
                    f = _f * _dr[d]
                    cuda.atomic.add(_forces[_bonds[i, 1]], d, -f)
                    cuda.atomic.add(_forces[_bonds[i, 2]], d, f)

            @cuda.jit(
                "void(float64[:,:], float64[:], int64[:], float64[:,:], float64[:,:],"
                "float64, float64[:], float64[:], float64, float64, float64, float64, float64)"
            )
            def _fire(_x, _box, _is_end, _vel, _f, _dt, _fdt, _facoef, _facoef0, _finc, _fdec, _falpha, _fdtmax):
                i = cuda.grid(1)
                if i >= _x.shape[0]:
                    return
                vf = 0
                vv = 0
                ff = 0
                for d in range(_x.shape[1]):
                    vf += _vel[i, d] * _f[i, d]
                    vv += _vel[i, d] * _vel[i, d]
                    ff += _f[i, d] * _f[i, d]
                if vf < 0:
                    for d in range(_x.shape[1]):
                        _vel[i, d] = 0
                    _fdt[i] = _fdt[i] * _fdec
                    _facoef[i] = _facoef0
                if vf > 0:
                    cF = _facoef[i] * sqrt(vv / ff)
                    cV = 1 - _facoef[i]
                    for d in range(_x.shape[1]):
                        _vel[d] = cV * _vel[i, d] + cF * _f[i, d]
                    _fdt[i] = _fdt[i] * _finc
                    if _fdt[i] > _fdtmax:
                        _fdt[i] = _fdtmax
                    _facoef[i] = _facoef[i] * _falpha
                if _fdt[i] < 5e-4:
                    _fdt[i] = 1e-3  # faster, faster, faster
                if _is_end[i] < 1:
                    for d in range(_x.shape[1]):
                        _vel[i, d] = _vel[i, d] + _dt * _f[i, d]
                        _x[i, d] = _x[i, d] + _fdt[i] * _vel[i, d]
                        _x[i, d] = _x[i, d] - _box[d] * floor(_x[i, d] / _box[d] + 0.5)

            @cuda.jit("void(float64[:,:], float64[:,:], float64[:], float64, int64[:])")
            def _rebuild(_x, _y, _box, _r_buf2, _s):
                i = cuda.grid(1)
                if i >= _x.shape[0]:
                    return
                _dr = cuda.local.array(_n_dim, float64)
                _pbc_vec(_x[i], _y[i], _box, _dr)
                r2 = 0
                for d in range(_x.shape[1]):
                    r2 = r2 + _dr[d] ** 2
                if r2 > _r_buf2:
                    cuda.atomic.max(_s, 0, 1)

            @cuda.jit("void(float64[:], float64)")
            def _set_float_1d(arr, val):
                i = cuda.grid(1)
                if i >= arr.shape[0]:
                    return
                arr[i] = val

            @cuda.jit("void(int64[:], int64)")
            def _set_int_1d(arr, val):
                i = cuda.grid(1)
                if i >= arr.shape[0]:
                    return
                arr[i] = val

            @cuda.jit("void(float64[:,:], float64)")
            def _set_float_2d(arr, val):
                i = cuda.grid(1)
                if i >= arr.shape[0]:
                    return
                for d in range(arr.shape[1]):
                    arr[i, d] = val

            self.build_nn = _build_nn
            self.non_bond_nn = _non_bond_nn
            self.bond_force = _bond_force
            self.rebuild = _rebuild
            self.fire = _fire
            self.set_float_1d = _set_float_1d
            self.set_float_2d = _set_float_2d
            self.set_int_1d = _set_int_1d
            self.build_cell_list = _build_cell_list

    def process(self, x, box, bond_info, r_buf):
        print("Processing data...")
        s = time.time()
        self.n_atoms = x.shape[0]
        self.n_dim = box.shape[0]
        self.bond_info = bond_info
        bond_vectors = pbc(x[bond_info.T[1]] - x[bond_info.T[2]], box)
        bond_lengths = np.linalg.norm(bond_vectors, axis=-1)
        self.lj_sigma = bond_lengths.mean()
        self.lj_sigma = 1.0  # for test
        rc = self.lj_sigma  # lj cut off on sigma
        self.box = box
        self.cell_dim = np.asarray(box / (rc + r_buf), dtype=np.int64)
        self.mol_info, self.mol_length, self.bond_hash = molecules(bond_info, self.n_atoms)
        self.chain_ends = np.asarray(
            [[_ for _ in __ if _ != -1 and len(self.bond_hash[_]) == 1] for __ in self.mol_info],
            dtype=np.int64
        ).ravel()
        self.mol_table = build_mol_table(self.mol_info, self.n_atoms)
        self.is_end = np.zeros(self.n_atoms, dtype=np.int64)
        self.is_end[self.chain_ends] = 1
        self.n_cell = np.multiply.reduce(self.cell_dim)
        self.cell_map = build_cell_struct(self.cell_dim, self.n_cell)
        print("Cell dim: %d %d %d, lj sigma, epsilon: %.4f %.4f, Time costing %.4fs." % (
            *self.cell_dim, self.lj_sigma, self.lj_epsilon, time.time() - s))

    def minimize(self, x, box, bond_info, r_buf=0.25):
        x = pbc(x, box)  # ensure 0 at center.
        self.process(x, box, bond_info, r_buf)
        rc2 = (self.lj_sigma + r_buf) ** 2
        fire_finc = 1.6
        fire_fdec = 0.8
        fire_acoef0 = 0.2
        fire_falpha = 0.99
        fire_dt = 0.001
        dt = 0.005
        fire_fdtmax = 0.003
        v = np.zeros_like(x)
        forces = np.zeros_like(x)
        e0 = 0
        converge = False
        rebuild_p = True
        guess_cl = 30
        guess_nn = 10
        with cuda.gpus[self.gpu]:
            d_x = cuda.to_device(x)
            d_box = cuda.to_device(box)
            d_last_x = cuda.device_array((self.n_atoms, self.n_dim), dtype=np.float64)
            d_cell_dim = cuda.to_device(self.cell_dim)
            d_bond = cuda.to_device(bond_info)
            device = cuda.get_current_device()
            tpb = device.WARP_SIZE
            bpg = int(ceil(self.n_atoms / tpb))
            bpg_bond = int(ceil(bond_info.shape[0] / tpb))
            bpg_cell = int(ceil(self.n_cell / tpb))
            d_vel = cuda.to_device(v)
            d_is_end = cuda.to_device(self.is_end)
            d_mol_info = d_mols = cuda.to_device(self.mol_info)
            d_mol_table = cuda.to_device(self.mol_table)
            d_facoef = cuda.device_array((self.n_atoms,), dtype=np.float64)
            d_fdt = cuda.device_array((self.n_atoms,), dtype=np.float64)
            d_cell_map = cuda.to_device(self.cell_map)
            self.set_float_1d[bpg, tpb](d_facoef, fire_acoef0)
            self.set_float_1d[bpg, tpb](d_fdt, fire_dt)
            s_nn = cuda.pinned_array(1, dtype=np.int64)
            s_nn[0] = 0
            s_rebuild = cuda.pinned_array(1, dtype=np.int64)
            s_rebuild[0] = 0
            s_cell_max = cuda.pinned_array(1, dtype=np.int64)
            s_cell_max[0] = 0
            d_n_count = cuda.device_array((self.n_atoms,), dtype=np.int64)
            d_cell_count = cuda.device_array((self.n_cell,), dtype=np.int64)
            d_nb_energy = cuda.pinned_array(1, dtype=np.float64)
            d_bd_energy = cuda.pinned_array(1, dtype=np.float64)
            d_cell_index = cuda.device_array((self.n_atoms,), dtype=np.int64)
            d_forces = cuda.to_device(forces)
            d_cell_list = cuda.device_array((self.n_cell, guess_cl), dtype=np.int64)
            d_nn = cuda.device_array((self.n_atoms, guess_nn), dtype=np.int64)

        while not converge:
            d_bd_energy[0] = 0.0
            d_nb_energy[0] = 0.0
            self.set_float_2d[bpg, tpb](d_forces, 0.0)
            if rebuild_p:
                s_rebuild[0] = 0
                rebuild_p = False
                self.set_int_1d[bpg_cell, tpb](d_cell_count, 0)
                self.build_cell_list[bpg, tpb](d_x, d_box, d_cell_dim, d_cell_list, d_cell_count, d_cell_index,
                                               s_cell_max)
                if s_cell_max[0] > guess_cl:
                    guess_cl = s_cell_max[0]
                    self.set_int_1d[bpg_cell, tpb](d_cell_count, 0)
                    d_cell_list = cuda.device_array((self.n_cell, s_cell_max[0]), dtype=np.int64)
                    self.build_cell_list[bpg, tpb](d_x, d_box, d_cell_dim, d_cell_list, d_cell_count, d_cell_index,
                                                   s_cell_max)
                self.build_nn[bpg, tpb](
                    d_x, rc2, d_box, d_cell_index, d_cell_dim, d_mol_table, d_mol_info,
                    d_cell_list, d_cell_count, d_cell_map, d_n_count, d_nn, s_nn, d_last_x
                )
                if s_nn[0] > guess_nn:
                    guess_nn = s_nn[0]
                    d_nn = cuda.device_array((self.n_atoms, s_nn[0]), dtype=np.int64)
                    self.build_nn[bpg, tpb](
                        d_x, rc2, d_box, d_cell_index, d_cell_dim, d_mol_table, d_mol_info,
                        d_cell_list, d_cell_count, d_cell_map, d_n_count, d_nn, s_nn, d_last_x
                    )
            # lj_sigma^2 != rc2, rc2 = (rc + r_buf) ** 2, rc = lj cutoff = simga for purely repulsive force
            self.non_bond_nn[bpg, tpb](d_x, self.lj_sigma ** 2, d_box,
                                       self.lj_sigma,
                                       self.lj_epsilon,
                                       d_nn, d_n_count,
                                       d_forces,
                                       d_nb_energy)
            self.bond_force[bpg_bond, tpb](d_x, d_box, d_bond, d_bd_energy, d_forces)
            self.fire[bpg, tpb](d_x, d_box, d_is_end, d_vel, d_forces, dt, d_fdt, d_facoef,
                                fire_acoef0, fire_finc, fire_fdec, fire_falpha, fire_fdtmax)
            # d_forces.copy_to_host(forces)
            # d_X.copy_to_host(X)
            self.rebuild[bpg, tpb](d_x, d_last_x, d_box, r_buf ** 2, s_rebuild)
            if s_rebuild[0] > 0:
                rebuild_p = True
            cuda.synchronize()
            e = d_bd_energy[0] + d_nb_energy[0]
            if np.abs(e - e0) < 0.05:
                converge = True
            e0 = e
            print("b: %.4f, n: %.4f, total: %.4f" % (d_bd_energy[0], d_nb_energy[0], e))
            # break  # for test
        self.x = d_x.copy_to_host()
        cuda.synchronize()


xml = Xml(argv[1], needed=['position', 'bond'])
bond = xml.nodes['bond']
bond.T[0] = 0
bond = bond.astype(np.int64)
box = np.asarray([xml.box.lx, xml.box.ly, xml.box.lz], dtype=np.float64)
x = xml.nodes['position']
ppa = PPA_energy_minimizer(gpu=0, n_dim=3)
ppa.minimize(x, box, bond, 0.25)
np.savetxt('ppa_out.txt', ppa.x, fmt='%.4f')
