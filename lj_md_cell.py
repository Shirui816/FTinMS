from math import pow
from random import shuffle

import numba as nb
import numpy as np
import tqdm

# from matplotlib import pyplot as plt
# import matplotlib.animation as animation


# Generate positions
# parameters
box_length = 10  # length of box
unit_cell = np.array(
    [[0, 0], [1, 1], [0, 1], [1, 0]]
)  # particles in a unit cell (x, y, z) for 3D simulations, 2d for animation
uc_length = 2.0  # unit cell length
types_uc = [0, 1, 0, 1]  # type index of 4 particles in a unit cell

# mesh box and put particles in box
n_dim = unit_cell.shape[1]  # spatial dimension of system
types = list(set(types_uc))
n_types = len(types)
box = np.array([box_length] * n_dim, dtype=np.float64)
box_vec = np.linspace(-box_length / 2, box_length / 2, int(box_length / uc_length),
                      endpoint=False)  # how many cells in one dimension
box_mesh = np.vstack([_.ravel() for _ in np.meshgrid(*((box_vec,) * n_dim))]).T  # mesh box
positions = (unit_cell + box_mesh[:, None]).reshape(-1, n_dim)  # move unit cell to mesh grids
typeids = types_uc * box_mesh.shape[0]
shuffle(typeids)  # mix particles by mixing ids ~~
typeid = np.asarray(typeids, dtype=np.int64)
# c = np.array(['k', 'r'])[typeid]

# animate functions
# fig, ax = plt.subplots()
# def animate(frame):
#    fig.clear()
#    ax = fig.add_subplot(111, aspect='equal', xlim=(-box_length/2, box_length/2), ylim=(-box_length/2, box_length/2))
#    ax.set_xlim(-box_length/2, box_length/2)
#    ax.set_ylim(-box_length/2, box_length/2)
#    s = ax.scatter(frame.T[0], frame.T[1],c=c)


# Simulation
# parameters
# -24 \epsilon \sigma^6 r^{-14} (r^6 - 2 \sigma^6) -- parameter[0] -> -24\epsilon,
# parameter[1] -> \simga^6, parameter[2] -> r_cut^2
parameters = np.empty((2, 2, 3), dtype=np.float64)
r_cut = 2.5
for ti in types:
    for tj in types:
        if ti == tj:
            parameters[ti, tj] = [-24, 1, r_cut ** 2]
        else:
            parameters[ti, tj] = [-24, 1, (2 ** (1 / 6)) ** 2]


# functions
@nb.jit(nopython=True, nogil=True)
def pbc(x, d):
    return x - d * np.rint(x / d)


# -24 \epsilon \sigma^6 r^{-14} (r^6 - 2 \sigma^6) -- parameter[0] -> -24\epsilon, parameter[1] -> \simga^6
@nb.jit(nopython=True, nogil=True)
def lj_div_r(r2, parameter):
    return parameter[0] * parameter[1] * pow(r2, -7) * (pow(r2, 3) - 2 * parameter[1])


@nb.jit(nopython=True, nogil=True)
def cell_id(x, b, ib):
    ret = int((x[0] / b[0] + 0.5) * ib[0])
    tmp = ib[0]
    for i in range(1, x.shape[0]):
        ret += tmp * int((x[i] / b[i] + 0.5) * ib[i])
        tmp = tmp * ib[i]
    return ret


@nb.jit(nopython=True, nogil=True)
def linked_cl(x, b, ib, h, bd):
    for i in range(x.shape[0]):
        ic = cell_id(x[i], b, ib)
        bd[i] = h[ic]
        h[ic] = i


def build_cell_list(x, b, ib):
    h = np.zeros(np.multiply.reduce(ib), dtype=np.int64) - 1
    bd = np.zeros((x.shape[0],), dtype=np.int64) - 1
    linked_cl(x, b, ib, h, bd)
    return h, bd


@nb.jit(nopython=True, nogil=True)
def unravel_index_f(i, dim):
    d = dim.shape[0]
    ret = np.zeros(d, dtype=np.int64)
    for k in range(d):
        ret[k] = i % dim[k]
        i = (i - ret[k]) / dim[k]
    return ret


@nb.jit(nopython=True, nogil=True)
def jth_neighbour(veci, vecj, dim):
    d = dim.shape[0]
    ret = (veci[0] + vecj[0] - 1 + dim[0]) % dim[0]
    tmp = dim[0]
    for k in range(1, d):
        ret += ((veci[k] + vecj[k] - 1 + dim[k]) % dim[k]) * tmp
        tmp *= dim[k]
    return ret


def build_cell_struct(ib):
    n_cell = np.multiply.reduce(ib)
    n_d = ib.shape[0]
    ret = np.zeros((n_cell, 3 ** n_d), dtype=np.int64)
    dim = np.zeros((n_d,), dtype=np.int64) + 3
    j_vecs = np.empty((3 ** n_d, n_d), dtype=np.int64)
    for i in range(3 ** n_d):
        j_vecs[i] = unravel_index_f(i, dim)
    for ic in range(n_cell):
        veci = unravel_index_f(ic, ib)
        for j in range(3 ** n_d):
            jc = jth_neighbour(veci, j_vecs[j], ib)
            ret[ic, j] = jc
    return ret


@nb.jit(nopython=True, nogil=True)
def force(x, t_id, b, params, ib, h, bd, cm):
    forces = np.zeros_like(x)
    for i in range(x.shape[0]):
        ti = t_id[i]
        ic = cell_id(x[i], b, ib)
        # print(cell_map[ic], ic)
        for jc in cm[ic]:
            j = h[jc]
            while j != -1:
                if j <= i:
                    j = bd[j]
                    continue
                tj = t_id[j]
                parameter = params[ti, tj]
                dij = pbc(x[j] - x[i], b)
                # dij = positions[j] - positions[i]
                rij2 = dij.dot(dij)
                if rij2 < parameter[-1]:  # parameter[-1] r_{cut}^2
                    f = lj_div_r(rij2, parameter) * dij
                    # r_ij and f_ji
                    forces[i] += -f  # f^j = f_{lj} / dr * dr^j, dr^j = dr^x, dr^y, dr^z
                    forces[j] += f
                j = bd[j]
    return forces


# @nb.jit(nopython=True, nogil=True)
# def force_bf(x, t_id, b, params):
#    forces = np.zeros_like(x)
#    for i in range(x.shape[0] - 1):
#        ti = t_id[i]
#        for j in range(i + 1, x.shape[0]):
#            tj = t_id[j]
#            parameter = params[ti, tj]
#            dij = pbc(x[j] - x[i], b)
#            # dij = positions[j] - positions[i]
#            rij2 = dij.dot(dij)
#            if rij2 < parameter[-1]:  # parameter[-1] r_{cut}^2
#                f = lj_div_r(rij2, parameter) * dij
#                # r_ij and f_ji
#                forces[i] += -f  # f^j = f_{lj} / dr * dr^j, dr^j = dr^x, dr^y, dr^z
#                forces[j] += f
#    return forces


# integrate, vv
dt = 0.008
n_steps = 30000
# v = np.zeros_like(positions)  # initialize v
v = np.random.random(positions.shape)
v = v - v.mean(axis=0)
v = v / (np.mean(v ** 2) * 0.5 * n_dim) ** 0.2
positions = pbc(positions, box)

print("Run simulation with %d %d-D particles." % positions.shape)

g = 0.5 * (n_dim * positions.shape[0] + 1)  # nh g
Q = n_dim * positions.shape[0] * 0.5  # nh Q
t_target = 1.0  # target temperature
zeta = 0.0  # initial zeta for nh

# parameters
ibox = np.asarray(box / r_cut, dtype=np.int64)
cell_map = build_cell_struct(ibox)

# movie
# frames = []

# init progress bar
ts_bar = tqdm.trange(n_steps, desc='00000th step, t:0.0000', leave=True, unit='step')

# nve
# for ts in ts_bar:
#    head, body = build_cell_list(positions, box, ibox)
#    a0 = force(positions, typeid, box, parameters, ibox, head, body, cell_map)
#    a0 = force_bf(positions, typeid, box, parameters)
#    v = v + 0.5 * a0 * dt
#    positions = pbc(positions + v * dt, box)
#    at = force(positions, typeid, box, parameters, ibox, head, body, cell_map)
#    at = force_bf(positions, typeid, box, parameters)  # brute force version
#    v = v + 0.5 * at * dt
#    ts_bar.set_description("%05dth step, t:%.4f" % (ts, 0.5 * np.mean(v ** 2) * n_dim))

# nh
for ts in ts_bar:
    head, body = build_cell_list(positions, box, ibox)
    # if not ts % 100:
    #    frames.append(positions)
    a0 = force(positions, typeid, box, parameters, ibox, head, body, cell_map) - zeta * v
    # a0 = force_bf(positions, typeid, box, parameters) - zeta * v  # brute force version
    positions = pbc(positions + v * dt + 0.5 * a0 * dt ** 2, box)
    vh = v + 0.5 * dt * a0
    at = force(positions, typeid, box, parameters, ibox, head, body, cell_map)
    # at = force_bf(positions, typeid, box, parameters)  # brute force version
    K = 0.5 * np.sum(v ** 2)
    zeta_h = zeta + 0.5 * dt / Q * (K - g * t_target)
    zeta = zeta_h + 0.5 * dt / Q * (0.5 * np.sum(vh ** 2) - g * t_target)
    v = (vh + 0.5 * at * dt) / (1 + 0.5 * dt * zeta)
    ts_bar.set_description("%05dth step, t:%.4f" % (ts, K / positions.shape[0]))


# movie
# ani = animation.FuncAnimation(fig, animate, interval=12.5, frames=frames)
# ani.save('animation.gif', writer='pillow')


# backup functions, parallel version of force, maybe slow for small systems
@nb.jit(nopython=True, nogil=True, parallel=True)
def force_parallel(x, t_id, b, params):
    forces = np.zeros_like(x)
    for i in nb.prange(x.shape[0]):
        ti = t_id[i]
        for j in range(x.shape[0]):
            if i == j:
                continue
            tj = t_id[j]
            parameter = params[ti, tj]
            dij = pbc(x[j] - x[i], b)
            # dij = positions[j] - positions[i]
            rij2 = dij.dot(dij)
            if rij2 < parameter[-1]:  # parameter[-1] r_{cut}^2
                f = lj_div_r(rij2, parameter) * dij
                # r_ij and f_ji
                forces[i] += -f  # f^j = f_{lj} / dr * dr^j, dr^j = dr^x, dr^y, dr^z
    return forces
