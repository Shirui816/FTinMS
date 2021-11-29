from math import pow
from random import shuffle

import numba as nb
import numpy as np
import tqdm
from scipy.spatial import cKDTree

# from matplotlib import pyplot as plt
# import matplotlib.animation as animation


# Generate positions
# parameters
box_length = 20  # length of box
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
r_cut, r_buf = 2.5, 0.2
for ti in types:
    for tj in types:
        if ti == tj:
            parameters[ti, tj] = [-24 * 3, 1, r_cut ** 2]
        else:
            parameters[ti, tj] = [-24, 1, r_cut ** 2]


# functions
@nb.jit(nopython=True, nogil=True)
def pbc(x, d):
    return x - d * np.rint(x / d)


# -24 \epsilon \sigma^6 r^{-14} (r^6 - 2 \sigma^6) -- parameter[0] -> -24\epsilon, parameter[1] -> \simga^6
@nb.jit(nopython=True, nogil=True)
def lj_div_r(r2, parameter):
    res = np.zeros(2, dtype=np.float64)
    sigma6r6inv = pow(r2, -3) * parameter[1]
    res[1] = parameter[0] * parameter[1] * pow(r2, -7) * (pow(r2, 3) - 2 * parameter[1])
    res[0] = -parameter[0] / 6 * sigma6r6inv * (sigma6r6inv - 1)
    return res


@nb.jit(nopython=True, nogil=True)
def rebuild(div, td):
    for d in div:
        r2 = np.sum(d ** 2)
        if r2 > td:
            return True
    return False


@nb.jit(nopython=True, nogil=True)
def force(x, t_id, b, params, nn):
    forces = np.zeros_like(x)
    energy = np.zeros((x.shape[0],), dtype=np.float64)
    for i, ni in enumerate(nn):
        ti = t_id[i]
        for j in ni[ni > i]:
            tj = t_id[j]
            parameter = params[ti, tj]
            dij = pbc(x[j] - x[i], b)
            # dij = positions[j] - positions[i]
            rij2 = dij.dot(dij)
            if rij2 < parameter[-1]:
                lj = lj_div_r(rij2, parameter)
                f = lj[1] * dij
                # r_ij and f_ji
                forces[i] += -f  # f^j = f_{lj} / dr * dr^j, dr^j = dr^x, dr^y, dr^z
                forces[j] += f
                energy[i] += lj[0] / 2
                energy[j] += lj[0] / 2
    return forces, energy


# integrate, vv
dt = 0.005
n_steps = 20000
# v = np.zeros_like(positions)  # initialize v
v = np.random.random(positions.shape)
v = v - v.mean(axis=0)
v = v / (np.mean(v ** 2) * 0.5 * n_dim) ** 0.2
positions = pbc(positions, box)
print("Run simulation with %d %d-D particles." % positions.shape)

g = 0.5 * (n_dim * positions.shape[0] + 1)  # nh g
Q = n_dim * positions.shape[0] * 0.2  # nh Q
t_target = 1.0  # target temperature
zeta = 0.0  # initial zeta for nh
rebuild_p = True
_d0 = np.zeros_like(positions)

# movie
# frames = []

# init progress bar
ts_bar = tqdm.trange(n_steps, desc='00000th step, t:0.0000, e:0.0000', leave=True, unit='step')
at = np.zeros_like(positions)

# nve
for ts in ts_bar:
    if rebuild_p:
        _d0 = np.copy(positions)
        _positions = positions + box / 2
        tree = cKDTree(_positions, boxsize=box)
        _nn = tree.query_ball_point(_positions, r_cut + 1 * r_buf)
        nn = np.zeros([len(_nn), len(max(_nn, key=lambda x: len(x)))], dtype=np.int64) - 1
        for index, item in enumerate(_nn):
            nn[index][0:len(item)] = item
    v = v + 0.5 * at * dt
    positions = pbc(positions + v * dt, box)
    rebuild_p = rebuild(pbc(positions - _d0, box), r_buf ** 2)
    ft, et = force(positions, typeid, box, parameters, nn)
    at = ft
    v = v + 0.5 * at * dt
    K = np.sum(v ** 2) * 0.5
    # energy should be constant here
    ts_bar.set_description("%05dth step, t:%.4f, e:%.4f" % (ts, K / positions.shape[0], et.sum() + K))

# nh
# for ts in ts_bar:
#     if rebuild_p:
#         _d0 = np.copy(positions)
#         _positions = positions + box / 2
#         tree = cKDTree(_positions, boxsize=box)
#         _nn = tree.query_ball_point(_positions, r_cut + r_buf)
#         nn = np.zeros([len(_nn), len(max(_nn, key=lambda x: len(x)))], dtype=np.int64) - 1
#         for index, item in enumerate(_nn):
#             nn[index][0:len(item)] = item
#     at = at - zeta * v  # all 0 for 1th step
#     positions = pbc(positions + v * dt + 0.5 * at * dt ** 2, box)
#     rebuild_p = rebuild(pbc(positions - _d0, box), r_buf ** 2)
#     vh = v + 0.5 * dt * at
#     # if not ts % 100:
#     #    frames.append(positions)
#     ft, et = force(positions, typeid, box, parameters, nn)
#     at = ft
#     K = 0.5 * np.sum(v ** 2)
#     zeta_h = zeta + 0.5 * dt / Q * (K - g * t_target)
#     zeta = zeta_h + 0.5 * dt / Q * (0.5 * np.sum(vh ** 2) - g * t_target)
#     v = (vh + 0.5 * at * dt) / (1 + 0.5 * dt * zeta)
#     ts_bar.set_description("%05dth step, t:%.4f, e:%.4f" % (ts, K / positions.shape[0], et.sum() + K))

# movie
# ani = animation.FuncAnimation(fig, animate, interval=12.5, frames=frames)
# ani.save('animation.gif', writer='pillow')
