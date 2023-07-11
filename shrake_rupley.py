import numpy as np
import numba as nb
from scipy.spatial import cKDTree

@nb.jit(nopython=True, nogil=True)
def fibonacci_sphere(n=1000):
    points = np.empty((n, 3))
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points[i] = (x, y, z)

    return points


@nb.jit(nopython=True, nogil=True)
def pbc(r, d):
    return r - d * np.rint(r / d)


def shrake_rupley_sasa(x, radiivdw, probe_radius=1.4, n_points=960, box=None):
    sph = fibonacci_sphere(n_points)
    radiivdw = radiivdw + probe_radius
    cut_off = radiivdw.max() * 2.
    asa_ary = np.zeros((len(x)), dtype=np.int64)
    if box is None:
        box = np.ones(3) * (x.max(axis=0) - x.min(axis=0) + 2 * cut_off)
    x = np.mod(x, box)
    kdt = cKDTree(x, boxsize=box)
    for i in range(len(x)):
        r_i = radiivdw[i]
        s_on_i = np.mod(sph * r_i + x[i], box)
        avail_sph = np.ones(n_points)
        kdt_sph = cKDTree(s_on_i, boxsize=box)
        for j in kdt.query_ball_point(x[i], cut_off):
            if i != j:
                dij = np.linalg.norm(pbc(x[i] - x[j], box), axis=-1)
                if dij < r_i + radiivdw[j]:
                    avail_sph[kdt_sph.query_ball_point(x[j], radiivdw[j])] = 0
        asa_ary[i] = avail_sph.sum()
    f = 4. * np.pi * radiivdw ** 2. / n_points
    asa_ary = asa_ary * f
    return asa_ary
  
