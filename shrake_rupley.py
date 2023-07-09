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
    if np.sum(d ** 2) <= 1e-6:
        return r
    return r - d * np.rint(r / d)


def shrake_rupley_sasa(x, radiivdw, probe_radius=1.4, n_points=960, box=np.zeros(3)):
    sph = fibonacci_sphere(n_points)
    radiivdw = radiivdw + probe_radius
    twice_maxradii = radiivdw.max() * 2.
    asa_ary = np.zeros((len(x)), dtype=np.int64)
    periodic_p = False
    if np.sum(box ** 2) > 1e-6:
        periodic_p = True
    if periodic_p:
        kdt = cKDTree(x, boxsize=box)
    else:
        kdt = cKDTree(x)
    for i in range(len(x)):
        r_i = radiivdw[i]
        s_on_i = pbc(sph * r_i + x[i], box)
        avail_sph = np.ones(n_points)
        if periodic_p:
            kdt_sph = cKDTree(s_on_i, boxsize=box)
        else:
            kdt_sph = cKDTree(s_on_i)
        for j in kdt.query_ball_point(x[i], twice_maxradii):
            if i != j:
                dij = np.linalg.norm(pbc(x[i] - x[j], box), axis=-1)
                if dij < r_i + radiivdw[j]:
                    avail_sph[kdt_sph.query_ball_point(x[j], radiivdw[j])] = 0
        asa_ary[i] = avail_sph.sum()
    f = 4. * np.pi * radiivdw ** 2. / n_points
    asa_ary = asa_ary * f
    return asa_ary
  
