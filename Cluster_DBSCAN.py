import time
import numpy as np
import pandas as pd
from numba import cuda
from sys import argv
from math import floor, sqrt, ceil
from sklearn.cluster import DBSCAN

# For usage of cuda.jit, run `conda install cudatoolkit=9.0` after the
# installation of Anaconda env. numba.cuda cannot compile under
# latest `cudatoolkit=9.2' currently.


def pbc(p, d):
    return p - d * np.round(p/d)


def mid(pos, n, box):
    r'''Same as circular mean. \sum_i\sin(\alpha_i)->
    \sum_ig_i\sin(\alpha_{g_i}) for \alpha_{g_i} is angle
    in the bin of \alpha_{g_i}+\delta\alpha and g_i is the
    count of angles in this bin.
    '''
    ret = []
    w = np.exp(-2j*np.pi*np.arange(n) / n)
    for i in range(box.shape[0]):
        y, _ = np.histogram(pos.T[i], bins=n, range=(-box[i]/2, box[i]/2))
        if not np.all(y > 0):  # com if procolate, else median
            y = (y > 0).astype(np.float)  # no weight for median
        a = np.angle((w.dot(y)).conj()) % (2 * np.pi)
        # box.lo + a/2/np.pi * (box.hi-box.lo)
        ret.append(-box[i]/2 + box[i] * a/2/np.pi)
    return np.array(ret)


@cuda.jit('float64(float64[:], float64[:], float64[:])', device=True)
def pbc_dist_cu(a, b, box):
    tmp = 0
    for i in range(a.shape[0]):
        d = b[i] - a[i]
        d = d - floor(d/box[i]+0.5)*box[i]
        tmp += d * d
    return sqrt(tmp)


@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64[:,:])')
def pbc_pairwise_distance(X, Y, box, res):
    i, j = cuda.grid(2)
    if i >= X.shape[0] or j >= Y.shape[0]:
        return
    if j >= i:
        return
    r = pbc_dist_cu(X[i], Y[j], box)
    res[j, i] = r
    res[i, j] = r


# Variables
box = np.array([50.0, 50.0, 50.0], dtype=np.float)  # Modify box here.
minCls = 10  # clusters that are smaller than `minCLS' will be ignored.
epsilon = 1.08
minPts = 5
# Optimized for DPD, RTFM for these 2 parameters pls.

# Usage: python <this_file> <position_file>
# <position_file> should only contain positions:
# x0 y0 z0
# x1 y1 z1
# ...
pos = pd.read_csv(argv[1], squeeze=1, header=None,
                  delim_whitespace=True).values
# End of Variables

ret = np.zeros((pos.shape[0],)*2, dtype=np.float)

device = cuda.get_current_device()
tpb2d = (device.WARP_SIZE,)*2
bpg2d = (ceil(pos.shape[0]/tpb2d[0]), ceil(pos.shape[0]/tpb2d[1]))

# Very fast, RAM has limits, up to ~30k coordinates on GTX1080
s = time.time()
pbc_pairwise_distance[bpg2d, tpb2d](pos, pos, box, ret)
db_fitted = DBSCAN(metric='precomputed',
                   n_jobs=-1, eps=epsilon, min_samples=minPts).fit(ret)
print('%.4f secs.' % (time.time()-s))

# Relatively slow, however no RAM limits.
# from functions import cpbc
# or define a pbc metric function, cython is recommanded.
# s = time.time()
# db_fitted = DBSCAN(metric=cpbc, metric_params={'box':box},
#                    n_jobs=-1, eps=epsilon, min_samples=minPts).fit(pos)
# sklearn version >= 0.19
# print('%.4f secs.' % (time.time()-s))

clusters = [pos[db_fitted.labels_ == _]
            for _ in list(set(db_fitted.labels_)) if _ != -1]
noises = pos[db_fitted.labels_ == -1]

# Output is for 3-D cases
# cluster_meta.txt contains meta info of clusters:
# cluster id, number of particles in the cluster, eigen values and
# corresponding eigen vectors are given for each cluster. Cluster
# positions are pbc-unwrapped and COM-zeroed.
# cluster_meta.txt:
# <cls_id> <n_pars> <eig_1> <eig_2> <eig_3> <v_1(xyz)> <v_2(xyz)> <v_3(xyz)>
# cls_id.xyz:
# xyz file (cls_id.xyz) contains.
# center-zeroed and pbc-unwrapped cluster coordinates.

o = open('cluster_meta.txt', 'w')
# modify following codes if 2-D outputs are desired:
fmt = '%04d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n'
for i, cluster in enumerate(clusters):
    if cluster.shape[0] < minCls:
        continue
    p = cluster
    p_pbc = pbc(p-mid(p, 10000, box), box)
    p_pbc -= np.mean(p_pbc, axis=0)
    rgTensor = p_pbc.T.dot(p_pbc)/p_pbc.shape[0]
    e, v = np.linalg.eig(rgTensor)
    o.write(fmt % (i, p.shape[0], e[0], e[1], e[2],
                   v.T[0, 0], v.T[0, 1], v.T[0, 2],
                   v.T[1, 0], v.T[1, 1], v.T[1, 2],
                   v.T[2, 0], v.T[2, 1], v.T[2, 2]))
    oo = open('%04d.xyz' % (i), 'w')
    oo.write('%d\nmeta\n' % (p_pbc.shape[0]))
    for _ in p_pbc:
        oo.write('C %.4f %.4f %.4f\n' % (_[0], _[1], _[2]))
    oo.close()
o.close()
# End of Output
