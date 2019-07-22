import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange, parallel
cimport openmp
from libc.math cimport floor,sqrt,pow
from libc.stdlib cimport malloc, free


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef long * unravel_index_f(long i, long[:] dim) nogil:
    cdef long k, n
    n = dim.shape[0]
    cdef long * ret = <long *> malloc(n * sizeof(long))
    for k in range(n):
        ret[k] = i % dim[k]
        i = (i - ret[k]) / dim[k]
    return ret


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef long jth_neighbour(long ic, long jc, long[:] dim, long[:] ndim) nogil:
    cdef long ret, d, tmp, k
    cdef long * tmpi
    cdef long * tmpj
    d = dim.shape[0]
    tmpi = unravel_index_f(ic, dim)  # ibox
    tmpj = unravel_index_f(jc, ndim)  # ndim for dimension
    ret = (tmpi[0] + tmpj[0] - 1 + dim[0]) % dim[0]  # re-ravel tmpi + tmpj - 1 to cell_j
    # -1 for indices from -1, -1, -1 to 1, 1, 1 rather than 0,0,0 to 2,2,2
    tmp = dim[0]
    for k in range(1, d):
        ret += ((tmpi[k] + tmpj[k] - 1 + dim[k]) % dim[k]) * tmp
        tmp *= dim[k]
    return ret


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef double pbc_dist(double[:] x, double[:] y, double[:] b) nogil:
    cdef long i, d
    cdef double tmp=0, r=0
    d = b.shape[0]
    for i in range(d):
        tmp = x[i]-y[i]
        tmp = tmp - b[i] * floor(tmp/b[i]+0.5)
        r = r + pow(tmp, 2)
    return sqrt(r)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef long cell_id(double[:] p, double[:] box, long[:] ibox) nogil:  # In the Fortran way
    cdef long ret, tmp, i, n
    n = p.shape[0]
    ret = <long> floor((p[0] / box[0] + 0.5) * ibox[0])
    tmp = ibox[0]
    for i in range(1, n):
        ret = ret + tmp * <long> floor((p[i] / box[i] + 0.5) * ibox[i])
        tmp = tmp * ibox[i]
    return ret


@cython.wraparound(False)
@cython.boundscheck(False)
def linked_cl(double[:, :] pos, double[:] box, long[:] ibox):
    cdef np.ndarray[np.int64_t, ndim=1] head
    cdef np.ndarray[np.int64_t, ndim=1] body
    cdef long i, n
    n = pos.shape[0]
    head = np.zeros(np.multiply.reduce(ibox), dtype=np.int64) - 1
    body = np.zeros(pos.shape[0], dtype=np.int64) - 1
    for i in range(n):
        ic = cell_id(pos[i], box, ibox)
        body[i] = head[ic]
        head[ic] = i
    return head, body


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def rdf(double[:,:] x, double[:,:] y, double[:] box, long[:] head, long[:] body, long[:] ibox, double bs, long nbins):
    cdef long i, j, k, l, n, d3, d, ic, jc
    cdef np.ndarray[np.double_t, ndim=2] ret
    cdef double r, r_cut
    cdef long[:] dim
    r_cut = nbins * bs
    n = x.shape[0]
    d = x.shape[1]
    d3 = 3 ** d
    ret = np.zeros((n, nbins), dtype=np.float)
    dim = np.zeros((d,), dtype=np.int64) + 3
    with nogil, parallel():
        for i in prange(n, schedule='dynamic'):
            ic = cell_id(x[i], box, ibox)
            for j in range(d3):
                jc = jth_neighbour(ic, j, ibox, dim)
                k = head[jc]
                while k != -1:
                    r = pbc_dist(x[i], y[k], box)
                    if r < r_cut:
                        l = <long> (r/bs)
                        ret[i,l]+=1
                    k = body[k]
    return ret
