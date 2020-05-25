# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange, parallel
cimport openmp
from libc.math cimport floor,sqrt,pow
from libc.stdlib cimport malloc, free
import multiprocessing


cdef long * unravel_index_f(long i, long[:] dim) nogil:
    cdef long k, d
    d = dim.shape[0]
    cdef long * ret = <long *> malloc(d * sizeof(long))
    for k in range(d):
        ret[k] = i % dim[k]
        i = (i - ret[k]) / dim[k]
    return ret


cdef long ravel_index_f(long * vec, long[:] dim) nogil:
    cdef long ret, d, tmp, k
    d = dim.shape[0]
    ret = (vec[0] + dim[0]) % dim[0]
    tmp = dim[0]
    for k in range(1,d):
        ret += ((vec[k] + dim[k]) % dim[k]) * tmp
        tmp *= dim[k]
    return ret


cdef long jth_neighbour(long * veci, long * vecj, long[:] dim) nogil:
    cdef long ret, d, tmp, k
    d = dim.shape[0]
    ret = (veci[0] + vecj[0] - 1 + dim[0]) % dim[0]
    # re-ravel tmpi + tmpj - 1 to cell_j
    # -1 for indices from -1, -1, -1 to 1, 1, 1 rather than 0,0,0 to 2,2,2
    tmp = dim[0]
    for k in range(1, d):
        ret += ((veci[k] + vecj[k] - 1 + dim[k]) % dim[k]) * tmp
        tmp *= dim[k]
    return ret
    

cdef double pbc_dist(double[:] x, double[:] y, double[:] b) nogil:
    cdef long i, d
    cdef double tmp=0, r=0
    d = b.shape[0]
    for i in range(d):
        tmp = x[i]-y[i]
        tmp = tmp - b[i] * floor(tmp/b[i]+0.5)
        r = r + pow(tmp, 2)
    return sqrt(r)


cdef long cell_id(double[:] p, double[:] box, long[:] ibox) nogil:
    # In the Fortran way
    cdef long ret, tmp, i, n
    n = p.shape[0]
    ret = <long> floor((p[0] / box[0] + 0.5) * ibox[0])
    tmp = ibox[0]
    for i in range(1, n):
        ret = ret + tmp * <long> floor((p[i] / box[i] + 0.5) * ibox[i])
        tmp = tmp * ibox[i]
    return ret


cdef void linked_cl(double[:, :] pos, double[:] box, long[:] ibox, long[:] head, long[:] body) nogil:
    cdef long i, n, ic
    n = pos.shape[0]
    for i in range(n):
        ic = cell_id(pos[i], box, ibox)
        body[i] = head[ic]
        head[ic] = i


def rdf(double[:,:] x, double[:,:] y, double[:] box, double bs, long nbins):
    cdef long i, j, k, l, n, m, d3, d, ic, jc
    cdef np.ndarray[np.double_t, ndim=2] ret
    cdef long[:] head, body, ibox, dim
    cdef double r, r_cut
    cdef long ** j_vecs
    cdef long * veci
    cdef int num_threads, thread_num
    num_threads = multiprocessing.cpu_count()
    r_cut = nbins * bs
    n = x.shape[0]
    d = x.shape[1]
    m = y.shape[0]
    d3 = 3 ** d
    ibox = np.zeros((d,), dtype=np.int64)
    for i in range(d):
        ibox[i] = <long> floor(box[i] / r_cut + 0.5)
    head = np.zeros(np.multiply.reduce(ibox), dtype=np.int64) - 1
    body = np.zeros((m,), dtype=np.int64) - 1
    linked_cl(y, box, ibox, head, body)
    ret = np.zeros((num_threads, nbins), dtype=np.float)
    # or ret = (n, nbins) for each particles
    dim = np.zeros((d,), dtype=np.int64) + 3
    j_vecs = <long **> malloc(sizeof(long *) * d3)
    for i in range(d3):
        j_vecs[i] = unravel_index_f(i, dim)  
    with nogil, parallel(num_threads=num_threads):
        for i in prange(n, schedule='static'):
            ic = cell_id(x[i], box, ibox)
            thread_num = openmp.omp_get_thread_num()
            veci = unravel_index_f(ic, ibox)
            for j in range(d3):
                jc = jth_neighbour(veci, j_vecs[j], ibox)
                k = head[jc]
                while k != -1:
                    r = pbc_dist(x[i], y[k], box)
                    l = <long> (r/bs)
                    if l < nbins:
                        ret[thread_num, l]+=1
                    k = body[k]
            free(veci)
    free(j_vecs)
    return np.sum(ret, axis=0)
