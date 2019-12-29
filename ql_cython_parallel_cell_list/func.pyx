# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange, parallel
cimport openmp
from libc.math cimport floor,sqrt,pow,sin,cos,fmod,pi,fabs,atan2, acos
from libc.stdlib cimport malloc, free
import multiprocessing
from scipy.special.cython_special cimport gamma, sph_harm

cdef extern from "complex.h":
    double complex exp(double complex)
    double cabs(double complex)

cdef double legendre(long l, long m, double x) nogil:
    cdef double somx2, fact, pmm1, pll, pmmp1
    cdef double pmm = 1.0
    cdef long ll
    if m > 0:
        somx2 = sqrt((1. - x) * (1. + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    for ll in range(m + 2, l + 1):
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll

cdef double complex sphereHarmonics(long l, long m, double cosTheta, double phi) nogil:
    cdef long m1
    cdef double c
    cdef double complex y
    m1 = <long>fabs(m)
    c = sqrt((2 * l + 1) * gamma(l - m1 + 1.0) / (4 * pi * gamma(l + m1 + 1.0)))
    c *= legendre(l, m1, cosTheta)
    y =  cos(m * phi) + 1j * sin(m * phi)
    if fmod(m, 2) == -1.:
        y *= -1
    return y * c + 0j

def test(long l, long m, double cosTheta, double phi):
    return cabs(sphereHarmonics(l, m, cosTheta, phi))


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


cdef void linked_cl(double[:, :] pos, double[:] box, long[:] ibox, long[:] head, long[:] body, long[:] ct, long ncell) nogil:
    cdef long i, n, ic
    n = pos.shape[0]
    for i in range(n):
        ic = cell_id(pos[i], box, ibox)
        body[i] = head[ic]
        head[ic] = i
        ct[ic] += 1


# def ql(double[:,:] x, double[:,:] y, long[:] ls, double[:] box, double rc):
#     cdef long i, j, k, l, n, m, d3, d, ic, jc, nl, ml, _m
#     cdef np.ndarray[np.complex128_t, ndim=3] ret
#     cdef np.ndarray[np.double_t, ndim=1] count
#     cdef long[:] head, body, ibox, dim
#     cdef double dr, dx, dy, dz, cosTheta, phi
#     cdef long ** j_vecs
#     cdef long * veci
#     cdef int num_threads, thread_num
#     num_threads = multiprocessing.cpu_count()
#     n = x.shape[0]
#     d = x.shape[1]
#     m = y.shape[0]
#     nl = ls.shape[0]
#     ml = 2 * max(ls) + 1
#     d3 = 3 ** d
#     ibox = np.zeros((d,), dtype=np.int64)
#     for i in range(d):
#         ibox[i] = <long> floor(box[i] / rc + 0.5)
#     head = np.zeros(np.multiply.reduce(ibox), dtype=np.int64) - 1
#     body = np.zeros((m,), dtype=np.int64) - 1
#     linked_cl(y, box, ibox, head, body)
#     ret = np.zeros((n, nl, ml), dtype=np.complex128)
#     count = np.zeros((n,), dtype=np.float)
#     # or ret = (n, nl) for each particles
#     dim = np.zeros((d,), dtype=np.int64) + 3
#     j_vecs = <long **> malloc(sizeof(long *) * d3)
#     for i in range(d3):
#         j_vecs[i] = unravel_index_f(i, dim)  
#     with nogil, parallel(num_threads=num_threads):
#         for i in prange(n, schedule='static'):
#             ic = cell_id(x[i], box, ibox)
#             thread_num = openmp.omp_get_thread_num()
#             veci = unravel_index_f(ic, ibox)
#             for j in range(d3):
#                 jc = jth_neighbour(veci, j_vecs[j], ibox)
#                 k = head[jc]
#                 while k != -1:
#                     dx = y[k, 0] - x[i, 0]
#                     dy = y[k, 1] - x[i, 1]
#                     dz = y[k, 2] - x[i, 2]
#                     dx = dx - box[0] * floor(dx / box[0] + 0.5)
#                     dy = dy - box[1] * floor(dy / box[1] + 0.5)
#                     dz = dz - box[2] * floor(dy / box[2] + 0.5)
#                     dr = sqrt(dx * dx  + dy * dy + dz * dz)
#                     if 1e-5 < dr < rc:
#                         count[i] += 1.0
#                         #cosTheta = dz / dr # hand-made sph_harm
#                         #phi = atan2(dy, dx) + pi
#                         cosTheta = atan2(dy, dx) + pi
#                         phi = acos(dz/dr) # scipy sph_harm exp(-i \theta) sin(\phi)...
#                         for l in range(nl):
#                             for _m in range(-ls[l], ls[l]+1):
#                                 #ret[i, l, _m+ls[l]] += sphereHarmonics(ls[l], _m, cosTheta, phi)
#                                 ret[i, l, _m+ls[l]] += sph_harm(_m, ls[l], cosTheta, phi)
#                     k = body[k]
#             free(veci)
#     free(j_vecs)
#     count[count < 1.0] = 1.0
#     #return np.sqrt(4 * np.pi * np.sum(np.abs(ret) ** 2, axis=-1)/ (2 * ls + 1).reshape(1, -1))
#     #return ret, count
#     #return np.sqrt(4 * np.pi * np.sum(np.abs(ret/c[:,None,None]) ** 2, axis=-1)/ (2 * ls + 1).reshape(1, -1))
#     return ret, count

def neighbour_list(double[:,:] x, double[:] box, double rc):
    cdef long i, j, k, l, n, m, d3, d, ic, jc, mc, ncell
    cdef np.ndarray[np.int64_t, ndim=2] ret
    cdef np.ndarray[np.int64_t, ndim=1] count
    cdef long[:] head, body, ibox, dim, ct
    cdef double dr, dx, dy, dz
    cdef long ** j_vecs
    cdef long * veci
    cdef int num_threads, thread_num
    num_threads = multiprocessing.cpu_count()
    n = x.shape[0]
    d = x.shape[1]
    d3 = 3 ** d
    ncell = 1
    ibox = np.zeros((d,), dtype=np.int64)
    for i in range(d):
        ibox[i] = <long> floor(box[i] / rc + 0.5)
        ncell *= ibox[i]
    head = np.zeros(np.multiply.reduce(ibox), dtype=np.int64) - 1
    body = np.zeros((n,), dtype=np.int64) - 1
    ct = np.zeros((ncell,), dtype=np.int64)
    linked_cl(x, box, ibox, head, body, ct, ncell)
    mc = np.max(ct) * d3
    ret = np.zeros((n, mc), dtype=np.int64)
    count = np.zeros((n,), dtype=np.int64)
    # or ret = (n, nl) for each particles
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
                    dx = x[k, 0] - x[i, 0]
                    dy = x[k, 1] - x[i, 1]
                    dz = x[k, 2] - x[i, 2]
                    dx = dx - box[0] * floor(dx / box[0] + 0.5)
                    dy = dy - box[1] * floor(dy / box[1] + 0.5)
                    dz = dz - box[2] * floor(dy / box[2] + 0.5)
                    dr = sqrt(dx * dx  + dy * dy + dz * dz)
                    if dr < rc:
                        ret[i, count[i]] = k
                        count[i] += 1
                    k = body[k]
            free(veci)
    free(j_vecs)
    return ret, count


def q(double[:,:] x, double[:] box, double rc, long[:,:] nl, long[:] nc):
    cdef long m, n, d, i, j, k, pj, pk, num_threads, l
    cdef np.ndarray[np.complex128_t, ndim=3] ret
    cdef np.ndarray[np.int64_t, ndim=1] count
    cdef double dr, dx, dy, dz, theta, phi
    cdef long[:] ls = np.zeros((2,), dtype=np.int64)
    ls[0] = 4
    ls[1] = 6
    n = x.shape[0]
    ret = np.zeros((n, 2, 13), dtype=np.complex128) # q4, q6
    count = np.zeros((n,), dtype=np.int64)
    num_threads = multiprocessing.cpu_count()
    with nogil, parallel(num_threads=num_threads):
        for i in prange(n, schedule='static'):
            for j in range(nc[i] - 1):
                pj = nl[i, j]
                for k in range(j + 1, nc[i]):
                    pk = nl[i, k]
                    dx = x[pk, 0] - x[pj, 0]
                    dy = x[pk, 1] - x[pj, 1]
                    dz = x[pk, 2] - x[pj, 2]
                    dx = dx - box[0] * floor(dx / box[0] + 0.5)
                    dy = dy - box[1] * floor(dy / box[1] + 0.5)
                    dz = dz - box[2] * floor(dy / box[2] + 0.5)
                    dr = sqrt(dx * dx + dy * dy + dz * dz)
                    if dr < rc:
                        count[i] += 1
                        theta = atan2(dy, dx) + pi
                        phi = acos(dz/dr)
                        for l in range(2):
                            for m in range(-ls[l], ls[l]+1):
                                ret[i, l, m+ls[l]] += sph_harm(m, ls[l], theta, phi)
    #np.sqrt(4 * np.pi * np.sum(np.abs(ret/ct[:,None,None]) ** 2, axis=-1)/ (2 * ls + 1).reshape(1, -1))
    return ret, count
