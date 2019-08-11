# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange, parallel
cimport openmp
from libc.math cimport floor,sqrt,pow
from libc.stdlib cimport malloc, free
import multiprocessing


cdef double unravel_index_f_r(long i, long[:] dim) nogil:
    cdef long k, d, tmp
    cdef double r
    d = dim.shape[0]
    for k in range(d):
        tmp = i % dim[k]
        r += <double> (tmp)**2
        i = (i - tmp) / dim[k]
    return sqrt(r)


def hist_to_r(double[:] x, long[:] shape, double dr, double bs, double rc):
    r"""\int_{V} f(\bm{r}) \delta(|\bm{r}|-r) \mathrm{d}\bm{r}, using Fortran-raveled arrays.

    :param x: raveled nd-array in Fortran order using `x.ravel('F')
    :param shape: original shape of x
    :param dr: dx
    :param bs: bin size of histogram
    :param rc: r cut
    :return: tuple(summed array, count)
    """
    cdef long i, n, j, n_bins
    cdef np.ndarray[np.double_t, ndim=2] ret, count
    cdef int num_threads, thread_num
    n_bins = <long> (rc / bs)
    num_threads = multiprocessing.cpu_count()
    ret = np.zeros((num_threads, n_bins), dtype=np.float64)
    count = np.zeros((num_threads, n_bins), dtype=np.float64)
    n = x.shape[0]
    with nogil, parallel(num_threads=num_threads):
        for i in prange(n, schedule='dynamic'):
            j = <long> (unravel_index_f_r(i, shape)*dr/bs)
            thread_num = openmp.omp_get_thread_num()
            if j < n_bins:
                ret[thread_num, j] += x[i]
                count[thread_num, j] += 1
    return np.sum(ret,axis=0), np.sum(count,axis=0)
