from numba import guvectorize
from numba import float64
import numpy as np
import numba as nb


@guvectorize([(float64[:, :], float64[:, :], float64[:, :])], '(n,p),(p,m)->(n,m)',
             target='parallel')  # target='cpu','gpu'
def batch_dot(a, b, ret):  # much more faster than np.tensordot or np.einsum
    r"""Vectorized universal function.
    :param a: np.ndarray, (...,N,P)
    :param b: np.ndarray, (...,P,M)
    axes will be assigned automatically to last 2 axes due to the signatures.
    :param ret: np.ndarray, results. (...,N,M)
    :return: np.ndarray ret.
    """
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            tmp = 0.
            for k in range(a.shape[1]):
                tmp += a[i, k] * b[k, j]
            ret[i, j] = tmp


@nb.jit(nopython=True, nogil=True)
def pbc(r, d):
    return r - d * np.rint(r / d)


def batchRgTensor(samples, boxes):
    # samples: (..., n_chains, n_monomers, n_dimension)
    # boxes: (..., n_dimension)
    # samples can be (n_batch, n_frame, n_chains, n_monomers, n_dimension) for example
    # or just simply be (n_chains, n_monomers, n_dimension)
    # if there was no batch or frame data, or box is constant of all the simulations
    # boxes = (n_dimension,) would be fine.
    # if boxes were given as (n_batch, n_frame, n_dimension), it must be expand to
    # (n_batch, n_frame, 1, 1, n_dimension) so that for all chains in same frame of
    # one batch of datas, the box is same.
    # same if the samples were given as (n_frames, n_chains, n_monomers, n_dimension)
    # boxes is (n_frames, n_dimension) and must be expand to (n_frames, 1, 1, n_dimension)
    # so that for all chains in same frame has same box lengths.
    if samples.ndim <= 3:  # (n_chains, n_monomers, n_dimensions), samples.ndim>3 means at least there was frame info
        raise ValueError("NO~~~Are you using multiple box values for 1 frame data?")
    else:
        boxes = np.expand_dims(np.expand_dims(boxes, -2), -3)
    # boxes' dimension is always lower than samples by 2 (n_chains, n_monomers)
    # e.g., (10, 3) for 10 frames (10, n_chains, n_monomers, 3) for sample
    chain_length = samples.shape[-2]
    samples = pbc(np.diff(samples, axis=-2, prepend=samples[..., :1, :]), boxes).cumsum(axis=-2)
    # samples -> (..., n_chains, n_monomers, n_dim)
    com = samples.sum(axis=-2, keepdims=True) / chain_length  # com -> (..., n_chains, 1, n_dim)
    samples = samples - com
    rgTensors = batch_dot(np.swapaxes(samples, -2, -1), samples) / chain_length
    # batch_dot == np.einsum('...mp,...pn->...mn', np.swapaxes(samples, -2, -1), samples) -> (..., n_chains, n_dim, n_dim)
    # batch_dot == np.einsum('...mp,...pn->...mn', (..., n_chains, n_dim, n_monomers),  (..., n_chains, n_monomers, n_dim))
    # batch_dot is way more faster than np.einsum 
    return np.linalg.eigh(rgTensors)  # work on last (..., M, M) matrices
