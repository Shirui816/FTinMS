from math import floor

import numba as nb
import numpy as np
from numba import cuda
from scipy.spatial import cKDTree

np.set_printoptions(precision=3)


@cuda.jit(device=True)
def cu_mat_dot_v_pbc(a, b, c, box, ret):
    for i in range(a.shape[0]):
        tmp = 0
        for j in range(a.shape[1]):
            dc = b[j] - c[j]
            dc = dc - box[j] * floor(dc / box[j] + 0.5)
            tmp += a[i, j] * dc
        ret[i] = tmp


@cuda.jit(device=True)
def cu_mat_dot_v(a, b, ret):
    for i in range(a.shape[0]):
        tmp = 0
        for j in range(a.shape[1]):
            tmp += a[i, j] * b[j]
        ret[i] = tmp


class LocalizedNonAffineDisp(object):
    def __init__(self, n_dim=3, gpu=0):
        self.n_dim = n_dim
        self.gpu = gpu
        self._cu_XY = None
        self._cu_DIV = None
        print(f"Generating {self.n_dim}D calculators.", end="", flush=True)
        with cuda.gpus[self.gpu]:
            self._build_calc()

    def set_dim(self, n_dim):
        self.n_dim = n_dim
        self._cu_XY = None
        self._cu_DIV = None
        print(f"Re-generating {self.n_dim}D calculators.", end="", flush=True)
        with cuda.gpus[self.gpu]:
            self._build_calc()

    def _build_calc(self):
        n_dim = self.n_dim

        @cuda.jit
        def _cu_XY(_a, _b, _box, _nl, _nc, _Xij, _Yij, _l0, _lt):
            i = cuda.grid(1)
            if i >= _a.shape[0]:
                return
            _dr0 = cuda.local.array(n_dim, nb.float64)
            _drt = cuda.local.array(n_dim, nb.float64)
            if _nc[i] < n_dim + 1:
                # n_nei (without self) < n_dim is filtered out.
                # there may exist collinear vectors...
                for k in range(n_dim):
                    _Xij[i, k, k] = 1
                    _Yij[i, k, k] = 1

            for j in range(_nc[i]):
                pj = _nl[i, j]
                if pj == i:
                    continue
                cu_mat_dot_v_pbc(_l0, _a[pj], _a[i], _box, _dr0)
                cu_mat_dot_v_pbc(_lt, _b[pj], _b[i], _box, _drt)
                for k in range(n_dim):
                    for l in range(n_dim):
                        _Xij[i, k, l] += _drt[k] * _dr0[l]
                        _Yij[i, k, l] += _dr0[k] * _dr0[l]

        print(".", end="", flush=True)

        @cuda.jit
        def _cu_DIV(_a, _b, _box, _nl, _nc, _XIY, _l0, _lt, _ret):
            i = cuda.grid(1)
            if i >= _a.shape[0]:
                return
            _dr0 = cuda.local.array(n_dim, nb.float64)
            _drt = cuda.local.array(n_dim, nb.float64)
            _dr = cuda.local.array(n_dim, nb.float64)
            for j in range(_nc[i]):
                pj = _nl[i, j]
                if pj == i:
                    continue
                cu_mat_dot_v_pbc(_l0, _a[pj], _a[i], _box, _dr0)
                cu_mat_dot_v_pbc(_lt, _b[pj], _b[i], _box, _drt)
                cu_mat_dot_v(_XIY[i], _dr0, _dr)
                for k in range(n_dim):
                    _ret[i] += (_drt[k] - _dr[k]) ** 2
            if _nc[i] > 1:
                _ret[i] = _ret[i] / (_nc[i])
            if _nc[i] < n_dim + 1:
                _ret[i] = -1

        print("[Done]")
        self._cu_XY = _cu_XY
        self._cu_DIV = _cu_DIV

    def calculate(self, x_ref, x_def, box_ref, box_def, rc=3.0, k=None):
        r"""Calculating localized non-affine displacement and strain tensor.
        :param x_ref: np.ndarray[n_pos, n_dim], X of reference
        :param x_def: np.ndarray[n_pos, n_dim], X of deformed frame
        :param box_ref: np.ndarray[n_dim, n_dim], box tensor of the reference
        :param box_def: np.ndarray[n_dim, n_dim], box tensor of the deformed frame
        :param rc: double, cut-off to construct neighbor list
        :param k: int, n-nearest neighbor, if k is set, rc will be ignored.
        :return: (np.ndarray[n_pos, n_dim, n_dim], np.ndarray[n_pos]),
        non-affine strain tensor and displacement of each particle
        EXAMPLE:
        >>> x0 = np.random.random((30, 3)) * 10 # positions for reference
        >>> s0 = np.array([[1, 0.2, 0], [0, 1, 0], [0., 0, 1]]) # strain tensor
        >>> x1 = x_ref.dot(s0.T) # deformed positions, in column vector regime
        >>> box0 = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10.]]) # box tensor for reference
        >>> box1 = box_ref.dot(s0.T) # box tensor for deformed
        >>> strain = (np.linalg.inv(box_ref).dot(box_def)).T # strain of ref->deformed
        >>> print(strain, s0) # should be same
        >>> f = LocalizedNonAffineDisp(3) # calculator
        >>> print(f.calculate(x_ref, x_def, box_ref, box_def, rc=3.0))
        # strains are mostly zeros for the example is constructed with affine deformation
        # for particles with n_nei (without self) < n_dim, the displacement is -1.
        >>> print(f.calculate(x_ref, x_def, box_ref, box_def, rc=3.0, k=4))
        # strains are mostly zeros for the example is constructed with affine deformation
        # for particles with n_nei < n_dim, the displacement is -1.
        """
        n_dim = x_ref.shape[1]
        if n_dim != self.n_dim:
            self.set_dim(n_dim)
        ob0 = np.diag(np.diag(box_ref))  # as the reference orthogonal-box
        box = np.zeros(n_dim)
        for d in range(n_dim):
            box[d] = ob0[d, d]
        os0 = (np.linalg.inv(ob0).dot(box_ref)).T
        os1 = (np.linalg.inv(ob0).dot(box_def)).T
        strain0to1 = (np.linalg.inv(box_ref).dot(box_def)).T
        ox0 = x_ref.dot(np.linalg.inv(os0).T)  # col vec regime
        ox1 = x_def.dot(np.linalg.inv(os1).T)
        tree0 = cKDTree(np.mod(ox0, ob0.diagonal()), boxsize=box)
        print(f"Reference orthogonal box is: {np.diag(ob0)}")
        # print(f"Strains of X0 and X1 are\n{os0}\nand\n{os1}")
        print(f"Strain is\n{strain0to1}.")
        print(f"Constructing neighbor list...", end="", flush=True)
        if k is None:
            _nn0 = tree0.query_ball_point(np.mod(ox0, box), r=rc)
            nn = np.zeros((ox0.shape[0], max(map(len, _nn0))), dtype=np.int64) - 1
            nc = np.zeros(ox0.shape[0])
            for i, item in enumerate(_nn0):
                nn[i, :len(item)] = item
                nc[i] = len(item)
        else:
            _, _nn0 = tree0.query(np.mod(ox0, ob0.diagonal()), k=k)
            nn = np.zeros((ox0.shape[0], k), dtype=np.int64) - 1
            nc = np.zeros(ox0.shape[0]) + k
            for i, item in enumerate(_nn0):
                nn[i] = item
        print(f"[Done]")
        print(f"Start calculating...", end="", flush=True)
        x_ij = np.zeros((x_ref.shape[0], n_dim, n_dim), dtype=np.float64)
        y_ij = np.zeros((x_ref.shape[0], n_dim, n_dim), dtype=np.float64)
        div2 = np.zeros((x_ref.shape[0],), dtype=np.float64)
        with cuda.gpus[self.gpu]:
            device = cuda.get_current_device()
            tpb = device.WARP_SIZE
            bpg = int(x_ref.shape[0] / tpb) + 1
            d_ox0 = cuda.to_device(ox0)
            d_ox1 = cuda.to_device(ox1)
            d_ob0 = cuda.to_device(box)
            d_nn0 = cuda.to_device(nn)
            d_nc0 = cuda.to_device(nc)
            d_Xij = cuda.to_device(x_ij)
            d_Yij = cuda.to_device(y_ij)
            d_DIV = cuda.to_device(div2)
            d_os0 = cuda.to_device(os0)
            d_os1 = cuda.to_device(os1)
            self._cu_XY[bpg, tpb](
                d_ox0, d_ox1, d_ob0, d_nn0, d_nc0, d_Xij, d_Yij, d_os0, d_os1
            )
            d_Xij.copy_to_host(x_ij)
            d_Yij.copy_to_host(y_ij)
            cuda.synchronize()
            xi_y = np.matmul(x_ij, np.linalg.pinv(y_ij, hermitian=True))
            # no appropriate cuda method to do this currently
            d_XIY = cuda.to_device(xi_y)
            self._cu_DIV[bpg, tpb](
                d_ox0, d_ox1, d_ob0, d_nn0, d_nc0, d_XIY, d_os0, d_os1, d_DIV
            )
            d_DIV.copy_to_host(div2)
            cuda.synchronize()

        print(f"[Done]")
        return xi_y - strain0to1, div2


if __name__ == '__main__':
    x0 = np.random.random((30, 3)) * 10
    s0 = np.array([[1, 0.2, 0], [0, 1, 0], [0., 0, 1]])
    x1 = x0.dot(s0.T)
    box0 = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10.]])
    box1 = box0.dot(s0.T)
    strain = (np.linalg.inv(box0).dot(box1)).T
    f = LocalizedNonAffineDisp(3)
    print(f.calculate(x0, x1, box0, box1, rc=3.0))
