from func import rdf, linked_cl
import numpy as np

if __name__ == "__main__":
    import sys
    _a, _b = np.loadtxt(sys.argv[1], dtype=np.float), np.loadtxt(sys.argv[2], dtype=np.float)
    _box = np.array([50, 50, 50], dtype=np.float)
    _rc = 3.0
    _ibox = np.asarray(np.round(_box / _rc), dtype=np.int64)
    _bs = 0.1
    _nb = int(_rc / _bs)
    _head, _body = linked_cl(_b, _box, _ibox)
    print('done')
    _ret = rdf(_a, _b, _box, _head, _body, _ibox, _bs, _nb)
    _rho = _b.shape[0] / np.multiply.reduce(_box)
    _r = (np.arange(_ret.shape[1] + 1) + 0.5) * _bs
    _dV = 4 / 3 * np.pi * np.diff(_r ** 3)
    np.savetxt('rdf.txt', np.vstack([_r[:-1], _ret.mean(axis=0) / _dV / _rho]).T, fmt='%.6f')
