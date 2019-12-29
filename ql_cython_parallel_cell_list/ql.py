import numpy as np
import func


if __name__ == "__main__":
    from sys import argv

    _a, _b = np.loadtxt(argv[1], dtype=np.float), np.loadtxt(argv[2], dtype=np.float)
    _box = np.array([50, 50, 50], dtype=np.float)
    _rc = 2.0
    _ibox = np.asarray((_box / _rc), dtype=np.int64)
    #print(_ibox)
    _bs = 0.1
    _nb = int(_rc/_bs)
    #_head, _body = linked_cl(_b, _box, _ibox)
    #print('done')
    import time
    s = time.time()
    _ret = func.rdf(_a, _b, _box, _bs, _nb)
    print(round(time.time()-s,3))
    _rho = _b.shape[0] / np.multiply.reduce(_box)
    _r = (np.arange(_ret.shape[0] + 1) + 0.5) * _bs
    _dV = 4 / 3 * np.pi * np.diff(_r ** 3)
    #np.savetxt('heqd.txt', _head, fmt='%d')
    #np.savetxt('body.txt', _body, fmt='%d')
    #np.savetxt('rdf.txt', np.vstack([_r[:-1], _ret.sum(axis=0)]).T, fmt='%.6f')#.mean(axis=0)/_dV / _rho]).T, fmt='%.6f')
    np.savetxt('rdf_dv.txt', np.vstack([_r[:-1], _ret/_a.shape[0]/_dV / _rho]).T, fmt='%.6f')
