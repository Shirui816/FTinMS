from sys import argv
import pandas as pd
import numpy as np
from numba import jit

fn = argv[1]
box = np.array([50, 50, 50])
pos = pd.read_csv(fn, delim_whitespace=True, squeeze=1, header=None).values

Ndim = 500  # Finess of delta function
p, e = np.histogramdd(pos, bins=(Ndim, Ndim, Ndim),
                      range=((-box[0]/2, box[0]/2), (-box[1]/2, box[1]/2),
                      (-box[2]/2, box[2]/2)))
fp = np.fft.rfftn(p)
# rfft for efficiency.
# anaconda.accelerate package is strange,
# np.fft is actually faster than accelerate.mkl.fftpack.fft
# for N-dimensional ffts....
FP = fp*fp.conj()
RDF = np.fft.irfftn(FP, FP.shape).real
# IFFT{<rho(K)rho(-K)>}, 1/N\sum_i......(see numpy.fft, so N_bins is needed)
RDF[0, 0, 0] -= pos.shape[0]
# g(\bm{r}) = IFFT{<rho(K)rho(-K)>} - N\delta(\bm{r}) for gAA cases.
RDF = np.fft.fftshift(RDF)
# Move to box center for e is in -box/2 box/2
rbin = 0.2  # histogram for g(\bm{r}) to g(r)
rx = e[0][:Ndim] + 0.5*(e[0][-1]-e[0][-2])
ry = e[1][:Ndim] + 0.5*(e[1][-1]-e[1][-2])
rz = e[2][:Ndim] + 0.5*(e[2][-1]-e[2][-2])


@jit  # histogram g(R) to g(r)
def norm_r(RDF, rbin, rx, ry, rz):
    r"""Add up $gr(x,y,z) -> g(\sqrt{x^2+y^2+z^2})$."""
    rdf = np.zeros(int(box.max()/2*3**0.5/rbin)+1, dtype=np.float)
    cter = np.zeros(rdf.shape, dtype=np.float)
    for i in range(Ndim):
        for j in range(Ndim):
            for k in range(Ndim):
                rr = rx[i]**2+ry[j]**2+rz[k]**2
                r = int(rr**0.5/rbin)
                rdf[r] += RDF[i, j, k]
                cter[r] += 1
    return np.nan_to_num(rdf/cter)


rdf = norm_r(RDF, rbin, rx, ry, rz)
rdf /= pos.shape[0] * pos.shape[0]  # NA*NB for gAB(r)
rdf *= Ndim**3  # 1/N problem induced by FFT
rr = np.arange(rdf.shape[0])*rbin

o = open('rdf.txt', 'w')
for i, y in enumerate(rdf):
    o.write('%.8f %.8f\n' % ((i + 0.5) * rbin, y))
o.close()
