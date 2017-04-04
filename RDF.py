from sys import argv
import pandas as pd
import numpy as np
import time
import accelerate as acc
fn = argv[1]
print('Box origin must be at the center!')
pos = pd.read_csv(fn, delim_whitespace=True, squeeze=1, header=None).values

Ndim = 500 # Finess of delta function
V = box[0]*box[1]*box[2]
rho_bins = Ndim**3/V # Number density of bins
rho = pos.shape[0]/V
s = time.time()
p, e = np.histogramdd(pos, bins=(Ndim, Ndim, Ndim), range=((-box[0]/2, box[0]/2), (-box[1]/2, box[1]/2),(-box[2]/2, box[2]/2)))
print('Binning particles: %s' % (time.time()-s))
p = np.fft.fftshift(p) # POS is of center-origin, here move origin to cornor.
s = time.time()
fp = acc.mkl.fftpack.fftn(p) # Accelerate package
print('FFT time: %s' % (time.time()-s))
FP = fp*fp.conj()
s = time.time()
RDF = np.fft.ifftn(FP).real # IFFT{<rho(K)rho(-K)>}, 1/N\sum_i......(see numpy.fft, so rho_bins is needed)
print('IFFT time: %s' % (time.time()-s))
RDF[0,0,0] -= pos.shape[0]
RDF = np.fft.fftshift(RDF)
rbin = 0.2 # histogram for g(\bm{r}) to g(r)
rx = e[0][:Ndim] + 0.5*(e[0][-1]-e[0][-2])
ry = e[1][:Ndim] + 0.5*(e[1][-1]-e[1][-2])
rz = e[2][:Ndim] + 0.5*(e[2][-1]-e[2][-2])

from numba import jit
@jit # histogram g(R) to g(r)
def norm_r(RDF, rbin, rx, ry, rz):
    rdf = np.zeros(int(box.max()/2*3**0.5/rbin)+1, dtype=np.float)
    cter = np.zeros(rdf.shape, dtype=np.float)
    for i in range(Ndim):
        for j in range(Ndim):
            for k in range(Ndim):
                rr = rx[i]**2+ry[j]**2+rz[k]**2
                r = int(rr**0.5/rbin)
                rdf[r] += RDF[i,j,k]
                cter[r] += 1
    return np.nan_to_num(rdf/cter)


rdf = norm_r(RDF, rbin, rx,ry,rz)
rdf /= pos.shape[0] * rho # NA*NB/V for gAB(r)
rdf *= rho_bins # NORMED BY BIN DENSITY
rr = np.arange(rdf.shape[0])*rbin

o = open('rdf.txt', 'w')
for i, y in enumerate(rdf):
    o.write('%.8f %.8f\n' % ((i+0.5) * rbin, y))
o.close()
