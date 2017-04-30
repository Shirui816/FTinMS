import numpy as np
from numpy.fft import fftn, fftshift
import scipy
from pandas import read_csv
from sys import argv
import time
import accelerate.cuda as acc
cufft = acc.fft.fft
gpu = 1

zero_padding = 1 # Zero-padding
_lambda = 1.5 # In Angstrom.
kmax = 100 * zero_padding
N = 100 * zero_padding
Ndiv = 2 * N + 1 # fineness of a delta function, usually 0.1 is fine: box * zero_padding / Ndiv ~ 0.1
origin = 'corner' # 'corner' by default or 'center'

'''
This `center or corner` option is not necessary for we just care about FT{rho}FT{rho}.conj() which is
abs(FT{rho})**2, but FT{f(x-a)} = exp(-2pi*i*k*a)FT{f(x)}, therefor the abs(FT{f}) is under translation
invariance. Moving particles to make box center be origin is just for easy-debugging.
'''

box = np.array([0, 0, 0]) # Angstrom. 0 then box is taken as the r_max - r_min of sample coordinates.
mode = 'STRF' # 'XRD' for xrd 'STRF' for structure factor

print("-----------------------------------------------")
print("Origin is at %s" % (origin))
print("lambda is %.4f" % (_lambda))
if box.dot(box) == 0:
    print("Box is not set, using r_max - r_min of samples.")
print("%s mode is in using, if XRD mode, DATA should be 4 columns as 'atomic_number x y z'" % (mode))
print("-----------------------------------------------")
# create arrays of positions, get the box dimenstions
X = read_csv(argv[1], squeeze=1, header=None, delim_whitespace=1).values # Angstrom
Natoms = X.shape[0]
if mode == 'XRD':
    if not X.shape[1] == 4:
        W = np.ones(X.shape[0])
        POS = X
        print("Warning, weights are not specified, using 1 for default.")
    else:
        W = X[:,0]
        POS = X[:,1:]
else:
    if X.shape[1] == 4:
        POS = X[:,1:]
    else:
        POS = X
    W = np.ones(X.shape[0])

if box.dot(box) == 0:
    xM, yM, zM = max(POS[:,0]), max(POS[:,1]), max(POS[:,2])
    xm, ym, zm = min(POS[:,0]), min(POS[:,1]), min(POS[:,2])
    box = np.array([xM-xm, yM-ym, zM-zm])
if origin == 'corner':
    POS -= box/2
box *= zero_padding
Lx, Ly, Lz = box[0], box[1], box[2]
rho_bins = Ndiv ** 3 / Lx / Ly / Lz
# binning and histogramming
s = time.time()
f, edges = np.histogramdd(POS, bins=(Ndiv, Ndiv, Ndiv), weights=W, range=((-Lx/2, Lx/2), (-Ly/2, Ly/2), (-Lz/2,Lz/2)))# $Z(\bm{r})\rho(\bm{r})$
F = fftshift(f).astype(np.complex64) # For cufft_C2C, for systems that 0 at box center, this shift makes 0 at corner. But is there any difference between shift/noshift arrays?
print("Binning partiles", time.time()-s)
s = time.time()
#ftk = fftshift(fftn(fftshift(f))) # steric integration of exp(r)DiracDelta(r) form, shift zero-feq value to center. CPU VER
cuftk = np.empty(F.shape).astype(np.complex64)
with cuda.gpus[gpu]:
    cufft(F, cuftk)
ftk = fftshift(cuftk)
sk = np.abs(ftk)**2 / float(Natoms) # $\frac{1}{N}\rho(\bm{k})\rho(-\bm{k})$
print("FFT and S(k)", time.time()-s)
kbin = 2 * np.pi * np.sqrt((1/Lx ** 2 + 1/Ly ** 2 + 1/Lz ** 2)/3) # binsize of $\bm{k}$
normk = 2 * np.pi * np.array([1/Lx, 1/Ly, 1/Lz]) # Norm vector in rep-space

from numba import cuda
from math import floor, sqrt

@cuda.jit('void(int64, float32[:,:,:], int64, float32, float32[:], float32[:], float32[:], uint32[:])')
def norm_sq_cu(Ndiv, sk, kmax, kbin, normk, C, D, E):
    i = cuda.grid(1)
    if -kmax<=i-N<=kmax:
        for j in range(Ndiv):
            if -kmax<=j-N<=kmax:
                for k in range(Ndiv):
                    if -kmax<=k-N<=kmax:
                        Qx = (i-N) * normk[0] # 0-freq intensity is at center of array due to fftshift
                        Qy = (j-N) * normk[1]
                        Qz = (k-N) * normk[2]
                        q = floor(sqrt(Qx * Qx + Qy * Qy + Qz * Qz)/kbin)
                        if q!=0:
                            cuda.atomic.add(C, q, sk[i, j, k])
                            cuda.atomic.add(D, q, q)
                            cuda.atomic.add(E, q, 1)

def call_norm_cu(N, sk, kmax, kbin, normk, gpu=1):
    Ndiv = 2 * N+1
    Nbins = int(kmax * sqrt(3))+1
    C = np.zeros((Nbins,)).astype(np.float32)
    D = np.zeros((Nbins,)).astype(np.float32)
    E = np.zeros((Nbins,)).astype(np.uint32)
    SK = sk.astype(np.float32)
    NORMK = normk.astype(np.float32)
    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = int(np.ceil(float(Ndiv)/tpb))
        norm_sq_cu[bpg, tpb](Ndiv, SK, kmax, kbin, NORMK, C, D, E)
    E[E==0]=1
    return C/E, D/E*kbin, E # C is SQ, D is Q and E is the counter
    

s = time.time()
C, D, E = call_norm_cu(N, sk, kmax, kbin, normk, gpu=gpu)
print("Normalize to S(q)", time.time()-s)
o = open('%s.txt' % (mode), 'w')
if mode == 'XRD':
    o.write("#2theta Intensity q q_count\n")
    xaxis = np.arcsin(D*_lambda/4/np.pi)/np.pi*360
else:
    o.write("#Q S(Q) Q_count\n")
    xaxis = D
for i,j,k in zip(xaxis, C, E):
    o.write('%.4f %.4f %s\n' % (i, j, k))
o.close()
