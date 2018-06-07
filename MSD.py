import sys
import numpy as np
from simpletraj.dcd.dcd import DCDReader

# MSD calculation using FFT algorithm.
# $g(m):=\frac{1}{N-m}\sum_{k=0}^{N-m-1}(r(k+m)-r(k))^2$
# $g(m)=\frac{1}{N-m}\sum_{k=0}^{N-m-1}(r(k+m)^2+r(k)^2-2r(k+m)r(k))
# For $\frac{1}{N-m}\sum_{k=0}^{N-m-1}(r(k+m)r(k))=C_r(m)$ and $C_r(t)$ is
# the autocorrelation function of $r(t)$, by Convolution theorem, the auto-
# -correlation part can be calculated by FFT algorithm.
# For the Square part $SQ(m):=\sum_{k=0}^{N-m-1}(r(k+m)^2+r(k)^2)$, one can
# calculate for every $m$ via a recursive relation:
# $SQ(m-1)-SQ(m)=r(m-1)^2+r(N-m)^2$
# and for $m=0$, the square part is
# $2\sum_{k=0}^{N-1}r(k)^2$


def msd_Correlation(allX):
    """Autocorrelation part of MSD."""
    M = allX.shape[0]
    # numpy with MKL (i.e. intelpython distribution), the fft wont be
    # accelerated unless axis along 0 or -1
    # perform FT along n_frame axis
    # (n_frams, n_particles, n_dim) -> (n_frames_Ft, n_particles, n_dim)
    allFX = np.fft.rfft(allX, axis=0, n=M*2)
    # sum over n_dim axis
    corr = np.sum(abs(allFX)**2, axis=-1)  # (n_frames_ft, n_particles)
    # IFT over n_frame_ft axis (axis=0), whole operation euqals to
    # fx = fft(_.T[0]), fy =... for _ in
    # allX.swapaxes(0,1) -> (n_particles, n_frames, n_dim)
    # then sum fx, fy, fz...fndim
    # rfft for real inpus, higher eff
    return np.sum(np.fft.irfft(corr, axis=0)[:M].real,
                  axis=1)/np.arange(M, 0, -1)  # (n_frames, n_particles)


def msd_Square(allX):
    """Square part of MSD."""
    M = allX.shape[0]  # n_frame, n_particle, n_dim
    S = np.square(allX).sum(axis=1).sum(axis=-1)
    S = np.append(S, 0)  # for SS[-1] == SS[M] == 0
    SS = 2 * S.sum()
    SQ = np.zeros(M)
    for m in range(M):
        SS = SS - SS[m - 1] - SS[M - m]
        SQ[m] = SS / (M - m)
    return SQ


dcd = DCDReader('particle.dcd')
if dcd.periodic:
    print("Error, Periodic data found!")
    sys.exit(1)
allX = np.asarray([np.copy(_) for _ in dcd])
# Or read by chunk, cumulate every chunk then average by particles
msd = (msd_Square(allX) - 2 * msd_Correlation(allX))/allX.shape[1]
np.savetxt('msd.txt', np.vstack([np.arange(msd.shape[0]), msd]).T, fmt='%.6f')
