import numpy as np
import pandas as pd
from sys import argv

# Autocorrelation function calculation for polymer normal modes with reading
# trajectory files by chunk. The trajectory files should consist of normal
# modes in rows and in chain sequence -> (n_chains * n_modes, n_dim)

n_chains, n_modes, all_chains = 400, 250, 832  # buffer counted as n_chains
# The get_chunk of pd.read_csv: in this case, it will read 3 times with shapes
# (400 * 250, 3), (400 * 250, 3) and (32 * 250, 3)
allFiles = [pd.read_csv(_, squeeze=1, header=None, delim_whitespace=1,
                        chunksize=n_chains*n_modes, comment='#')
            for _ in argv[1:]]
n_frames = len(allFiles)
bufferArr = np.zeros((n_frames, n_chains, n_modes, 3))
autoCorr = np.zeros((n_frames, n_modes))

# Unless axis = 0 or -1, MKL_FFT will not accelerate.
for _ in range(0, all_chains, n_chains):
    allX = np.asarray([_.get_chunk().values.reshape(-1, n_modes, 3)
                       for _ in allFiles])
    # n_frames, n_chains, n_modes, n_dim, for sequential data
    FX = np.fft.rfft(allX, axis=0, n=2*n_frames)
    # n_frames_ft, n_chains, n_modes, n_dim
    # summing up along n_dim then perform ift, then suming over n_chains
    autoCorr = autoCorr +\
        np.fft.irfft(np.sum(abs(FX)**2, axis=3), axis=0).sum(axis=1)[:n_frames]
    # n_frames, n_modes
autoCorr = autoCorr / all_chains / np.arange(n_frames, 0, -1)[:, np.newaxis]
autoCorr = autoCorr / autoCorr[0]

np.savetxt('autocorr.txt',
           np.hstack([np.arange(autoCorr.shape[0])[:, np.newaxis], autoCorr]),
           fmt='%.6f')
