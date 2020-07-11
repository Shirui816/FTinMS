import numpy as np
from scipy.signal import correlate2d

rho = np.random.random((4,4)) + 1j * np.random.random((4,4))
res0 = correlate2d(rho, rho, 'full', 'fill', 0)  # unwrapped
res1 = np.fft.fftshift(np.fft.ifftn(np.abs(np.fft.fftn(rho, s=2*np.array(rho.shape)-1))**2))
np.allclose(res0, np.flip(res1, axis=(0,1)).conj())

res0 = correlate2d(rho, rho, 'same', 'wrap')  # wrapped
res1 = np.fft.fftshift(np.fft.ifftn(np.abs(np.fft.fftn(rho))**2))
np.allclose(res0, np.flip(res1, axis=(0,1)).conj())

# real values
rho = np.random.random((4,4))
res0 = correlate2d(rho, rho, 'full', 'fill', 0)  # unwrapped
res1 = np.fft.fftshift(np.fft.irfftn(np.abs(np.fft.rfftn(rho, s=2*np.array(rho.shape)-1))**2, s=2*np.array(rho.shape)-1))
np.allclose(res0, np.flip(res1, axis=(0,1)).conj())

res0 = correlate2d(rho, rho, 'same', 'wrap')  # wrapped
res1 = np.fft.fftshift(np.fft.irfftn(np.abs(np.fft.rfftn(rho))**2, s=rho.shape))
np.allclose(res0, np.flip(res1, axis=(0,1)).conj())
