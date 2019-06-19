import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import curve_fit
import scipy.linalg as sla
from matplotlib import pyplot as plt
from numba import jit
from sys import argv
from tqdm import tqdm
from numba import guvectorize, float64, jit


def rouse_mat(n): 
	ret = np.array([[-1,1] + [0] * (n-2)])
	for i in range(1, n-1):
		ret = np.append(ret, np.array([[0] * (i-1) + [1,-2,1] + [0] * (n-2-i)]), axis=0)
	return -np.append(ret, np.array([[0] * (n-2) + [1,-1]]), axis=0)


def zeta_mat(n, alpha, delta):
	return sla.expm(-delta * toeplitz(np.exp(-alpha * np.arange(n))))


def Roessler2010_SRK2_rouse(A, B, y0, t, dW=None):
	'''Simulate EQU as dX/dt = AX + B dW.
	For ROUSE systems: 
	dr_i = 1/z_i * -((k_{i-1}(r_{i-1}-r_i)+k_i(r_{i+1}-r_i)) dt + 1/z_i \sqrt{2k_BTz_i} dW
	coefficients in the LHS must be 1, k_i and mobility z_i can be modified.
	and k_i and z_i must be constants.
	:param A: matrix in RHS of eqs
	:param B: fluctuations in RHS of eqs
	:param y0: initial positions
	:param t: time
	'''
	A2 = A.dot(A)
	dt = t[1] - t[0]
	if dW is None:
		dW = np.random.normal(0, dt**0.5, (t.shape[0]-1, *y0.shape))
	y = np.zeros((t.shape[0], *y0.shape))
	y[0] = y0
	for i in range(t.shape[0]-1):
		yi = y[i]
		y[i+1] = yi + A.dot(yi) * dt + 0.5 * A2.dot(yi) * dt ** 2 + dW[i] * B
	return y


ndim = 3
ns = np.asarray(argv[1:], dtype=np.int)
T = 100
nT = int(T/0.02)
t=np.linspace(0,T,nT,endpoint=False)


for n in ns:
	ret = np.zeros((nT, ndim))
	msd = np.zeros((nT,))
	R = rouse_mat(n)
	for i in tqdm(range(1000), ascii=True, desc='Chian length of %d' % (n)):
		r = Roessler2010_SRK2_rouse(-3*R, np.ones((n,1))*np.sqrt(2), np.zeros((n,ndim)), t).mean(axis=1)
		ret += r
		msd += np.sum(r ** 2, axis=-1)
	np.savetxt('traj_cm_%d.txt' % (n), np.vstack([t,ret.T/1000]).T)
	np.savetxt('msd_cm_%d.txt' % (n), np.vstack([t,msd/1000]).T)
