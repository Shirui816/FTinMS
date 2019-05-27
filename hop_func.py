a = np.random.random((10,1, 11, 1, 3))

a = np.append(a[:,:,:6], a[:,:,5:], axis=1)

c = (a-np.flip(np.expand_dims(a.mean(axis=2),2),axis=1))

ret = np.multiply.reduce(np.mean(c **2, axis=2).sum(axis=-1),axis=1)

pp = np.mean(a**2, axis=2)

p = np.mean(a, axis=2)

pab = np.einsum('...j,...j->...', p[:,0],p[:,1])

ret = (pp.sum(axis=-1)[:,0]-2*pab+(p[:,1]**2).sum(axis=-1))*(pp.sum(axis=-1)[:,1]-2*pab+(p[:,0]**2).sum(axis=-1))

########################################################################################################################################
# ret &:= \langle(r_i^A-\langle r_i^B\rangle_B)^2\rangle_A \langle(r_i^B-\langle r_i^A \rangle_A)^2\rangle_B \\
#     & = (\langle |r_i^A|^2 \rangle_A - 2\langle r_i^A \rangle_A\cdot\langle r_i^B \rangle_B + (\langle r_i^B \rangle_B)^2)\times \\
#     & (\langle |r_i^B|^2 \rangle_B - 2\langle r_i^A \rangle_A\cdot\langle r_i^B \rangle_B + (\langle r_i^A \rangle_A)^2)
########################################################################################################################################
