a = np.random.random((10,1, 11, 1, 3))  # traj (nframes, nparticles, ndims) -> (nsegments, 1, segmentlength, nparticles, ndims)

a = np.append(a[:,:,:6], a[:,:,5:], axis=1)  # (nsegments, 2, [t-tr/2,t];[t+tr/2], nparticles, ndims)

c = (a-np.flip(np.expand_dims(a.mean(axis=2),2),axis=1))  # (nseg, 2, (r^A-mean(r^B));(r^B-mean(r^A)), np, nd)

ret = np.multiply.reduce(np.mean(c **2, axis=2).sum(axis=-1),axis=1)  # (nseg, (r^A-mean(r^B))^2\times(r^B-mean(r^A))^2, np, nd)

pp = np.mean(a**2, axis=2)

p = np.mean(a, axis=2)

pab = np.einsum('...j,...j->...', p[:,0],p[:,1])

ret = (pp.sum(axis=-1)[:,0]-2*pab+(p[:,1]**2).sum(axis=-1))*(pp.sum(axis=-1)[:,1]-2*pab+(p[:,0]**2).sum(axis=-1))

########################################################################################################################################
# ret &:= \langle(r_i^A-\langle r_i^B\rangle_B)^2\rangle_A \langle(r_i^B-\langle r_i^A \rangle_A)^2\rangle_B \\
#     & = (\langle |r_i^A|^2 \rangle_A - 2\langle r_i^A \rangle_A\cdot\langle r_i^B \rangle_B + (\langle r_i^B \rangle_B)^2)\times \\
#     & (\langle |r_i^B|^2 \rangle_B - 2\langle r_i^A \rangle_A\cdot\langle r_i^B \rangle_B + (\langle r_i^A \rangle_A)^2)
########################################################################################################################################

# This loop can be accelerated by parallel python, etc., divide traj.
for frame in range(5, nf-5):
    seg = traj[frame-5:frame+6]  # total 11 frames (11, np, 3)
    seg = np.expand_ndims(seg, axis=0)  # (1, 11, np, 3)
    seg = np.vstack([seg[:,:6], seg[:,5:]])  # (2, 6, np, 3) with dup frame_5
    c = (seg-np.flip(np.expand_dims(seg.mean(axis=1),1),axis=0))  # (2, 6, np, 3)
    ret = np.multiply.reduce(np.mean(c **2, axis=1).sum(axis=-1),axis=0) ** 0.5  # (2, 6, np, 3) -> (np,)
    #....deal with ret
    inds = np.flatnonzero(ret > 0.2)  # some threshold
    rdf_frame = traj[frame]
    # calculate rdfs of particles in inds ...

# Example on parallel
from multiprocessing import Pool

# read entire traj as np array

def f(frame):
    seg = traj[frame-5:frame+6]  # total 11 frames (11, np, 3)
    seg = np.expand_ndims(seg, axis=0)  # (1, 11, np, 3)
    seg = np.vstack([seg[:,:6], seg[:,5:]])  # (2, 6, np, 3) with dup frame_5
    c = (seg-np.flip(np.expand_dims(seg.mean(axis=1),1),axis=0))  # (2, 6, np, 3)
    ret = np.multiply.reduce(np.mean(c **2, axis=1).sum(axis=-1),axis=0) ** 0.5  # (2, 6, np, 3) -> (np,)
    #....deal with ret
    inds = np.flatnonzero(ret > 0.2)  # some threshold
    rdf_frame = traj[frame]
    #...
    return feature_vec

if __name__ == '__main__':
    with Pool(48) as p:
        ret = p.map(f, range(5, nf-5))
