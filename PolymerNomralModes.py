import numpy as np
from hoomd_xml_pd import hoomd_xml
from sys import argv

# An example of Polymer Normal Mode calculation, by defination of
# $X_p:=\frac{1}{N}\sum_{i=1}^N \cos{(\frac{p\pi}{N}(i-\frac{1}{2}))}$
# Using numpy.tensordot for efficiency. If the defination of $X_p$ is
# $\frac{1}{N}\sum_{i=1}^N \cos{(\frac{pi\pi}{N})}$
# np.fft.rfft(r -> (n_position, n_dim), axis=0, n=2*n_position) is more
# effective for calculation of ALL modes.


chain_length = 250
factors = 1/chain_length *\
          np.asarray([np.cos(p*np.pi/chain_length *
                             (np.arange(1, chain_length+1)-1/2))
                      for p in range(1, chain_length+1)])
# factors for mode 1 to mode N

for f in argv[1:]:
    print(f)
    xml = hoomd_xml(f, needed=['position', 'image'])
    # remove 208*250 other molecules in the system
    pos = xml.nodes['position'][250*208:]+xml.box*xml.nodes['image'][250*208:]
    pos = pos.reshape(-1, chain_length, 3)
    normalModes = np.swapaxes(np.tensordot(factors, pos, axes=[1, 1]), 0, 1)
    np.savetxt(f.replace('xml', 'nm.txt'),
               normalModes.reshape(-1, 3), fmt='%.6f')
