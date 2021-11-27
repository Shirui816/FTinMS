import numpy as np
import pandas as pd
from sys import argv



filename = argv[1]
data = pd.read_csv(filename, header=None, comment='#', delim_whitespace=1).values
dx = np.abs(np.diff(data.T[0]))
dx = np.min(dx[dx>0])
dy = np.abs(np.diff(data.T[1]))
dy = np.min(dy[dy>0])
dz = np.abs(np.diff(data.T[2]))
dz = np.min(dz[dz>0])
Nx = int((data.T[0].max()-data.T[0].min())/dx) + 1
Ny = int((data.T[1].max()-data.T[1].min())/dy) + 1
Nz = int((data.T[2].max()-data.T[2].min())/dz) + 1
print("Grid dimension: ", Nx, Ny, Nz)
print("Grid spacing: %.4f %.4f %.4f" % (dx, dy, dz))
print("Box dimension: ", np.array([Nx, Ny, Nz]) * np.array([dx, dy, dz]))
#dx, dy, dz = 0.1, 0.1, 0.1
#Nx, Ny, Nz = 200, 200, 100

dx, dy, dz = 10 * dx/0.529177, 10 * dy/0.529177, 10 * dz/0.529177
data = data.reshape((Nx, Ny, Nz, -1))

o = open('%s.cub' % (filename.replace('.txt', '')), 'w')
o.close()
header = '''CPMD CUBE FILE.
OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z
   1    0.000000    0.000000    0.000000
   %d    %.6f    0.000000    0.000000
   %d    0.000000    %.6f    0.000000
   %d    0.000000    0.000000    %.6f
   1    0.000000    20.000000   20.000000   10.000000
''' % (Nx, dx, Ny, dy, Nz, dz)

o = open('%s.cub' % (filename.replace('.txt', '')), 'a')
o.write(header)
    
#for ix in range(data.shape[0]):
#    for iy in range(data.shape[1]):
#        for iz in range(data.shape[2]):
#            o.write('%.4f ' % data[ix, iy, iz, 3])
#            if iz % 6 == 5:
#                o.write('\n')
#        o.write('\n')

for index in np.ndindex(data.shape[:3]):
    ix, iy, iz = index
    p = data[ix, iy, iz, 3] + data[ix, iy, iz, 7] + data[ix, iy, iz, 11]
    o.write('%.4f ' % (p/3))
    if iz == data.shape[2] - 1:
        o.write('\n')
    if iz % 6 == 5:
        o.write('\n')
