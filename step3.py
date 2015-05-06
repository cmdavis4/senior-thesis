import contour_functions as cf
import numpy as np
from numpy import savez, load
from contour_functions import contour_settings as cs
import matplotlib.pyplot as plt

add = 'pair' if cs['USE_PAIR_ADDS'] else 'neighbor'
real = 'real' if cs['REAL_PAIRS'] else 'fake'
xb = cs['xbins']
yb = cs['ybins']
nside = cs['nside']
scaled = 'scaled' if cs['SCALED'] else 'unscaled'
weighted = 'weighted' if cs['WEIGHTED'] else 'unweighted'
angular = 'angular' if cs['ANGULAR_BINS'] else 'physical'

grid = load('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/step2_0_o_%d_%s_%s_adds_%s_%s_%s_%dx%d_nside%d.npz' % (cs['chunks'], real, add, scaled, weighted, angular, xb, yb, nside))['grid']
print np.max(grid)

for j in range(0, cs['chunks'] + 1):
    temp = load('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/step2_%d_o_%d_%s_%s_adds_%s_%s_%s_%dx%d_nside%d.npz' % (j, cs['chunks'], real, add, scaled, weighted, angular, xb, yb, nside))['grid']    
    grid += temp

print np.max(grid)

savez('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/step2_complete_%s_%s_adds_%s_%s_%s_%dx%d_nside%d.npz' % (real, add, scaled, weighted, angular, xb, yb, nside), grid=grid)

plt.imshow(grid, interpolation='None')
plt.show()
#cf.plotContours(grid)
