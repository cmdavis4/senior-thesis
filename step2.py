import contour_functions as cf
import numpy as np
import os
from contour_functions import contour_settings as cs
import pickle_functions as pf

add = 'pair' if cs['USE_PAIR_ADDS'] else 'neighbor'
real = 'real' if cs['REAL_PAIRS'] else 'fake'
xb = cs['xbins']
yb = cs['ybins']
nside = cs['nside']
scaled = 'scaled' if cs['SCALED'] else 'unscaled'
weighted = 'weighted' if cs['WEIGHTED'] else 'unweighted'
angular = 'angular' if cs['ANGULAR_BINS'] else 'physical'

print ('Reading...')
pairs = np.load('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/step1_%s.npz' % real)['pairs']
neighbors = np.load('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/neighbors.npy')

if not cs['REAL_PAIRS']:
    reals = np.load('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/step1_real.npz')['pairs']
    rfd = pf.load_obj('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/pfd_real.pkl')

pix_dict = pf.load_obj('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/pix_dict.pkl')
nfd = pf.load_obj('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/nfd.pkl')
pfd = pf.load_obj('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/pfd_%s.pkl' % real)

print len(nfd)
print len(neighbors[0])
print neighbors[1]
print nfd

chunknum = int(os.environ['SGE_TASK_ID']) - 1
#chunknum = 62

reals_arg = reals if not cs['REAL_PAIRS'] else None
rfd_arg = rfd if not cs['REAL_PAIRS'] else None

if cs['ANGULAR_BINS']:
    grid = cf.genAngContours(pairs, neighbors, pix_dict, pfd, nfd, chunknum, cs['chunks'], reals_arg, rfd_arg)
else:
    grid = cf.genContours(pairs, neighbors, pix_dict, pfd, nfd, chunknum, cs['chunks'], reals_arg, rfd_arg)


np.savez('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/step2_%d_o_%d_%s_%s_adds_%s_%s_%s_%dx%d_nside%d.npz' % (chunknum, cs['chunks'], real, add, scaled, weighted, angular, xb, yb, nside), grid=grid)
