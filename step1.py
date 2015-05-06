import contour_functions as cf
import numpy as np
import pickle_functions as pf
from contour_functions import contour_settings as cs
import cProfile
import time

real = 'real' if cs['REAL_PAIRS'] else 'fake'

pairs, neighbors, pfd, nfd = cf.addPixAndRadians()
t = time.time()

neighbors, nfd = cf.weight(neighbors, nfd)
neighbors = cf.cleanNeighbors(neighbors, nfd)
pix_dict = cf.pixDict(neighbors, nfd)
np.savez('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/step1_%s.npz' % real, neighbors = neighbors, pairs=pairs)
np.save('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/neighbors.npy', neighbors)
pf.save_obj(pix_dict,'/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/pix_dict.pkl')
pf.save_obj(pfd, '/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/pfd_%s.pkl' % real)
pf.save_obj(nfd, '/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/nfd.pkl')
