import numpy as np
import healpy as hp
import bisect as bs
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt
import time
import cProfile
from cosmology_settings import cosmology_params as cp
from scipy.optimize import brentq
from scipy.integrate import quad

###LOCAL IMPORTS###
import cosmology_functions as cf
import spherical_geo as sg
from user_settings import user_settings as us

contour_settings = {
    'USE_PAIR_ADDS':True,
    'ANGULAR_BINS':True,
    'REAL_PAIRS':False,
    'SCALED':True,
    'WEIGHTED':False,
    'xbins':20,
    'ybins':20,
    'out_name':'test_fig.png',
    'max_distance':15.,
    'nside':16,
    'chunks':600,
    'r_band_cutoff':17.7777,
    'out_dir':'/home/chadavis/catalog_creation/LRG-pair-filaments/steps/results/',
    'real_pair_path':'/home/chadavis/catalog_creation/LRG-pair-filaments/pairs_6_10.csv',
    'fake_pair_path':'/home/chadavis/catalog_creation/LRG-pair-filaments/steps/fake_pairs_sampled.csv',
    #'neighbor_path':'/home/chadavis/catalog_creation/LRG-pair-filaments/steps/neighbors_ra_dec.csv'
    'neighbor_path':'/home/chadavis/catalog_creation/LRG-pair-filaments/dr10_allspec.csv'
    }

cs = contour_settings

pair_adds = cs['USE_PAIR_ADDS']
scaled = 'scaled' if cs['SCALED'] else 'unscaled'
add_word = 'pair' if pair_adds else 'neighbor'
nside = cs['nside']
cut = cs['r_band_cutoff']
real = cs['REAL_PAIRS']

c = 3E8

measured_ks = np.load('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/k_r_early.npy')

def interpolate(x, xs, ys):
    #Linear interpolation
    #Assumes no duplicate values of x
    x = float(x)
    if not isinstance(xs, np.ndarray): xs = np.array(xs)
    if not isinstance(ys, np.ndarray): ys = np.array(ys)
    xs = xs.astype(float)
    ys = ys.astype(float)
    upper = bs.bisect(xs, x)
    if upper == 0: return ys[0]
    if upper == len(xs): return ys[-1]
    fraction = (x - xs[upper - 1]) / (xs[upper] - xs[upper - 1])
    diff = ys[upper]  - ys[upper - 1]
    return ys[upper - 1] + (fraction * diff)

if cs['ANGULAR_BINS']:
    half_width = 15. / cf.angDiamDistSingle(.16)
    half_height = half_width
else:
    half_width = cs['max_distance'] if not scaled else cs['max_distance'] / 3.
    half_height = cs['max_distance'] if not scaled else cs['max_distance'] / 3.


#Converts dec from the usual astrophysical convention to
#the otherwise usual convention, with theta=0 at the pole
#and theta varying from 0 to pi
def healpyDecConvention(decs):
    return np.multiply(np.subtract(decs, np.pi / 2.), -1.)

#Reads pairs and neighbors from the paths specified in contour_settings.py
#and returns structured arrays of the same catalogs, with ra/dec in
#radians and with pixel numbers added under the field 'pix'.
def addPixAndRadians():
    print 'Reading pairs...'
    if cs['REAL_PAIRS']:
        pairs = np.genfromtxt(cs['real_pair_path'], delimiter=',', dtype=str)
    else:
        pairs = np.genfromtxt(cs['fake_pair_path'], delimiter=',', dtype=str)
    pairs_header = pairs[0]
    pairs = pairs[1:]
    print 'Reading neighbors...'
    neighbors = np.genfromtxt(cs['neighbor_path'], delimiter=',', dtype=str)
    neighbors_header = neighbors[0]
    neighbors = neighbors[1:]

    pfd = {pairs_header[j] : j for j in range(len(pairs_header))}

    #####
    #pairs = pairs.astype(float)
    #print np.max(pairs[:,pfd['z1']])
    #print np.max(pairs[:,pfd['z2']])
    #print np.min(pairs[:,pfd['z1']])
    #print np.min(pairs[:,pfd['z2']])
    #quit()
    #####
    
    nfd = {neighbors_header[j] : j for j in range(len(neighbors_header))}

    neighbor_objids = neighbors[:,nfd['objid']]
    pair_objids1 = pairs[:,pfd['objid1']]
    pair_objids2 = pairs[:,pfd['objid2']]

    pairs = pairs.astype(np.float)
    neighbors = neighbors.astype(np.float)
    
    print 'Converting to radians...'
    for i in ['ra_mid', 'dec_mid', 'ra1', 'dec1', 'ra2', 'dec2']:
        pairs[:,pfd[i]] = map(np.radians, pairs[:,pfd[i]])
    for i in ['ra_gal', 'dec_gal']:
        neighbors[:,nfd[i]] = map(np.radians, neighbors[:,nfd[i]])

    print 'Calculating pixels...'
    pair_pix = hp.ang2pix(
        nside,
        healpyDecConvention(pairs[:,pfd['dec_mid']]),
        pairs[:,pfd['ra_mid']],
        nest=True)
    neighbor_pix = hp.ang2pix(
        nside,
        healpyDecConvention(neighbors[:,nfd['dec_gal']]),
        neighbors[:,nfd['ra_gal']],
        nest=True)

    print 'Calculating angles...'
    angles = map(
        sg.calc_distance,
        zip(pairs[:,pfd['ra1']], pairs[:,pfd['dec1']]),
        [[x] for x in zip(pairs[:,pfd['ra2']], pairs[:,pfd['dec2']])]
        )
    angles = np.array(angles).reshape(len(angles), 1)
    
    pairs = np.hstack((pairs, pair_pix.reshape(len(pair_pix), 1), angles))
    neighbors = np.hstack((neighbors, neighbor_pix.reshape(len(neighbor_pix), 1)))
    pairs_header = np.append(pairs_header, 'pix')
    pairs_header = np.append(pairs_header, 'angle')
    neighbors_header = np.append(neighbors_header, 'pix')
    pairs = pairs.astype(str)
    neighbors = neighbors.astype(str)
    pairs[:,pfd['objid1']] = pair_objids1
    pairs[:,pfd['objid2']] = pair_objids2
    neighbors[:,nfd['objid']] = neighbor_objids
    pairs = np.vstack((pairs_header, pairs))
    neighbors = np.vstack((neighbors_header, neighbors))
    nfd['pix'] = len(neighbors_header) - 1
    pfd['pix'] = len(pairs_header) - 2
    pfd['angle'] = len(pairs_header) - 1

    return pairs, neighbors, pfd, nfd

def cleanNeighbors(neighbors, nfd):
    print len(neighbors)
    data = neighbors[1:].astype(float)
    inds = np.where((data[:,nfd['z']] < 0) | (data[:,nfd['weights']] < 0))[0]
    inds = np.add(1, inds)
    neighbors = np.delete(neighbors, inds, axis = 0)
    print len(neighbors)
    return neighbors

def hubble(z):
    return cp['H100'] * cp['h'] * np.sqrt((cp['omega_m'] * (1. + z)**3) + (cp['omega_k'] * (1. + z)**2) + cp['omega_l'])

def kcorr_r(zs): return interpolate(zs, measured_ks[:,0], measured_ks[:,1])

def volumeIntegral(z):
    if z < 0 : return -9999
    f = lambda z: cf.angDiamDistSingle(z)**3 * c / (hubble(z) * (1. + z))
    return quad(f, 0, z)[0]

def q(z):
    return .86 * z

zs = np.linspace(.000001, 3, 10000)
fixed = np.array(map(lambda z: cut - 5 * np.log10(cf.luminosityDistance(z) / 1E-5) - kcorr_r(z) + q(z), zs))
f_of_z = lambda z: interpolate(z, zs, fixed)

def weight(neighbors, nfd):
    print 'Calculating weights...'
    orig = neighbors
    if not cs['WEIGHTED']: weights = np.array([1] * (len(neighbors) - 1))
    else:
        neighbors = neighbors[1:].astype(float)
        dl = np.array(map(cf.luminosityDistance, neighbors[:,nfd['z']]))
        M = neighbors[:,nfd['petroMag_r']] - 5 * np.array(map(np.log10, dl / 1E-5)) - map(kcorr_r, neighbors[:,nfd['z']]) + np.array(map(q, neighbors[:,nfd['z']]))
        z_max = []
        count = 0
        sameSign = 0
        for j in M:
            try:
                z_max.append(brentq(lambda z: j - f_of_z(z), 0, 5))
            except ValueError:
                z_max.append(-9999)
                sameSign+=1
            count +=1
        print sameSign
        weights = np.divide(1, np.array(map(volumeIntegral, z_max)))
    weights = weights.astype(str).reshape((len(weights), 1))
    weights = np.vstack(('weights', weights))
    #print np.shape(orig)
    #print np.shape(weights)
    orig = np.hstack((orig, weights))
    nfd = {orig[0,i]:i for i in range(len(orig[0]))}
    return orig, nfd
    


    
    #dl = map(cf.luminosityDistance, neighbors[:,nfd['z']])
    #M = neighbors[:,nfd['petroMag_r']] - 5 * log(dl/10) - neighbors[:,nfd['kcorrR']] + neighbors[:,nfd['z']]
    
    
    

def pixDict(neighbors, nfd):
    print 'Generating pixel dictionary...'
    pix_dict = {}
    pix_nums = neighbors[1:,nfd['pix']].astype(float).astype(int)
    #Add the index of each object as a value, with its pixel
    #number as key
    for n in range(len(pix_nums)):
        pix_dict.setdefault(pix_nums[n], []).append(n)
    return pix_dict

def genContours(pairs, neighbors, pix_dict, pfd, nfd,  chunknum, chunks, real=None, rfd=None):
    if ((real != None) and (rfd != None) and not cs['REAL_PAIRS']):
        real = real[1:].astype(float)
        angles = map(
            sg.calc_distance,
            zip(real[:,rfd['ra1']], real[:,rfd['dec1']]),
            np.array([[x] for x in zip(real[:,rfd['ra2']], real[:,rfd['dec2']])])
            )
        angles = np.array(angles).reshape(len(real), 1)
        real = np.hstack((real, angles))
        real = real[np.argsort(real[:,-1])]
        angles = real[:,-1]
        #angles = angles[np.argsort(angles)]
        #real = real[np.argsort(angles)]
    #print pairs[0]
    #print pfd
    #print min(angles)
    #print max(angles)
    lb = (chunknum * len(pairs) / chunks)
    ub = ((chunknum+1) * len(pairs) / chunks)
    pairs = pairs[1:]
    neighbors = neighbors[1:]
    pair_objid1 = pairs[:,pfd['objid1']]
    pair_objid2 = pairs[:,pfd['objid2']]
    neighbor_objid = neighbors[:,nfd['objid']]
    pairs = pairs.astype(np.float)
    neighbors = neighbors.astype(np.float)
    pairs = pairs[lb:ub]
    no_neighbors = 0
    oob_pairs = 0
    oob_neighbors = 0

    x_bins = np.linspace(-1. * half_width, half_width, cs['xbins'])[1:-1]
    y_bins = np.linspace(-1. * half_height, half_height, cs['ybins'])[1:-1]

    grid = np.zeros(shape=(cs['xbins'], cs['ybins']))

    t = time.time()
    print 'Adding to contours...'
    for p in range(len(pairs)):
        print (p + lb)
        curr = pairs[p]
        near_pix = np.append(hp.get_all_neighbours(nside, int(curr[pfd['pix']]), nest=True), [curr[pfd['pix']]])
        near_pix = near_pix[np.where(near_pix != -1)[0]]
        candidate_inds = []
        for j in near_pix:
            #print j
            try:
                candidate_inds.append(pix_dict[int(j)])
            except KeyError:
                pass
        candidate_inds = np.concatenate(candidate_inds)
        candidate_inds = candidate_inds[np.where(
            (neighbor_objid[candidate_inds] != pair_objid1[p])
            & (neighbor_objid[candidate_inds] != pair_objid2[p])
            )[0]]
        #print len(candidate_inds)
        candidates = neighbors[np.array(candidate_inds)]
        #print curr[pfd['ra1']]
        if cs['REAL_PAIRS']:
            distances = cf.physicalDistance(
                (curr[pfd['ra_mid']], curr[pfd['dec_mid']]),
                zip(candidates[:,nfd['ra_gal']], candidates[:,nfd['dec_gal']]),
                np.array([curr[pfd['z']]] * len(candidates)) if pair_adds else candidates[:,nfd['z']]
                )
        else:
            curr_angle = sg.calc_distance(
                (curr[pfd['ra1']], curr[pfd['dec1']]),
                [(curr[pfd['ra2']], curr[pfd['dec2']])]
                )[0]
            #print curr_angle
            match_ind = np.searchsorted(angles, curr_angle)
            #print angles[match_ind]
            real_z = real[match_ind, rfd['z']]
            distances = cf.physicalDistance(
                (curr[pfd['ra_mid']], curr[pfd['dec_mid']]),
                zip(candidates[:,nfd['ra_gal']], candidates[:,nfd['dec_gal']]),
                np.array([real_z] * len(candidates)) if pair_adds else candidates[:,nfd['z']]
                )
        #print real_z
        #print curr_angle
        #print real_z
        pair_radius = cf.physicalDistance(
        (curr[pfd['ra_mid']], curr[pfd['dec_mid']]),
        [(curr[pfd['ra1']], curr[pfd['dec1']])],
        [curr[pfd['z']] if cs['REAL_PAIRS'] else real_z]
        )
        
        inbounds = candidates[np.where(distances < cs['max_distance'])[0]]
        oob_neighbors += len(candidates) - len(inbounds)
        distances = distances[np.where(distances < cs['max_distance'])[0]]
        #print distances[5:10]
        unrotated_neighbors = np.ones((len(inbounds), 3))
        unrotated_neighbors[:,1] = inbounds[:,nfd['ra_gal']]
        unrotated_neighbors[:,2] = inbounds[:,nfd['dec_gal']]
        unrotated_mid = (1., curr[pfd['ra_mid']], curr[pfd['dec_mid']])
        unrotated_right = (1., curr[pfd['ra1']], curr[pfd['dec1']])
        rotated_mid, rotated_right, rotated_neighbors = sg.rotate(
            unrotated_mid,
            unrotated_right,
            unrotated_neighbors
            )
        #print cf.physicalDistance(
        #    (rotated_mid[1], rotated_mid[2]),
        #    zip(rotated_neighbors[:,1], rotated_neighbors[:,2]),
        #    np.array([curr[pfd['z']]] * len(rotated_neighbors)) if pair_adds else candidates[:,nfd['z']])[5:10]
        if (pair_radius > 5 or pair_radius < 3):
            oob_pairs +=1
            print 'oob'
            continue
        if scaled:
            scaled_distances = np.divide(distances, pair_radius)
            xs = np.multiply(map(np.cos, rotated_neighbors[:,1]), scaled_distances)
            ys = np.multiply(map(np.sin, rotated_neighbors[:,1]), scaled_distances)
        else:
            xs = np.multiply(map(np.cos, rotated_neighbors[:,1]), distances)
            ys = np.multiply(map(np.sin, rotated_neighbors[:,1]), distances)
        xbs = map(lambda j: bs.bisect(x_bins, j), xs)
        ybs = map(lambda j: bs.bisect(y_bins, j), ys)
        def add_to_bins(ybin, xbin, weight):
            grid[ybin, xbin] += weight
            return
        if cs['WEIGHTED']: weights = np.divide(1., inb)
        else: weights = np.array([1.] * len(inbounds))
        map(add_to_bins, ybs, xbs, weights)
        #print len(weights)
        #print oob_neighbors
    print np.max(grid)

    return grid

def genAngContours(pairs, neighbors, pix_dict, pfd, nfd,  chunknum, chunks, real=None, rfd=None):
    print len(neighbors)
    #objids_removed = 0
    #rem_angles = []
    if ((real != None) and (rfd != None) and not cs['REAL_PAIRS']):
        real = real[1:].astype(float)
        real = real[np.argsort(real[:,rfd['angle']])]
        angles = real[:,-1]
        #angles = angles[np.argsort(angles)]
        #real = real[np.argsort(angles)]
    #print pairs[0]
    #print pfd
    #print min(angles)
    #print max(angles)
    lb = (chunknum * len(pairs) / chunks)
    ub = ((chunknum+1) * len(pairs) / chunks)
    pairs = pairs[1:]
    neighbors = neighbors[1:]
    pair_objid1 = pairs[:,pfd['objid1']]
    pair_objid2 = pairs[:,pfd['objid2']]
    neighbor_objid = neighbors[:,nfd['objid']]
    pairs = pairs.astype(np.float)
    neighbors = neighbors.astype(np.float)
    print neighbors[:5,nfd['weights']]
    pairs = pairs[lb:ub]
    no_neighbors = 0
    oob_pairs = 0
    oob_neighbors = 0

    x_bins = np.linspace(-1. * half_width, half_width, cs['xbins'])[1:-1]
    y_bins = np.linspace(-1. * half_height, half_height, cs['ybins'])[1:-1]

    grid = np.zeros(shape=(cs['xbins'], cs['ybins']))

    t = time.time()
    print 'Adding to contours...'
    for p in range(len(pairs)):
        print (p + lb)
        curr = pairs[p]
        near_pix = np.append(hp.get_all_neighbours(nside, int(curr[pfd['pix']]), nest=True), [curr[pfd['pix']]])
        near_pix = near_pix[np.where(near_pix != -1)[0]]
        candidate_inds = []
        for j in near_pix:
            #print j
            try:
                candidate_inds.append(pix_dict[int(j)])
            except KeyError:
                pass
        candidate_inds = np.concatenate(candidate_inds)
        #a = len(candidate_inds)
        #print '###'
        #print neighbor_objid[candidate_inds]
        #print pair_objid1[p]
        #print pair_objid2[p]
        #t = neighbor_objid[candidate_inds]
        #out_inds = np.where(
        #    (t == pair_objid1[p].astype('str'))
        #    | (t == pair_objid2[p].astype('str'))
        #    )[0]
        #print out_inds
        #candidate_inds = candidate_inds[np.where(
        #    (neighbor_objid[candidate_inds] != pair_objid1[p].astype('str'))
        #    & (neighbor_objid[candidate_inds] != pair_objid2[p].astype('str'))
        #    )[0]]
        #a -= len(candidate_inds)
        #objids_removed += a
        #print len(candidate_inds)
        candidates = neighbors[np.array(candidate_inds)]
        #candidates=neighbors
        #print curr[pfd['ra1']]
        if cs['REAL_PAIRS']:
            distances = cf.physicalDistance(
                (curr[pfd['ra_mid']], curr[pfd['dec_mid']]),
                zip(candidates[:,nfd['ra_gal']], candidates[:,nfd['dec_gal']]),
                np.array([curr[pfd['z']]] * len(candidates)) if pair_adds else candidates[:,nfd['z']]
                )
        else:
            curr_angle = curr[pfd['angle']]
            #print curr_angle
            match_ind = np.searchsorted(angles, curr_angle)
            #print angles[match_ind]
            real_z = real[match_ind, rfd['z']]
            distances = cf.physicalDistance(
                (curr[pfd['ra_mid']], curr[pfd['dec_mid']]),
                zip(candidates[:,nfd['ra_gal']], candidates[:,nfd['dec_gal']]),
                np.array([real_z] * len(candidates)) if pair_adds else candidates[:,nfd['z']]
                )
        #print real_z
        #print curr_angle
        #print real_z
        pair_radius = cf.physicalDistance(
        (curr[pfd['ra_mid']], curr[pfd['dec_mid']]),
        [(curr[pfd['ra1']], curr[pfd['dec1']])],
        [curr[pfd['z']] if cs['REAL_PAIRS'] else real_z]
        )
        #print pair_radius
        print len(candidates)
        inbounds = candidates[np.where(distances < cs['max_distance'])[0]]
        if len(inbounds) == 0: continue
        print len(inbounds)
        #print len(inbounds)
        oob_neighbors += len(candidates) - len(inbounds)
        distances = distances[np.where(distances < cs['max_distance'])[0]]
        #print distances[5:10]
        unrotated_neighbors = np.ones((len(inbounds), 3))
        unrotated_neighbors[:,1] = inbounds[:,nfd['ra_gal']]
        unrotated_neighbors[:,2] = inbounds[:,nfd['dec_gal']]
        unrotated_mid = (1., curr[pfd['ra_mid']], curr[pfd['dec_mid']])
        unrotated_right = (1., curr[pfd['ra1']], curr[pfd['dec1']])
        rotated_mid, rotated_right, rotated_neighbors = sg.rotate(
            unrotated_mid,
            unrotated_right,
            unrotated_neighbors
            )
        #print cf.physicalDistance(
        #    (rotated_mid[1], rotated_mid[2]),
        #    zip(rotated_neighbors[:,1], rotated_neighbors[:,2]),
        #    np.array([curr[pfd['z']]] * len(rotated_neighbors)) if pair_adds else candidates[:,nfd['z']])[5:10]
        if (pair_radius > 5 or pair_radius < 3):
            oob_pairs +=1
            print 'oob'
            continue
        print len(rotated_neighbors)
        neighbor_angles = sg.calc_distance(
            (rotated_mid[1], rotated_mid[2]),
            zip(rotated_neighbors[:,1], rotated_neighbors[:,2])
            )
        neighbor_angles = np.array(neighbor_angles)
        #rem_angles.append(neighbor_angles[out_inds])
        #print np.shape(neighbor_angles)
        #a = len(neighbor_angles)
        #print neighbor_angles[:5]
        #print np.where(neighbor_angles > .0000096)
        #neighbor_angles = neighbor_angles[np.where(neighbor_angles > .0000096)[0]]
        #print np.shape(neighbor_angles)
        #objids_removed += (a - len(neighbor_angles))
        #print (a - len(neighbor_angles))
        #print '###'
        xs = np.multiply(np.array(map(np.cos, rotated_neighbors[:,1])), neighbor_angles)
        ys = np.multiply(np.array(map(np.sin, rotated_neighbors[:,1])), neighbor_angles)
        xbs = map(lambda j: bs.bisect(x_bins, j), xs)
        ybs = map(lambda j: bs.bisect(y_bins, j), ys)
        def add_to_bins(ybin, xbin, weight):
            grid[ybin, xbin] += weight
            return
        print inbounds[:5,nfd['weights']]
        if cs['WEIGHTED']: weights = inbounds[:,nfd['weights']]
        else: weights = np.array([1.] * len(inbounds))
        map(add_to_bins, ybs, xbs, weights)
        #print len(weights)
        #print oob_neighbors
    #rem_angles = np.concatenate(rem_angles)
    #print rem_angles
    print np.max(grid)
    #print objids_removed

    return grid


def plotContours(grid):

    x = np.linspace(-1. * half_width, half_width, cs['xbins'])
    y = np.linspace(-1. * half_height, half_height, cs['ybins'])

    #print 'Saving grid...'
    #np.savetxt(cs['out_dir'] + 'grid_%s_adds_nside%d_%dx_%dy.csv' % (add_word, nside, cs['xbins'], cs['ybins']), grid, delimiter=',')
    plt.contour(x, y, grid)
    plt.savefig(cs['out_dir'] + 'plot_%s_%s_adds_%s_%s_nside%d_%dx_%dy.png' % (real, add_word, scaled, 'weighted' if cs['WEIGHTED'] else 'unweighted', nside, cs['xbins'], cs['ybins']))
    plt.show()


def plot3d(grid):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    x = np.linspace(-1. * half_width, half_width, cs['xbins'])
    y = np.linspace(-1. * half_height, half_height, cs['ybins'])

    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, grid)

    #print 'Saving grid...'
    #np.savetxt(cs['out_dir'] + 'grid_%s_adds_nside%d_%dx_%dy.csv' % (add_word, nside, cs['xbins'], cs['ybins']), grid, delimiter=',')
    
    #plt.contour(x, y, grid)
    #plt.savefig(cs['out_dir'] + 'plot_%s_adds_nside%d_%dx_%dy_3d.png' % (add_word, nside, cs['xbins'], cs['ybins']))
    plt.show()
