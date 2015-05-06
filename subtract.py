import contour_functions as cf
import numpy as np
import cosmology_functions as cosm
from numpy import savez, load
from contour_functions import contour_settings as cs
import matplotlib.pyplot as plt
from matplotlib import cm

add = 'pair' if cs['USE_PAIR_ADDS'] else 'neighbor'
real = 'real' if cs['REAL_PAIRS'] else 'fake'
xb = cs['xbins']
yb = cs['ybins']
nside = cs['nside']
scaled = 'scaled' if cs['SCALED'] else 'unscaled'
weighted = 'weighted' if cs['WEIGHTED'] else 'unweighted'
angular = 'angular' if cs['ANGULAR_BINS'] else 'physical'
half_width = 15. / cosm.angDiamDistSingle(.16)
half_height = half_width

real = load('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/step2_0_o_%d_real_%s_adds_%s_%s_%s_%dx%d_nside%d.npz' % (cs['chunks'], add, scaled, weighted, angular, xb, yb, nside))['grid']

fake = load('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/step2_0_o_%d_fake_%s_adds_%s_%s_%s_%dx%d_nside%d.npz' % (cs['chunks'], add, scaled, weighted, angular, xb, yb, nside))['grid']

for j in range(0, cs['chunks'] + 1):
    real += load('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/step2_%d_o_%d_real_%s_adds_%s_%s_%s_%dx%d_nside%d.npz' % (j, cs['chunks'], add, scaled, weighted, angular, xb, yb, nside))['grid']
    fake += load('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/step2_%d_o_%d_fake_%s_adds_%s_%s_%s_%dx%d_nside%d.npz' % (j, cs['chunks'], add, scaled, weighted, angular, xb, yb, nside))['grid']

#real = real[6:12,6:12]
#fake = fake[6:12,6:12]

subt = real - fake

###

#subt = np.where(fake != 0, np.divide(subt, fake), 0)

savez('/home/chadavis/catalog_creation/LRG-pair-filaments/steps/temp_files/step2_subtracted_%s_adds_%s_%s_%s_%dx%d_nside%d.npz' % (add, scaled, weighted, angular, xb, yb, nside), grid=subt)

fig, axes = plt.subplots(nrows=1, ncols=3)


print real[5,5]
print fake[5,5]
print subt[5,5]
print np.sqrt(real[5,5] + fake[5,5])
#real = np.divide(real, np.sqrt(real))
#fake = np.divide(fake, np.sqrt(fake))
#subt = np.divide(np.abs(subt), np.sqrt(np.add(fake, real)))
print np.max(subt[np.where(np.invert(np.isnan(subt)))[0]])

to_plot = (real, fake, subt)
names = ('Real Counts', 'Fake Counts', 'Subtracted Counts')
for i, ax in zip(range(len(to_plot)), axes.flat):
    if i == 2:
        im = ax.imshow(to_plot[i], interpolation='None', cmap=cm.coolwarm, extent=[-half_width, half_width, -half_height, half_height], vmax=.1)
    else:
        im = ax.imshow(to_plot[i], interpolation='None', cmap=cm.coolwarm, extent=[-half_width, half_width, -half_height, half_height])
    ax.set_title(names[i])
    ax.set_xlabel('Angle (radians)')
    ax.set_ylabel('Angle (radians)')

cax = fig.add_axes([.93, .1, .018, .8])
fig.colorbar(im, cax=cax, cmap=cm.coolwarm)
fig.subplots_adjust(wspace = .3)
#fig.tight_layout()

'''
plt.subplot(131)
plt.imshow(real, interpolation='None', cmap=cm.coolwarm, extent=[-half_width, half_width, -half_height, half_height])
plt.xlabel('Angle (radians)')
plt.ylabel('Angle (radians)')
plt.title('Real counts')
plt.subplot(132)
plt.imshow(fake, interpolation='None', cmap=cm.coolwarm, extent=[-half_width, half_width, -half_height, half_height])
plt.xlabel('Angle (radians)')
plt.ylabel('Angle (radians)')
plt.title('Fake counts')
plt.subplot(133)
plt.imshow(subt, interpolation='None', cmap=cm.coolwarm, extent=[-half_width, half_width, -half_height, half_height])
plt.xlabel('Angle (radians)')
plt.ylabel('Angle (radians)')
plt.title('Subtracted counts')
plt.colorbar()
#plt.tight_layout()
'''
plt.show()
#cf.plotContours(grid)
