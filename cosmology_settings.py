#+++++++++++++++++++++++++++++++++++++++
#
# Cosmology Settings
#
# Description:
# This file contains the settings for
# all cosmology calculations
# changing the parameters here should 
# affect all scripts in the repo
# 
#---------------------------------------

cosmology_params = {
    'omega_m':0.3,
    'omega_l':0.7,
    'omega_k':0.0,
    'h': 0.7, # unitless hubble parameter (H0/100 km/s/Mpc)
    'c': 3.0e5, #speed of light in km/s
    'H100':100.0, #km/s/Mpc; for conversions to/from h
    }

cosmology_params['Dh'] = cosmology_params['c']/cosmology_params['H100'] #h^-1 * Mpc
cosmology_params['H0'] = cosmology_params['h']*cosmology_params['H100']
