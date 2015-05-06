#+++++++++++++++++++++++++++++++++++++++
#
# Spherical Geometry
#
# Description:
# This file contains the functions for
# spherical geometry calcuations 
# including rotations, coordinate transformations
# and angular separations
# 
#---------------------------------------

import numpy as np
import numpy.linalg as LA

def to_sphere(point):
    """accepts a x,y,z, coordinate and returns the same point in spherical coords (r, theta, phi) i.e. (r, azimuthal angle, altitude angle) with phi=0 in the zhat direction and theta=0 in the xhat direction"""
    coord = (np.sqrt(np.sum(point**2)),
             np.arctan2(point[1],point[0]),
             np.pi/2.0 - np.arctan2(np.sqrt(point[0]**2 + point[1]**2),point[2])
             )
    
    return np.array(coord)
def to_cartesian(point):
    """accepts point in spherical coords (r, theta, phi) i.e. (r, azimuthal angle, altitude angle) with phi=0 in the zhat direction and theta=0 in the xhat direction and returns a x,y,z, coordinate"""
    coord = point[0]*np.array([np.cos(point[2])*np.cos(point[1]),
                                 np.cos(point[2])*np.sin(point[1]),
                                 np.sin(point[2])])
    
    
    return coord
def rotate(p1, p2, p3):
    """rotates coordinate axis so that p1 is at the pole of a new spherical coordinate system and p2 lies on phi (or azimuthal angle) = 0

inputs:
p1: vector in spherical coords (phi, theta, r) where phi is azimuthal angle (0 to 2 pi), theta is zenith angle or altitude (0 to pi), r is radius
p2: vector of same format 
p3: vector of same format

Output:
s1:vector in (r, theta, phi) with p1 on the z axis
s2:vector in (r,, theta, phi) with p2 on the phi-hat axis
s3:transformed vector of p3
"""
    p1_cc=to_cartesian(p1) 
    p2_cc=to_cartesian(p2)
    p3_cc=map(to_cartesian, p3)

    p1norm = p1_cc/LA.norm(p1_cc)
    p2norm = p2_cc/LA.norm(p2_cc)
    p3norm = map(lambda x: x/LA.norm(x), p3_cc)
    
    zhat_new =  p1norm
    x_new = p2norm - np.dot(p2norm, p1norm) * p1norm
    xhat_new = x_new/LA.norm(x_new)
    yhat_new = np.cross(zhat_new, xhat_new)
    
    dot_with_units = lambda x: [np.dot(x, xhat_new), np.dot(x, yhat_new),
                                np.dot(x, zhat_new)]
    
    s1 = np.array(dot_with_units(p1_cc))
    s2 = np.array(dot_with_units(p2_cc))
    s3 = np.array(map(dot_with_units, p3_cc))

    s1=to_sphere(s1)
    s2=to_sphere(s2)
    s3=np.array(map(to_sphere, s3))
    
    return s1, s2, s3

def calc_distance(center, points):
    '''Calculate the circular angular distance of two points on a sphere.'''
    center = np.array(center)
    points = np.array(points)
    lambda_diff = center[0]  - points[:,0]
    cos1 = np.cos(center[1])
    cos2 = map(np.cos, points[:,1])
    sin1 = np.sin(center[1])
    sin2 = map(np.sin, points[:,1])
    
    num1 = np.power(np.multiply(cos2, map(np.sin, lambda_diff)), 2.0)
    num2 = np.subtract(np.multiply(cos1, sin2), np.multiply(np.multiply(sin1, cos2), map(np.cos, lambda_diff)))
    num = np.add(num1, np.power(num2, 2.0))
    denom1 = np.multiply(sin1, sin2)
    denom2 = np.multiply(np.multiply(cos1, cos2), map(np.cos, lambda_diff))
    denom = np.add(denom1, denom2)
    
    return map(np.arctan2, map(np.sqrt, num), denom)