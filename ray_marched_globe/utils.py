import numpy as np
import torch, torch.nn.functional as F

# NOTE: float64 IS HIGHLY RECOMMENDED!
#       otherwise you get possibly substantial error.

# All linear distances in meters * R1^-1
# All angles in radians
# WebMercator in [-1,1]^2
#     In fact WM y may lie outside -1,1 if lat magnitude greater than 85 deg

#a = (6378137.0)
#b = (6356752.314245179)
a = (1.0)
b = (6356752.314245179/6378137.0)
#b = (6378137.0/6356752.314245179)
a2 = a*a
b2 = b*b
e2 = 1 - (b*b / a*a)
ae2 = a*e2

# https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
def ecef_to_geodetic(x):
    lng = torch.atan2(x[:,0],x[:,1])

    p = torch.norm(x[:,:2],dim=1)
    p2 = p * p
    z2 = x[:,2]**2

    #kappa = (p / x[:,:2]) * torch.tan(lat)

    # It seems that when height-above-ellipsoid (h) is 0, this is exact
    # and the iterative refinement not needed.
    # When h>0, it converges rapidly
    k_i = torch.ones_like(p) * (1./(1-e2))

    for i in range(2):
        #c_i = (p2 + (1 - e2) * z2 * k_i**2).pow(1.5) / ae2
        c_i = (p2 + (1 - e2) * z2).mul_(k_i.pow(2)).pow_(1.5).div_(ae2)

        #k_i_old = k_i
        #k_i = 1 + (p2 + (1 - e2) * z2 * k_i**3) / (c_i - p2)
        k_i = 1 + (p2 + (1 - e2) * z2).mul_(k_i.pow_(3)) .div_ (c_i - p2)

        #print('delta kappa:', (k_i - k_i_old))

    #print(' - Got k:\n',k_i.cpu().numpy())
    #lat = (k_i * Z / p).atan()
    lat = torch.atan2(k_i * x[:,2], p)

    h = (1/e2) * (1/k_i - (1-e2)) * (p2 + z2*(k_i**2)).sqrt()

    return torch.stack((lng,lat,h),-1)

def geodetic_to_ecef(llh):
    lamb, phi, h = llh.T

    sin_phi = torch.sin(phi)
    n_phi = a / torch.sqrt(1 - e2 * sin_phi**2)

    cos_phi, cos_lamb = phi.cos(), lamb.cos()
    sin_phi, sin_lamb = phi.sin(), lamb.sin()

    return torch.stack((
        (n_phi + h) * cos_phi * sin_lamb,
        (n_phi + h) * cos_phi * cos_lamb,
        ((b2/a2) * n_phi + h) * sin_phi), -1)

one_div_two_pi = 1. / (2*np.pi)
def geodetic_to_unit_wm(ll):
    return torch.stack((
        (ll[:,0] + np.pi).mul_(one_div_two_pi),
        (np.pi - (np.pi/4 + ll[:,1]*.5).tan_().log_()).mul_(one_div_two_pi)
        #one_div_two_pi * (np.pi - torch.log(torch.tan(np.pi/4 + ll[:,1]*.5)))
    ), -1) * 2 - 1

if __name__ == '__main__':
    #x = torch.randn(8,3)
    #x = x / torch.norm(x, dim=1, keepdim=True)
    llh_0 = (torch.rand(8,3) - .5) * np.pi * 2
    llh_0[:,2].mul_(1e-5)
    #llh_0 = llh_0.to(torch.float64)
    print(' - true llh:\n',llh_0)
    x = geodetic_to_ecef(llh_0)
    print(' - true x:\n',x)

    llh = ecef_to_geodetic(x)
    #llh = torch.cat((ll,torch.zeros_like(ll[:,:1])),-1)
    print(' - solved llh:\n',llh)
    rx = geodetic_to_ecef(llh)
    print(' - solved x:\n',rx)

    print(' - Cycle Error:\n', (rx-x).norm(dim=1))

    print(' - WM:\n', geodetic_to_unit_wm(llh))
