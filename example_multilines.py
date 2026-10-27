# -------- modules -----------
import os
import sys
import numpy as np
import pandas as pd
import shutil
import sys
import matplotlib

import ptsmodel
from ptsmodel import visualize
from ptsmodel.model_utils import read_lamda_moldata
# ----------------------------


# -------------- Constants -----------------
au    = 1.49598e13      # Astronomical Unit       [cm]
pc    = 3.08572e18      # Parsec                  [cm]
ms    = 1.98892e33      # Solar mass              [g]
ts    = 5.78e3          # Solar temperature       [K]
ls    = 3.8525e33       # Solar luminosity        [erg/s]
rs    = 6.96e10         # Solar radius            [cm]
Ggrav = 6.67428e-8      # gravitational constant  [dyn cm^2 g^-2]
mp    = 1.672621898e-24 # proton mass [g]
sigsb = 5.670367e-5     # Stefan-Boltzmann constant [erg s^-1 cm^-2 K^-4]
kb    = 1.38064852e-16  # Boltzman coefficient in cgs
clight = 2.99792458e10   # light speed [cm s^-1]
# ------------------------------------------


# ------------------- Input ---------------------
# give a list or set parameters & name here
lines = ['co', 'c18o']
Xmols = [1.e-4, 1.e-7]
iline = 2 # J=2--1 for both
# model name
modeldir   = 'run_disk'
modelname  = 'l1527_diskmodel'

# Monte Carlo parameters
# nphot    = 10000000
nphot    = 100000  # small photon number for quick example
iseed    = -17933201

# model name
dustopac  = 'nrmS03'

# Grid parameters !!! spherical coordiante !!!
nr = 32  # radius from center
ntheta = 32  # angle from z axis, 90 deg is disk plane
nphi = 1  # azimuthal
rmin = 1.*au  # minimum radius of the domein
rmax = 1000.*au  # maximum radius of the domein
thetamin = 0.  # from z axis

# input parameter set
# Global properties
mu         = 2.37    # H2 molecular weight. Kauffman (2008)
gtod_ratio = 100.   # gas to dust mass ratio
dist       = 140.   # pc

# Stellar parameters
mstar = 0.45  # stellar mass [Msun]
mstar = mstar*ms  # Msun --> g
lstar = 2.0*ls  # Steller Luminosity
rstar = 4.*rs  # Stahler, Shu and Taam (1980), for protostars
pstar = np.array([0.,0.,0.])  # position

# Disk parameters
# plsig: power-law index of the surface density
# rin, rout: disk inner and outer radius [cm]
mdisk = 1.3e-2*ms  # Total disk mass (gas mass) [Msun]
plsig = -1.  # power-law index of the surface density
rd_in = 1.0*au  # dust & gas disk inner radius [cm]
rd_out = 80.*au  # dust & gas disk outer radius [cm]
Tdisk_1au = 400.  # disk temperature at 1 au [K]
cs_1au = np.sqrt(kb*Tdisk_1au/(mu*mp))  # isothermal sound speed at 1 au [cm/s]
Omg_1au = np.sqrt(Ggrav*mstar/au**3)  # Keplerian angular velocity at 1 au[/s]
hr0 = cs_1au/Omg_1au/au  # H/r at 1au, assuming hydrostatic equillibrium
plh = 0.25  # Powerlaw of flaring h/r, assuming T prop r^-0.5 & Keplerian rotation

# Envelope parameters
rcent = rd_out  # centrifugal radius, determing rotational velocity [cm]
re_in = rd_in  # envelope inner radius [cm]
re_out = 2000.*au  # envelope outer radius [cm]

# Parameters for observations
inc = 85.  # inclination angle [deg]
# inc = 180.-inc  # Upside down; make the disk lower side face the observer
pa  = 90.  # position angle measured from north to east [deg]
sizeau = 500.  # image size [au]
npix = 512  # pixel size of image
width_spw = 5  # badwidth; +/- 5 km/s in this case
nchan = 32  # channel number; bandwidth / (nchan - 1) will be the channel spacing

# Parameters for fits export
obname = 'L1527 IRS'  # Object or model name
coordinate_center = '4h39m53.88s +26d03m09.55s'  # Map center in RA, Dec
frame = 'fk5'  # Coordinate frame
projection = 'SIN'  # Coordinate projection
vsys = 5.8  # systemic velocity [km/s]
beam_convolution = True  # Convolve a beam if True
beam = [0.1, 0.1, 0.]  # Beam size as [bmaj, bmin, bpa] in units of ['', '', deg]
Tb = False  # Return fits file in unit of Tb instead of Jy/beam (or Jy/pixel) if True.
add_noise = False  # Add thermal noise if True
rms = 0.22  # rms noise
# --------------------------------------------------


# --------------- Start -----------------
# Start modeling
dir_i       = modeldir
modelname_i = modelname
print ('Start modeling.')

# Output directory
if os.path.exists(dir_i):
    print ('Directory %s already exists'%dir_i)
else:
    print ('Make the directory %s'%dir_i)
    os.mkdir(dir_i)

# copy files necessary
shutil.copy('./infiles/dustkappa_nrmS03.inp', dir_i)
shutil.copy('./infiles/molecule_c18o.inp', dir_i)
shutil.copy('./infiles/molecule_co.inp', dir_i)

# change directory
os.chdir(dir_i)
os.getcwd()
print ('Now in %s'%dir_i)

# !!! Modeling !!!
# Grid
model_i = ptsmodel.PTSMODEL(modelname_i, nr, ntheta, nphi ,rmin, rmax,
 thetamin=thetamin, thetamax=np.pi*0.5, phimin=0., phimax=2.*np.pi, line=lines,
 Xmol=Xmols)
print ('Model name %s'%model_i.modelname)

# Star
model_i.prop_star(mstar, lstar, rstar, pstar, tstar=None)

# Disk
model_i.rho_cutoffdisk(mdisk, rd_in, rd_out, plsig, hr0, plh, sig0=False)

# Envelope
# Get disk edge density to use it as the reference envelope density
rho0_e = 0.5*model_i.rho_disk[model_i.r <= rd_out, -1, nphi//2][-1]
print ('Envelope density at Rd: %.2e'%rho0_e)
model_i.rho_envUl76(rho0_e, rcent, re_in=re_in, re_out=re_out)

# Outflow cavity
model_i.outflow_cavity(ang=45.)

# Density model
model_i.rho_model(mu, gtod_ratio, disk_height=2)

# Velocity field
model_i.vfield_model()

# Visualize density distributions
print('Plot model..')
visualize.dust_density(model_i, rlim = [0.4, 2e2],
    xlim=[-200., 200.], ylim=[-200., 200.], zlim=[0., 40.])
for i, line in enumerate(lines):
    visualize.gas_density(
        model_i,
        outname = 'gas_density_%s'%line,
        rlim = [0.4, 2e2],
        zlim=[0., 40.],
        xlim=[-200., 200.],
        ylim=[-200., 200.],
        imol = i)

# export to RADMC-3D format
model_i.export_to_radmc3d(nphot, dustopac, iseed)

# RADMC-3D
# calculate temperature profile
model_i.run_mctherm(nthreads=8)
visualize.temperature(
    model_i,
    rlim=[1., 200.],
    zlim=[0., 20.],
    xlim=[-200., 200.],
    ylim=[-200., 200.],
    t_range = [0, 120.],
    )

# solve radiative transfer
for i, line in enumerate(lines):
    model_i.solve_radtrans_line(npix, iline, sizeau, width_spw,
     nchan, pa, inc, imolspec=i+1, contsub=True)

    # export to fits
    for ext in ['', '_cont', '_contsub']:
        imfile  = 'image_%s%i%i%s.out'%(line, iline, iline-1, ext)
        ptsmodel.export_radmc_tofits.export_radmc_tofits(
            modelname_i,
            f=imfile,
            line = line,
            iline = iline,
            ext = ext,
            # restfreq=restfreq,
            dist=dist, obname=obname, vsys=vsys,
            coordinate_center=coordinate_center,
            beam_convolution=beam_convolution, beam=beam,
            Tb=Tb, add_noise=add_noise, rms=rms,
            frame=frame, projection=projection)

os.chdir('..')
# -------------- End ----------------
