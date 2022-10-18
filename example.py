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




# -------------- constants -----------------
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




# ------------------- input ---------------------
# model name & variables
# give a list or set parameters & name here
var_ilines = np.array([2, 3, 6])
lam_cont   = np.array([1300., 890., 450.]) # micron
restfreqs  = np.array([219.5603541e9, 329.3305525e9, 658.5532782e9])
modeldir   = 'run_disk'
modelname  = 'l1527_diskmodel'
# modellist = 'models.txt'


# Monte Carlo parameters
nphot    = 10000000
nphot    = 100000
#iseed    = -5415*2 # Default value is -17933201
iseed    = -17933201


# model name
dustopac  = 'nrmS03'
line      = 'c18o'


# Grid parameters !!! spherical coordiante !!!
nr       = 32              # radius from center
ntheta   = 32               # angle from z axis, 90 deg is disk plane
nphi     = 1               # azimuthal
rmin     = 1.*au           # minimum radius of the domein
rmax     = 1000.*au          # maximum radius of the domein
thetamin = 0.               # from z axis


# input parameter set
# Global properties
mu         = 2.8    # mean molecular weight. Kauffman (2008)
abunc18o   = 1.7e-7 # C18O abundance. Frerking, Langer and Wilson (1982), for rho-Oph and Taurus
gtod_ratio = 100.   # gas to dust mass ratio
dist       = 140.   # pc


# Stellar parameters
mstar = 0.45                                      # stellar mass [Msun]
mstar = mstar*ms                                  # Msun --> g
lstar = 2.0*ls                                    # Steller Luminosity
rstar = 4.*rs                                     # Stahler, Shu and Taam (1980), for protostars
pstar = np.array([0.,0.,0.])                      # position


# Disk parameters
# Mdust: total dust mass [Msun]
# Mdisk: total disk mass (gas mass) [Msun]
# plsig: power-law index of the surface density
# rin, rout: disk inner and outer radius [cm]

mdisk     = 1.3e-2*ms                             # from L1489 ALMA C2 B6 obs, opc Ossenkoph & Henning 94
plsig     = -1.7                                  # power-law index of the surface density
rd_in     = 0.1*au                                # dust & gas disk inner radius [cm]
rd_out    = 80.*au                                # dust & gas disk outer radius [cm]
Tdisk_1au = 400.                                  # disk temperature at 1 au [K]
cs_1au    = np.sqrt(kb*Tdisk_1au/(mu*mp))         # isothermal sound speed at 1 au [cm/s]
Omg_1au   = np.sqrt(Ggrav*mstar/au**3)            # Keplerian angular velocity at 1 au[/s]
hr0       = cs_1au/Omg_1au/au                     # H/r at 1au, assuming hydrostatic equillibrium
plh       = 0.2                                   # Powerlaw of flaring h/r, assuming T prop r^-0.5 & Keplerian rotation


# Envelope parameters
# rho0_e: volume density of the envelope at rcent [g]
# rcent: centrifugal radius [cm]
# re_in, re_out: the inner and outer radii of the envelope [cm]

#rho0_e = 1.6e-18  # ~1/10 of ~1.4e-17
vfac   = 0.5      # reduction factor for v_infall from free-fall velocity
rcent  = rd_out   # centrifugal radius
re_in  = rd_in    # envelope inner radius [cm]
re_out = 2000.*au # envelope outer radius [cm]


# parameters for observations
inc      = 85.                     # [deg]
#inc      = 180.-inc                # make lower side face us
sizeau   = 500.                    # image size [au]
pa       = 90.                     # position angle [deg]
vmin     = -5                      # minimum velocity to be imaged
vmax     = 5                       # maximum velocity to be imaged
vrange   = np.array([vmin,vmax])   # velocity range
nchan    = 70                      # channel number
width_spw = 5                      # +/- 5 km/s
iline    = 2                       # J=2--1
npix     = 512                     # pixel size of image
# ilines are now variable

# frequency --> wave length
#freqrange      = restfreq*(1. - vrange*1.e5/clight)
#lambdarange    = clight/freqrange       # in cm
#lambdarange    = lambdarange*1.e-2*1.e6 # cm --> micron
#lammin, lammax = lambdarange



# for fits output
obname            = 'L1527 IRS'                # name
outtxt            = '_c18o'                    # output fits name
dist              = 140.                       # distance to object [pc]
coordinate_center = '4h39m53.88s +26d03m09.55s' # RA
vsys              = 5.8                        # km/s
beam_convolution  = True                       # convolve a beam?
beam              = [0.1, 0.1, 0.]             # bmaj, bmin, bpa
Tb                = False                      # unit Tb
add_noise         = False                      # add noise
rms               = 0.22                       # Jy/beam
frame             = 'fk5'                      # frame
projection        = 'SIN'                      # projection
# --------------------------------------------------




# --------------- start -----------------
# start modeling
dir_i       = modeldir
modelname_i = modelname
print ('Start modeling.')

# check directory
if os.path.exists(dir_i):
	print ('Directory %s already exists'%dir_i)
else:
	print ('Make the directory %s'%dir_i)
	os.mkdir(dir_i)


# copy files necessary
shutil.copy('./infiles/dustkappa_nrmS03.inp', dir_i)
shutil.copy('./infiles/molecule_c18o.inp', dir_i)

# change directory
os.chdir(dir_i)
os.getcwd()
print ('Now in %s'%dir_i)



# !!! modeling !!!
# grid
model_i = ptsmodel.PTSMODEL(modelname_i, nr, ntheta, nphi ,rmin, rmax,
 thetamin=0., thetamax=np.pi*0.5, phimin=0., phimax=2.*np.pi)
print ('Model name %s'%model_i.modelname)


# star
model_i.prop_star(mstar, lstar, rstar, pstar, tstar=None)

# disk
model_i.rho_cutoffdisk(mdisk, rd_in, rd_out, plsig, hr0, plh, sig0=False)
rho0_e = 0.5*model_i.rho_disk[model_i.r <= rd_out, -1, nphi//2][-1]
print ('%.2e'%rho0_e)

# envelope
model_i.rho_envUl76(rho0_e, rcent, re_in=re_in, re_out=re_out)

# outflow cavity
model_i.outflow_cavity(ang=45.)

# density model
model_i.rho_model(abunc18o, mu, gtod_ratio, disk_height=2)

# v-field
model_i.vfield_model()

# visualize
model_i.show_density()
model_i.show_vfield(r_range=[10*au, 100*au])
visualize.gas_density(model_i, outname='gas_density_zoom', nrho_range=[1e-1, 1e8],
	 xlim=[-100, 100], ylim=[-100, 100], rlim=[0,100], zlim=[0,100],
	 figsize=(11.69,8.27), cmap='coolwarm',
	 fontsize=14, wspace=0.4, hspace=0.2)
model_i.export_to_radmc3d(nphot, dustopac, line, iseed)



# RADMC-3D
# calculate temperature profile
print ('Calculate temperature profile')
print ('radmc3d mctherm')
os.system('radmc3d mctherm')
model_i.plot_temperature(t_range=[0.,120.])
# !!! finish !!!


### solve radiative transfer
_, weight, nlevels, EJ, gJ, J, ntrans, Jup, Acoeff, freq, delE =\
read_lamda_moldata('molecule_'+model_i.line+'.inp')
restfreq = freq[iline-1]*1e9 # rest frequency (Hz)
lam = clight*1e-2/restfreq*1e6 # micron

# solve radiative transfer
# line
run_radmc = 'radmc3d image npix %i phi 0 iline %i\
 sizeau %.f widthkms %.2f linenlam %i posang %.2f\
 incl %.2f'%(npix, iline, sizeau, width_spw, nchan, pa, inc)
print ('Solve radiative transfer')
print (run_radmc)
os.system(run_radmc)

with open('obsinfo.txt','w+') as f:
	f.write('# Information of observation to make image.out\n')          # comment 1
	f.write('# inclination\trestfrequency\timagesize\tposition angle\n') # comment 2
	f.write('# [deg]\t[Hz]\t[au]\t[deg]\n')                              # comment 3
	f.write('\n')
	f.write('%d %d %d %d'%(inc,restfreq,sizeau,pa))

# export to fits
outname = modelname_i + outtxt + '%i%i'%(iline, iline-1)
ptsmodel.export_radmc_tofits.export_radmc_tofits(outname, restfreq=restfreq, dist=dist, obname=obname, vsys=vsys, coordinate_center=coordinate_center,
	beam_convolution=beam_convolution, beam=beam, Tb=Tb, add_noise=add_noise, rms=rms,
	frame=frame, projection=projection)
os.system('cp image.out image_line_%s%i%i.out'%(line,iline, iline-1))


# continuum
run_radmc = 'radmc3d image npix %i phi 0\
 sizeau %.f posang %.2f incl %.2f lambda %.13e'%(npix, sizeau, pa, inc, lam)
print ('Solve radiative transfer for continuum')
print (run_radmc)
os.system(run_radmc)


# export to fits
outname = modelname_i + outtxt + '%i%i'%(iline, iline-1) + '_cont'
ptsmodel.export_radmc_tofits.export_radmc_tofits(outname, dist=dist, obname=obname, vsys=vsys, coordinate_center=coordinate_center,
	beam_convolution=beam_convolution, beam=beam, Tb=Tb, add_noise=add_noise, rms=rms,
	frame=frame, projection=projection, restfreq=restfreq)
os.system('cp image.out image_cont_%s%i%i.out'%(line,iline, iline-1))
os.chdir('..')