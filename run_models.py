# -------- modules -----------
import os
import sys
import numpy as np
import pandas as pd
import shutil

import ptsmodel
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
var_alpha = np.arange(0.2,1.2,0.2)
#var_inc   = np.array([60., 73., 90.])
var_inc   = np.array([73.])
var_alpha = np.arange(0.2, 0.4)
modeldir  = 'run_disk_envUl76_v6'
modelname = 'l1489_envU76_model'
# modellist = 'models.txt'


# Monte Carlo parameters
nphot    = 10000000
nphot    = 100000


# model name
dustopac  = 'nrmS03'
line      = 'c18o'


# Grid parameters !!! spherical coordiante !!!
nr       = 32               # radius from center
ntheta   = 32               # angle from z axis, 90 deg is disk plane
nphi     = 32               # azimuthal
rmin     = 1.*au            # minimum radius of the domein
rmax     = 4000.*au         # maximum radius of the domein
thetamin = 0.               # from z axis


# input parameter set
# Global properties
mu         = 2.8    # mean molecular weight. Kauffman (2008)
abunc18o   = 1.7e-7 # C18O abundance. Frerking, Langer and Wilson (1982), for rho-Oph and Taurus
gtod_ratio = 100.   # gas to dust mass ratio
dist       = 140.   # pc


# Stellar parameters
mstar = 1.6                                       # stellar mass [Msun]
mstar = mstar*ms                                  # Msun --> g
lstar = 3.5*ls                                    # Steller Luminosity
rstar = 4.*rs                                     # Stahler, Shu and Taam (1980), for protostars
pstar = np.array([0.,0.,0.])                      # position


# Disk parameters
# Mdust: total dust mass [Msun]
# Mdisk: total disk mass (gas mass) [Msun]
# plsig: power-law index of the surface density
# rin, rout: disk inner and outer radius [cm]

mdisk     = 1.8e-2*ms                             # from L1489 ALMA C2 B6 obs, opc Ossenkoph & Henning 94
plsig     = -0.5e0                                # power-law index of the surface density
rd_in     = 1.*au                                 # dust & gas disk inner radius [cm]
rd_out    = 600.*au                               # dust & gas disk outer radius [cm]
Tdisk_1au = 400.                                  # disk temperature at 1 au [K]
cs_1au    = np.sqrt(kb*Tdisk_1au/(mu*mp))         # isothermal sound speed at 1 au [cm/s]
Omg_1au   = np.sqrt(Ggrav*mstar/au**3)            # Keplerian angular velocity at 1 au[/s]
hr0       = cs_1au/Omg_1au/au                     # H/r at 1au, assuming hydrostatic equillibrium
plh       = 0.25                                  # Powerlaw of flaring h/r, assuming T prop r^-0.5 & Keplerian rotation


# Envelope parameters
# rho0_e: volume density of the envelope at rcent [g]
# rcent: centrifugal radius [cm]
# re_in, re_out: the inner and outer radii of the envelope [cm]

rho0_e = 1.6e-18  # ~1/10 of ~1.4e-17
vfac   = 1.       # reduction factor for v_infall from free-fall velocity
rcent  = rd_out   # centrifugal radius
re_in  = rd_in    # envelope inner radius [cm]
re_out = 2800.*au # envelope outer radius [cm]


# parameters for observations
# inc is now variable
#inc      = 73.                     # [deg]
#inc      = 180.-inc                # make lower side face us
restfreq = 219.56035e9             # rest frequency of C18O 2--1 [Hz]
sizeau   = 8000.                   # image size [au]
pa       = 54.-90.                 # position angle [deg]
vmin     = -5                      # minimum velocity to be imaged
vmax     = 5                       # maximum velocity to be imaged
vrange   = np.array([vmin,vmax])   # velocity range
nchan    = 70                      # channel number
iline    = 2                       # J=2--1
npix     = 1000                    # pixel size of image

# frequency --> wave length
freqrange      = restfreq*(1. - vrange*1.e5/clight)
lambdarange    = clight/freqrange       # in cm
lambdarange    = lambdarange*1.e-2*1.e6 # cm --> micron
lammin, lammax = lambdarange



# for fits output
obname            = 'L1489 IRS'                # name
outtxt            = '_c18o21'                  # output fits name
dist              = 140.                       # distance to object [pc]
coordinate_center = '4h4m43.07s +26d18m56.20s' # RA
vsys              = 7.22                       # km/s
beam_convolution  = True                       # convolve a beam?
beam              = [7.7, 6.4, -85.]           # bmaj, bmin, bpa
Tb                = False                      # unit Tb
add_noise         = False                      # add noise
rms               = 0.22                       # Jy/beam
frame             = 'fk5'                      # frame
projection        = 'SFL'                      # projection
# --------------------------------------------------




# --------------- start -----------------
# read model list
#models = pd.read_csv(modellist, delimiter=' ', comment='#')
#modeldirs, modelnames, var_inc, var_alpha = models.values.T
#nmodel = len(modeldirs)

# loop
print ('Start modeling')
nvar_alpha = len(var_alpha)
nvar_inc   = len(var_inc)
nmodel     = nvar_alpha*nvar_inc

# file recording models
fmd = open('models.txt', mode='w')
fmd.write('model_directory model_name var_alpha var_inc\n')
fmd.write('\n')

for i in range(nvar_alpha):
	for j in range(nvar_inc):
		# get variables
		alpha = var_alpha[i]
		inc   = var_inc[j]
		inc   = 180. - inc

		# model i
		print ('Now #%i/%i'%(i*nvar_inc+j+1, nmodel))
		dir_i       = modeldir + '_vr%02ivff_i%i'%(alpha*10, var_inc[j])
		modelname_i = modelname + '_vr%02ivff_i%i'%(alpha*10, var_inc[j])


		if j == 0:
			# make directory
			if os.path.exists(dir_i):
				print ('Directory %s already exists'%dir_i)
			else:
				print ('Make the directory %s'%dir_i)
				os.mkdir(dir_i)

			# record the first directory
			dir_0 = dir_i

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

			# envelope
			model_i.rho_envUl76(rho0_e, rcent, re_in=re_in, re_out=re_out)

			# disk
			model_i.rho_cutoffdisk(mdisk, rd_in, rd_out, plsig, hr0, plh, sig0=False)

			# density model
			model_i.rho_model(abunc18o, mu, gtod_ratio)

			# v-field
			model_i.vfield_model()
			vr         = model_i.vr
			model_i.vr = vr*alpha

			model_i.show_density()
			model_i.show_vfield(r_range=[100*au, 4000*au])
			#print (inspect.getmembers(test))
			model_i.export_to_radmc3d(nphot, dustopac, line)


			# RADMC-3D
			# calculate temperature profile
			print ('Calculate temperature profile')
			print ('radmc3d mctherm')
			os.system('radmc3d mctherm')
			model_i.plot_temperature(t_range=[0.,120.])
			# !!! finish !!!
		elif j >= 1:
			# copy directory
			if os.path.exists(dir_i):
				os.system('rm -r %s'%dir_i)
			shutil.copytree(dir_0, dir_i)

			# change directory
			os.chdir(dir_i)
			os.getcwd()
			print ('Now in %s'%dir_i)
		else:
			continue


		# solve radiative transfer
		run_radmc = 'radmc3d image npix %i phi 0 iline %i\
		 sizeau %.f lambdarange %.15e %.15e posang %.2f\
		  incl %.2f nlam %.0f'%(npix, iline, sizeau, lammin,lammax,pa,inc,nchan)
		print ('Solve radiative transfer')
		print (run_radmc)
		os.system(run_radmc)


		# save observational info.
		# out put infos
		with open('obsinfo.txt','w+') as f:
			f.write('# Information of observation to make image.out\n')          # comment 1
			f.write('# inclination\trestfrequency\timagesize\tposition angle\n') # comment 2
			f.write('# [deg]\t[Hz]\t[au]\t[deg]\n')                              # comment 3
			f.write('\n')
			f.write('%d %d %d %d'%(inc,restfreq,sizeau,pa))


		# export to fits
		outname = modelname_i + outtxt
		ptsmodel.export_radmc_tofits.export_radmc_tofits(outname, dist=dist, obname=obname, vsys=vsys, coordinate_center=coordinate_center,
			beam_convolution=beam_convolution, beam=beam, Tb=Tb, add_noise=add_noise, rms=rms,
			frame=frame, projection=projection)


		# write to file
		fmd.write('%s %s %.1f %.f\n'%(dir_i, modelname_i, alpha, inc))


		# end
		# back to main directory
		os.chdir('..')

fmd.close()