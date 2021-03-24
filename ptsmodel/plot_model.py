import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt



def vprofile(outtxt=True, xlim=[], ylim=[]):
	'''
	Draw a radial profile of the given velocity.
	'''
	# constants
	au    = 1.49598e13 # Astronomical Unit       [cm]
	Ggrav = 6.67428e-8 # gravitational constant  [dyn cm^2 g^-2]

	# reading file
	# grid info.
	f                = 'amr_grid.inp'
	nrtp             = np.genfromtxt(f, max_rows=1, skip_header=5, delimiter=' ',dtype=int)
	nr, ntheta, nphi = nrtp
	arraysize        = (nr,ntheta,nphi)

	dread            = pd.read_csv(f, skiprows=6, comment='#', encoding='utf-8',header=None)
	coords           = dread.values
	ri, thetai, phii = np.split(coords,[nr+1,nr+ntheta+2])

	# centers of each cell
	rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )                 # centers of each cell
	thetac   = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
	phic     = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )

	# get grid
	rr, tt, phph = np.meshgrid(rc,thetac,phic,indexing='ij') # (r, theta, phi) in the spherical coordinate
	zr       = 0.5*np.pi - tt # angle from z axis (90deg - theta)
	rxy      = rr*np.sin(tt)  # r in xy-plane
	zz       = rr*np.cos(tt)  # z in xyz coordinate


	# gas velocity [cm/s]
	f                = 'gas_velocity.inp'
	dread            = pd.read_csv(f, skiprows=2, comment='#', encoding='utf-8',header=None, delimiter=' ',parse_dates=True, keep_date_col=True, skipinitialspace=True)
	vrtp             = dread.values
	vr, vtheta, vphi = vrtp.T
	vr               = np.reshape(vr,arraysize,order='F')
	vtheta           = np.reshape(vtheta,arraysize,order='F')
	vphi             = np.reshape(vphi,arraysize,order='F')
	#print vr.shape


	# Mid plane
	vr_mid     = vr[:,-1,nphi//2]*1e-5     # km/s
	vphi_mid   = vphi[:,-1,nphi//2]*1e-5   # km/s
	vtheta_mid = vtheta[:,-1,nphi//2]*1e-5 # km/s
	r_mid      = rr[:,-1,nphi//2]/au       # au


	# Plot
	fig = plt.figure(figsize=(8.27, 8.27))
	ax  = fig.add_subplot(111)

	ax.set_xscale('log')
	ax.set_yscale('log')

	if len(xlim) == 2:
		ax.set_xlim(xlim[0], xlim[1])

	if len(ylim) == 2:
		ax.set_ylim(ylim[0], ylim[1])

	ax.set_xlabel('Radius (au)')
	ax.set_ylabel(r'Velocity ($\mathrm{km\ s^{-1}}$)')
	#ax.set_aspect(1)
	ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)


	# v profiles
	ax.plot(r_mid, -vr_mid, ls='--', color='k', lw=2., label=r'$v_\mathrm{r}$')         # vr
	ax.plot(r_mid, vtheta_mid, ls='-.', color='k', lw=2., label=r'$v_\mathrm{\theta}$') # vtheta
	ax.plot(r_mid, vphi_mid, ls='-', color='k', lw=2., label=r'$v_\mathrm{\phi}$')      # vphi

	ax.legend(frameon=False)
	plt.savefig('gas_vprofile.pdf', transparent=True)


	if outtxt:
		# Export
		with open ('gas_vprofile.txt', 'w+') as f:
			f.write('# r_mid -vr_mid vtheta_mid vphi_mid\n')
			f.write('# au km/s km/s km/s\n')
			np.savetxt(f, np.array([r_mid, -vr_mid, vtheta_mid, vphi_mid]).T)

	return ax