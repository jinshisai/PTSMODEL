import numpy as np
import matplotlib.pyplot as plt

import pyfigures as pyfg

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



class Inflow():
	'''

	Unit must be in cgs and radian.
	'''

	def __init__(self, r, mstar, rc, theta0, phi0, degree=True):
		self.mstar  = mstar
		self.rc     = rc
		self.theta0 = theta0
		self.phi0   = phi0

		self.r = r
		self.inflow(r, mstar, rc, theta0, phi0)

		self.cartesian()

	def inflow(self, r, mstar, rc, theta0, phi0):
		# assuming symmetry about the xy-plane to treat pi/2 < theta < pi
		# if pi/2 < theta0 < pi
		# ...

		sinth0 = np.sin(theta0)
		costh0 = np.cos(theta0)
		tanth0 = sinth0/costh0

		costh  = np.cos(theta0) * (1 - rc*np.sin(theta0)**2/r)
		sinth  = np.sqrt(1 - costh**2) # 0 < theta < 180., sin is always positive
		tanth  = sinth/costh

		cosdphi = tanth0/tanth
		dphi    = np.arccos(cosdphi) # 0 < dphi < pi
		phi     = phi0 + dphi

		vr     = -np.sqrt(Ggrav*mstar/r) * np.sqrt(1 + costh/np.cos(theta0))**(0.5)
		vtheta = np.sqrt(Ggrav*mstar/r) * (costh0 - costh)*np.sqrt((costh0+costh)/(costh0*sinth*sinth))
		vphi   = np.sqrt(Ggrav*mstar/r) *(sinth0/sinth)*np.sqrt(1.-costh/costh0)


		self.costh = costh
		self.sinth = sinth
		self.cosph = np.cos(phi)
		self.sinph = np.sin(phi)

		self.vr     = vr
		self.vtheta = vtheta
		self.vphi   = vphi

	def cartesian(self):
		self.x = self.r*self.sinth*self.cosph
		self.y = self.r*self.sinth*self.sinph
		self.z = self.r*self.costh

		self.vx = self.vr*self.sinth*self.cosph + self.vtheta*self.costh*self.cosph - self.vphi*self.sinph
		self.vy = self.vr*self.sinth*self.sinph + self.vtheta*self.costh*self.sinph + self.vphi*self.cosph
		self.vz = self.vr*self.costh - self.vtheta*self.sinth


	def rotate_coordinates(self, angle, axis='x', degree=True):
		'''
		Rotate coordinates around an axis.

		Parameters
		----------
		 - angle: Rotation angle.
		 - axis: Axis around which coordinates are rotated.
		'''

		if degree:
			angle *= np.pi/180.

		if axis == 'x':
			self.x_rot = self.x
			self.y_rot = self.y*np.cos(angle) + self.z*np.sin(angle)
			self.z_rot = -self.y*np.sin(angle) + self.z*np.cos(angle)

			self.vx_rot = self.vx
			self.vy_rot = self.vy*np.cos(angle) + self.vz*np.sin(angle)
			self.vz_rot = -self.vy*np.sin(angle) + self.vz*np.cos(angle)

	def project_to(self, plane='xy'):

		if 'x_rot' in self.__dict__:
			self.project_x = self.x_rot
			self.project_y = self.z_rot
			self.vproj     = self.vy_rot
			pass
		else:
			print ('No coordinate rotation is found.')
			print ('No rotation is adopted.')

			self.project_x = self.x
			self.project_y = self.z
			self.vproj = self.vy


	def rot_projectedmap(self, pa):
		'''
		pa: position angle (deg)
		'''

		if 'project_x' not in self.__dict__:
			print ('No projection yet.')
			return

		xrot, yrot = _rotate2d(self.project_x, self.project_y, pa, coords=True)
		self.project_x, self.project_y = xrot, yrot # update


def _rotate2d(x, y, angle, deg=True, coords=False):
	'''
	Rotate Cartesian coordinates.
	Right hand direction will be positive.

	array2d: input array
	angle: rotational angle [deg or radian]
	axis: rotatinal axis. (0,1,2) mean (x,y,z). Default x.
	deg (bool): If True, angle will be treated as in degree. If False, as in radian.
	'''

	# degree --> radian
	if deg:
		angle = np.radians(angle)
	else:
		pass

	if coords:
		angle = -angle
	else:
		pass

	cos = np.cos(angle)
	sin = np.sin(angle)

	xrot = x*cos - y*sin
	yrot = x*sin + y*cos

	return xrot, yrot


def main():
	# ------------ input -------------

	### Monte Carlo parameters
	nphot    = 10000000
	#nphot    = 1000000

	### Grid parameters !!! spherical coordiante !!!
	nr       = 64               # radius from center
	ntheta   = 64               # angle from z axis, 90 deg is disk plane
	nphi     = 64               # azimuthal
	#nr, ntheta, nphi = 32, 32, 32
	rmin     = 1000.*au           # minimum radius of the domein
	rmax     = 5000.*au         # maximum radius of the domein
	thmin, thmax = 0., np.pi    # from z axis


	### global properties
	mu         = 2.8    # Kauffman (2008)
	abunc18o   = 1.7e-7 # Frerking, Langer and Wilson (1982), for rho-Oph and Taurus
	gtod_ratio = 100.   # gas to dust mass ratio

	### Star parameters

	mstar = 1.6                                       # mass [Msun]
	mstar = mstar*ms                                  # Msun --> g
	lstar = 3.5*ls                                    # Steller Luminosity
	rstar = 4.*rs                                     # Stahler, Shu and Taam (1980), for protostars
	tstar = (lstar/(4*np.pi*rstar*rstar)/sigsb)**0.25 # Stefan-Boltzmann
	#tstar = 4000.                                     # temperature [K]
	pstar = np.array([0.,0.,0.])                      # position
	print ('Tstar: %.f K from Stefan-Boltzmann'%tstar)


	### Disk parameters
	mdisk     = 0.71e-2*ms                    # from L1489 ALMA C2 B6 obs, opc BW90
	plsig     = -0.5e0                        # Powerlaw of the surface density
	rd_in     = 1.*au                         # disk inner radius [cm]
	rd_out    = 600.*au                       # dust & gas disk outer radius [cm]
	Tdisk_1au = 400.                          # disk temperature at 1 au [K]
	cs_1au    = np.sqrt(kb*Tdisk_1au/(mu*mp)) # isothermal sound speed at 1 au [cm/s]
	Omg_1au   = np.sqrt(Ggrav*mstar/au**3)    # Keplerian angular velocity at 1 au[/s]
	hr0       = cs_1au/Omg_1au/au             # H/r at 1au, assuming hydrostatic equillibrium
	plh       = 0.25                          # Powerlaw of flaring h/r, assuming T prop r^-0.5 & Keplerian rotation
	prot_d    = 0.5                           # power-law of the disk rotation


	### envelope parameters
	rho0_e   = 1.4e-18  # ~1/10 of ~1.4e-17
	rho0_e   = 2.8e-17
	rcent    = rd_out   # centrifugal radius
	re_in    = rd_in    # envelope inner radius [cm]
	re_out   = 5000.*au # envelope outer radius [cm]
	theta0   = 91.*np.pi/180.      # deg
	phi0     = 325.*np.pi/180.     # deg
	#theta0   = 100*np.pi/180.
	#phi0     = 330*np.pi/180.
	#delth0   = 5.       # deg
	#delph0   = 5.       # deg


	### observations
	inc      = 73.       # [deg]
	#inc      = 90.
	angle_rot = inc - 90.  # deg
	#inc      = 180.-inc


	### projection
	#l_thick = 500

	# -------------------------------




	# ------------- main ------------
	ri = np.linspace(rmin, rmax, nr+1)
	r  = 0.5 * ( ri[0:nr] + ri[1:nr+1] )

	model = Inflow(r, mstar, rcent, theta0, phi0)

	# rotation & projection
	model.rotate_coordinates(angle_rot)
	model.project_to(plane='xy')

	plt_x = model.project_x/au
	plt_y = model.project_y/au
	plt_c = model.vproj*1e-5 # km s^-1


	# plot
	fig = plt.figure()
	ax  = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	#plt.scatter(rawmodel.project_x/au, rawmodel.project_y/au, c=rawmodel.nrho_g*rawmodel.vproj)

	ax.scatter(plt_x, plt_y, c=plt_c)
	ax.set_aspect(1)

	ax2.plot(plt_x, plt_c)
	ax2.set_xscale('log')
	ax2.set_yscale('log')

	pyfg.change_aspect_ratio(ax2, 1, plottype='loglog')

	fig.subplots_adjust(wspace=0.4)
	plt.show()




if __name__ == '__main__':
	main()