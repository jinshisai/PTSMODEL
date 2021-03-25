# -*- coding: utf-8 -*-
'''
Made and developed by J. Sai.

email: jn.insa.sai@gmail.com
'''

# modules
import numpy as np
import time
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')



# constants

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




# ProToStellar MODEL
class PTSMODEL():
    '''
    Build a model of a protostar, a disk, and an envelope system.

    '''

    def __init__(self, modelname, nr, ntheta, nphi ,rmin, rmax,
     thetamin=0., thetamax=np.pi*0.5, phimin=0., phimax=2.*np.pi):
        self.modelname = modelname

        # makegrid
        self.nr     = nr
        self.ntheta = ntheta
        self.nphi   = nphi
        self.makegrid(nr, ntheta, nphi ,rmin, rmax,
            thetamin=thetamin, thetamax=thetamax, phimin=phimin, phimax=phimax)

        # initialize
        self.rho_disk = np.array([])
        self.rho_env  = np.array([])
        self.rho_g    = np.array([])
        self.turbulence = False


    def makegrid(self, nr, ntheta, nphi ,rmin, rmax,
     thetamin=0., thetamax=np.pi*0.5, phimin=0., phimax=2.*np.pi, logr=True):
        '''
        Make a grid for model calculation. Currently, only polar coodinates are allowed.

        Args:
            nr, ntheta, nphi: number of grid
            xxxmin, xxxmax: minimum and maximum values of axes
        '''
        # boundary of cells
        if logr:
            ri       = np.logspace(np.log10(rmin),np.log10(rmax),nr+1) # radius, log scale
        else:
            ri       = np.linspace(rmin,rmax,nr+1)                    # radius, linear scale
        thetai   = np.linspace(thetamin,thetamax,ntheta+1)       # from thetaup to 0.5*pi, ntheta cells
        phii     = np.linspace(phimin,phimax,nphi+1)             # from 0 to 2pi, nphi cells

        # for RADMC-3D
        self.ri     = ri
        self.thetai = thetai
        self.phii   = phii

        # centers of each cell
        rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )
        thetac   = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
        phic     = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )

        # make a grid
        qq = np.meshgrid(rc,thetac,phic,indexing='ij')

        # in spherical coordinates
        rr   = qq[0]              # r in spherical coordinate
        tt   = qq[1]              # theta in spherical coordinate
        phph = qq[2]              # phi in spherical coordinate

        # cylindrical coordinates (for disk)
        rxy  = rr*np.sin(tt)      # radius in xy-plane, r*sin(theta)
        zz   = rr*np.cos(tt)      # z, r*cos(theta)

        # save
        self.gridshape = rr.shape
        self.grid      = qq
        self.rr        = rr
        self.tt        = tt
        self.phph      = phph
        self.rxy       = rxy
        self.zz        = zz

        return


    # density distributions
    # cut-off disk
    def rho_cutoffdisk(self, mdisk, rin, rout, plsig, hr0, plh, sig0=False):
        '''
        Calculate density distribution of a cut-off disk.

        Args:
            mdisk: Disk mass (g). If sig0=True, mdisk will be treated
             as sigma_0, column density at 1 au.
            rin, rout:Disk inner and outer radii (cm)
            plsig: Power-law index of the radial profile of surface density.
            hr0: H/r at 1au, where H is the scale height.
            plh: Power-law index of the radial profile of h/r, which determines disk-flaring.
             It should be 0.25 if you assume T prop r^-0.5 & Keplerian rotation.
            sig0 (bool): If False, mdisk will be treated as Mdisk.
             If True, mdisk will be treated as sigma_0, column density at 1 au, and
              disk density profile is thus calculated from Sigma_0 rather than Mdisk.
        '''
        # grid
        rxy = self.rxy
        zz  = self.zz

        self.rdisk_in  = rin
        self.rdisk_out = rout

        # surface density profile
        # sig0, column density at 1 au
        if sig0:
            self.sig0 = mdisk
        else:
            sig0 = PTSMODEL.sig0_disk(mdisk, rin, rout, plsig)

        sigma = sig0 * (rxy/au)**plsig # surface density as a function of r

        # scale height
        hhr       = hr0 * (rxy/au)**plh # h/r at r, where h is the scale hight
        hh        = hhr * rxy           # the scale hight at r
        self.hh   = hh


        # make rho
        rho = np.zeros(self.gridshape)

        # rin < rdisk < rout
        sigma[np.where(rxy < rin)]  = 0.
        sigma[np.where(rxy > rout)] = 0.

        exp = np.exp(-zz*zz/(2.*hh*hh))
        rho = sigma*exp/(np.sqrt(2.*np.pi)*hh) # [g/cm^3]

        self.rho_disk = rho


    # Column density at 1 au
    def sig0_disk(mdisk, rin, rout, p):
        '''
        Calculate Sigma 0, column density of the disk at 1 au, from Mdisk.
        Disk is a cut-off disk.
        Mdisk = 2 pi int_rin^rout Sig0 (r/1 au)^(-p) dr

        Args:
            mdisk: disk mass (g)
            rin, rout: inner and outer radius of disk (cm)
            p: power-law index of the radial profile of surface density

        Return:
            sig0: Sigma_0, column density at 1 au.
        '''
        # Analytical solution
        # dxdy = r drdphi
        rint = ((1./(p+2.))*au**(-p))*(rout**(p+2.) - rin**(p+2.))\
         if p != -2. else (au**(-p))*(np.log(rout) - np.log(rin))
        sig0 = mdisk/(2.*np.pi*rint) # (g/cm^2)
        #print (rint, Sig0)

        return sig0


    # Density distribution of the envelope (Ulrich76)
    # Derive theta0 and phi0
    def der_th0ph0(r,theta,rc):
        '''
        Derive theta0 and phi0 from given r, theta, phi.
        See Ulrich (1976) for the detail. Here 0 < theta < pi/2.
        To get solutions, see also Mendoza (2004)
        '''
        costh = np.cos(theta)

        # solve costh0^3 + (-1 + r/rc) costh0 - costh*r/rc = 0
        # Mendoza (2004)
        costh0      = np.zeros(r.shape)
        term_rcos_2 = r/rc*costh/2.
        term_1r3    = (1.-r/rc)/3.

        # for r = rc
        if len(np.where(r == rc)[0]) >= 1:
            costh0[np.where(r == rc)] = costh[np.where(r == rc)]**(1./3.)

        # for r > rc
        if len(np.where(r > rc)[0]) >= 1:
            term_sinh = np.sinh(1./3.*np.arcsinh(term_rcos_2[np.where(r > rc)]/(-term_1r3[np.where(r > rc)])**(3./2.)))
            costh0[np.where(r > rc)] = 2.*(-term_1r3[np.where(r > rc)])**0.5*term_sinh

        # for r < rc
        case        = term_rcos_2**2. - term_1r3**3.
        if len(np.where((r < rc) & (case > 0.))[0]) >= 1:
            term_cosh   = np.cosh(1./3.*np.arccosh(term_rcos_2[np.where((r < rc) & (case > 0.))]\
                /(term_1r3[np.where((r < rc) & (case > 0.))])**(3./2.)))
            costh0[np.where((r < rc) & (case > 0.))] =\
             2.*(term_1r3[np.where((r < rc) & (case > 0.))])**0.5*term_cosh


        if len(np.where((r < rc) & (case < 0.))[0]) >= 1:
            term_cos    = np.cos(1./3.*np.arccos(term_rcos_2[np.where((r < rc) & (case < 0.))]\
                /(term_1r3[np.where((r < rc) & (case < 0.))])**(3./2.)))
            costh0[np.where((r < rc) & (case < 0.))] =\
             2.*(term_1r3[np.where((r < rc) & (case < 0.))])**0.5*term_cos

        return costh0


    # Density distribution
    def rho_envUl76(self, rho0, rc, re_in=None, re_out=None):
        '''
        Calculate density distribution of the infalling envelope model proposed in Ulrich (1976).
        See Ulrich (1976) for the detail.

        Args:
            rc: centrifugal radius (cm)
            rho0: density at rc (g cm^-3)
            re_in, re_out: inner and outer radii of an envelope (cm)
        '''
        self.rho_e0 = rho0
        self.rc     = rc
        self.re_in  = re_in
        self.re_out = re_out

        r     = self.rr
        theta = self.tt
        costh = np.cos(theta)

        # solve costh0^3 + (-1 + r/rc) costh0 - costh*r/rc = 0
        costh0 = PTSMODEL.der_th0ph0(r,theta,rc)
        sinth0 = np.sqrt(1.-costh0*costh0) # 0 < theta < pi/2, don't use here though
        self.costh0 = costh0
        self.sinth0 = sinth0

        rho = rho0*(r/rc)**(-1.5)*(1.+costh/costh0)**(-0.5)\
         *(costh/(2.*costh0)+rc/r*costh0*costh0)**(-1.)

        if re_in:
            rho[np.where(r < re_in)] = 0.

        if re_out:
            rho[np.where(r > re_out)] = 0.

        self.rho_env = rho


    def rho_envsheet(self, rho0, rc, eta, re_in=None, re_out=None):
        '''
        Calculate density distribution of the flattened, infalling envelope model,
         modifying the envelope model proposed in Ulrich (1976)
          following Momose et al. (1998) as following,

        rho_sheet(r,theta) = rho_u76(r, theta) x sech^2(eta cos0)(eta/tanh(eta)),
         where eta = r0/H, r0 is the initial radius at theta=theta0 and H is the scale height.

        eta=0 and infinity means no modulation and 2D thin disk, respectively.
        r0 could be considered as the outer edge of the envelope. Then, eta is simply like the ratio of
        the envelope major and minor axes. Momose+98 argue that eta=2 is the best fit to a Class I source.

        Reference:
         - Ulrich (1976), Hartman et al. (1996), Momose et al. (1998)

        Args:
         - rho0: density at rc (any unit)
         - rc: centrifugal radius (any unit)
         - eta: Factor characterizeing the degree of modulation on
         the original density distribution. It is defined as rc/H,
          where H is the scale height of the sheet. eta=0 means no modulation.
           eat getting infinity, the envelope becomes infinitesimally thin.
        '''
        PTSMODEL.rho_envUl76(self, rho0, rc, re_in=None, re_out=None)
        rho_u76 = self.rho_env
        costh0  = self.costh0
        sinth0  = self.sinth0

        cosh2   = np.cosh(eta*costh0)**2.
        sech2   = 1./cosh2

        rho_sheet = rho_u76 * sech2*(eta/np.tanh(eta))
        self.rho_env = rho_sheet


    def rho_model(self, Xconv, mu=2.8, gtod_ratio = 100.):
        '''
        Make a density model

        Args:
            gtod_ratio: gas-to-dust mass ratio
            mu: mean molecular weight
            Xconv: Abandance of a molecule you want to model.
        '''
        rho = np.zeros(self.gridshape)

        # Disk
        if self.rho_disk.size:
            where_disk = np.where((self.rxy <= self.rdisk_out) & (np.abs(self.zz) <= self.hh))

        # Envelope
        if self.rho_env.size:
            rho = self.rho_env # with envelope
        else:
            where_disk =  (array([]),) # only disk

        # density
        if len(where_disk[0]):
            rho[where_disk] = self.rho_disk[where_disk] # disk + envelope
        else:
            rho = self.rho_disk # only disk

        # density of the total gas (H2 gas) & dust
        rho_h2 = rho
        rho_d  = rho_h2/gtod_ratio

        # density of the target molecule
        rho_g  = rho_h2*Xconv
        nrho_g = rho_g/(mu*mp)

        # save
        self.rho_H2 = rho_h2
        self.rho_d  = rho/gtod_ratio
        self.rho_g  = rho_g
        self.nrho_g = nrho_g
        self.where_disk = where_disk


    # Velocity distributions
    def vfield_model(self):
        rr, tt, phph = self.grid
        rxy          = self.rxy

        # initialize
        vr     = np.zeros(self.gridshape)
        vtheta = np.zeros(self.gridshape)
        vphi   = np.zeros(self.gridshape)

        # read density
        if self.rho_g.size:
            pass
        else:
            print ('ERROR: No density distribution. Make density distribution before calculating v-field.')
            return

        if self.rho_env.size:
            # with envelope
            vr_env, vtheta_env, vphi_env = PTSMODEL.vinf_Ul76(rr, tt, self.mstar, self.rc)
            vr     = vr_env
            vtheta = vtheta_env
            vphi   = vphi_env

        if self.rho_disk.size:
            vr_disk, vtheta_disk, vphi_disk = PTSMODEL.v_steadydisk(rxy, self.mstar)
            where_disk = self.where_disk
            if len(where_disk[0]):
                vr[where_disk]     = vr_disk[where_disk]
                vtheta[where_disk] = vtheta_disk[where_disk]
                vphi[where_disk]   = vphi_disk[where_disk]
            else:
                vr     = vr_disk
                vtheta = vtheta_disk
                vphi   = vphi_disk

        self.vr     = vr
        self.vtheta = vtheta
        self.vphi   = vphi



    # Keplerian velocity
    def Vkep(radius, Mstar):
        '''
        Calculate Keplerian velocity in cgs unit.

        radius: radius [cm]
        Mstar: central stellar mass [g]
        '''
        Vkep = np.sqrt(Ggrav*Mstar/radius)
        return Vkep


    # Vrotation
    def vrot_plaw(radius, v0, r0, p):
        '''
        Calculate Keplerian velocity in cgs unit.

        radius: radius [cm]
        Mstar: central stellar mass [g]
        '''
        vrot = v0*(radius/r0)**(-p)
        return vrot


    # Velocity field of a disk
    def v_steadydisk(rxy, mstar):
        # v-field of the disk
        vr     = np.zeros(rxy.shape)
        vtheta = np.zeros(rxy.shape)
        vphi   = PTSMODEL.Vkep(rxy, mstar)   # Keplerian rotation

        return vr, vtheta, vphi


    # Infall velocity
    def vinf_Ul76(r, theta, mstar, rc):
        '''
        Ulrich (1976) infall model

        Args
        Mstar: central star mass[g]
        theta0, phi0: initial position in polar coorinate [rad]
        theta, phi, r: current poistion in polar coordinate [rad], [au]
        '''
        # coordinates
        costh  = np.cos(theta)
        sinth  = np.sin(theta)
        cgrav  = np.sqrt(Ggrav*mstar/r)

        # solve costh0^3 + (-1 + r/rc) costh0 - costh*r/rc = 0
        costh0 = PTSMODEL.der_th0ph0(r,theta,rc)
        sinth0 = np.sqrt(1.-costh0*costh0) # 0 < theta < pi/2

        vr     = - cgrav*np.sqrt(1+(costh/costh0))
        vtheta = cgrav*(costh0 - costh)\
            *np.sqrt((costh0+costh)/(costh0*sinth*sinth))
        vphi = cgrav*(sinth0/sinth)\
            *np.sqrt(1.-costh/costh0)

        return vr, vtheta, vphi

    # Turbulence
    def vturbulence(self, values):
        '''
        Add turbulence.

        Args:
         - values (float or numpy.ndarray): Velcity dispersion of turbulence, sigma_v (km/s).
            Not a line width (FWHM)! Relations between a velocity dispersion and line width is
            Delta_v = sqrt(8 x ln(2)) x sigma_v.
        '''
        if isinstance(values, float):
            vturb = np.full(self.gridshape, values)*1e5 # km/s --> cm/s
            self.vturb      = vturb
        elif isinstance(values, np.ndarray):
            self.vturb      = values*1e5     # km/s --> cm/s
        else:
            print ('ERROR:\tvturblence: Input values must be float or np.ndarray object.')
            return

        self.turbulence = True


    # 3D rotation
    def rotate3d(array3d, angle, axis=0, deg=True, coords=False):
        '''
        Rotate Cartesian coordinate or (1,3) array.
        Right hand direction will be positive.

        array3d: input array
        angle: rotational angle [deg or radian]
        axis: rotatinal axis. (0,1,2) mean (x,y,z). Default x.
        deg (bool): If True, angle will be treated as in degree. If False, as in radian.
        '''

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

        Rx = np.array([[1.,0.,0.],
                      [0.,cos,-sin],
                      [0.,sin,cos]])

        Ry = np.array([[cos,0.,sin],
                      [0.,1.,0.],
                      [-sin,0.,cos]])

        Rz = np.array([[cos,-sin,0.],
                      [sin,cos,0.],
                      [0.,0.,1.]])

        if axis == 0:
            # rotate around x axis
            newarray = np.dot(Rx,array3d)

        elif axis == 1:
            # rotate around y axis
            newarray = np.dot(Ry,array3d)

        elif axis == 2:
            # rotate around z axis
            newarray = np.dot(Rz,array3d)

        else:
            print ('ERROR\trotate3d: axis value is not suitable.\
             Choose value from (0,1,2). (0,1,2) mean rotatinal axes, (x,y,z) respectively.')

        return newarray


    # Star
    def prop_star(self, mstar, lstar, rstar, pstar, tstar=None):
        '''
        Stellar properties
        '''
        self.mstar = mstar
        self.lstar = lstar
        self.rstar = rstar
        self.pstar = pstar

        if tstar:
            self.tstar = tstar
        else:
            sigsb = 5.670367e-5     # Stefan-Boltzmann constant [erg s^-1 cm^-2 K^-4]
            tstar = (lstar/(4*np.pi*rstar*rstar)/sigsb)**0.25 # Stefan-Boltzmann
            self.tstar = tstar


    # Calculate temperature of a star
    def calc_tstar(self, lstar, rstar=4.*rs):
        '''
        Calculate stellar temperature from Stefan-Boltzmann.

        Args:
            lstar: Stellar luminosity (in cgs)
            rstar: Stellar radius (cm).
             rstar=4xR_sun for protostars (Stahler, Shu and Taam 1980)
        '''
        sigsb = 5.670367e-5     # Stefan-Boltzmann constant [erg s^-1 cm^-2 K^-4]
        tstar = (lstar/(4*np.pi*rstar*rstar)/sigsb)**0.25 # Stefan-Boltzmann
        self.tstar = tstar


    # Make input files for RADMC-3D
    def export_to_radmc3d(self, nphot, dustopac='nrmS03',line='c18o'):
        nr, ntheta, nphi = self.gridshape
        ############### Output the model into radmc3d files ############
        # Write the wavelength_micron.inp file
        #
        lam1     = 0.1e0
        lam2     = 7.0e0
        lam3     = 25.e0
        lam4     = 1.0e4
        n12      = 20
        n23      = 100
        n34      = 30
        lam12    = np.logspace(np.log10(lam1),np.log10(lam2),n12,endpoint=False)
        lam23    = np.logspace(np.log10(lam2),np.log10(lam3),n23,endpoint=False)
        lam34    = np.logspace(np.log10(lam3),np.log10(lam4),n34,endpoint=True)
        lam      = np.concatenate([lam12,lam23,lam34])
        nlam     = lam.size
        #
        # Write the wavelength file
        #
        with open('wavelength_micron.inp','w+') as f:
            f.write('%d\n'%(nlam))
            np.savetxt(f,lam.T,fmt=['%13.6e']) # .T command produce transposed matrix
        #
        #
        # Write the stars.inp file
        #
        with open('stars.inp','w+') as f:
            f.write('2\n')
            f.write('1 %d\n\n'%(nlam))
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(self.rstar,self.mstar,self.pstar[0],self.pstar[1],self.pstar[2]))
            np.savetxt(f,lam.T,fmt=['%13.6e'])
            f.write('\n%13.6e\n'%(-self.tstar))
        #
        # Write the grid file
        #
        with open('amr_grid.inp','w+') as f:
            f.write('1\n')                         # iformat
            f.write('0\n')                         # AMR grid style  (0=regular grid, no AMR)
            f.write('100\n')                       # Coordinate system: spherical
            f.write('0\n')                         # gridinfo
            f.write('1 1 1\n')                     # Include r,theta coordinates
            f.write('%d %d %d\n'%(nr,ntheta,nphi)) # Size of grid
            np.savetxt(f,self.ri.T,fmt=['%21.14e'])     # R coordinates (cell walls)
            np.savetxt(f,self.thetai.T,fmt=['%21.14e']) # Theta coordinates (cell walls)
            np.savetxt(f,self.phii.T,fmt=['%21.14e'])   # Phi coordinates (cell walls)
        #
        # Write the density file
        #
        with open('dust_density.inp','w+') as f:
            f.write('1\n')                                  # Format number
            f.write('%d\n'%(nr*ntheta*nphi))                # Nr of cells
            f.write('1\n')                                  # Nr of dust species
            data = self.rho_d.ravel(order='F')              # Create a 1-D view, fortran-style indexing
            np.savetxt(f,data.T,fmt=['%13.6e'])             # The data
        #
        # Dust opacity control file
        #
        with open('dustopac.inp','w+') as f:
            f.write('2               Format number of this file\n')
            f.write('1               Nr of dust species\n')
            f.write('============================================================================\n')
            f.write('1               Way in which this dust species is read\n')
            f.write('0               0=Thermal grain\n')
            f.write('%s          Extension of name of dustkappa_***.inp file\n'%dustopac)
            f.write('----------------------------------------------------------------------------\n')
        #
        # Write the molecule number density file.
        #
        with open('numberdens_%s.inp'%line,'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
            data = self.nrho_g.ravel(order='F')  # Create a 1-D view, fortran-style indexing
            np.savetxt(f,data.T,fmt=['%13.6e'])
        # Write the gas velocity field
        #
        with open('gas_velocity.inp','w+') as f:
            f.write('1\n')                        # Format number
            f.write('%d\n'%(nr*ntheta*nphi))      # Nr of cells
            wgv = [[[f.write('%13.6e %13.6e %13.6e\n'%(self.vr[ir,itheta,iphi],self.vtheta[ir,itheta,iphi],self.vphi[ir,itheta,iphi]))
            for ir in range(nr) ] for itheta in range(ntheta)] for iphi in range(nphi)]
            '''
            for iphi in range(nphi):
                for itheta in range(ntheta):
                    for ir in range(nr):
                        f.write('%13.6e %13.6e %13.6e\n'%(self.vr[ir,itheta,iphi],self.vtheta[ir,itheta,iphi],self.vphi[ir,itheta,iphi]))
            '''
        #
        # Write the microturbulence file
        #
        if self.turbulence:
            with open('microturbulence.inp','w+') as f:
                f.write('1\n')                       # Format number
                f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
                data = self.vturb.ravel(order='F')   # Create a 1-D view, fortran-style indexing
                np.savetxt(f,data.T,fmt=['%13.6e'])
        # Write the lines.inp control file
        #
        with open('lines.inp','w') as f:
            f.write('1\n')
            f.write('1\n')
            f.write('%s    leiden    0    0\n'%line)
        #
        # Write the radmc3d.inp control file
        #
        with open('radmc3d.inp','w+') as f:
            f.write('nphot = %d\n'%(nphot))
            f.write('scattering_mode_max = 0\n')
            f.write('iranfreqmode = 1\n')
            f.write('tgas_eq_tdust = 1')
        ####################### End output ########################


    # plot for check
    # plot density
    def show_density(self, rho_d_range=[], nrho_g_range=[],
        figsize=(11.69,8.27), cmap='coolwarm', fontsize=14, wspace=0.4, hspace=0.2):
        '''
        Visualize density distribution as 2-D slices.

        Args:
            rho_d_range:
            nrho_g_range:
        '''
        # modules
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib import pyplot as plt
        from matplotlib import cm
        import matplotlib.colors as colors
        import mpl_toolkits.axes_grid1

        # setting for figures
        #plt.rcParams['font.family'] ='Arial'    # font (Times New Roman, Helvetica, Arial)
        plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
        plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
        plt.rcParams['font.size'] = fontsize    # fontsize

        # read model
        # dimension
        nr, ntheta, nphi = self.gridshape

        # edge of cells
        #  cuz the plot method, pcolormesh, requires the edge of each cell
        ri     = self.ri
        thetai = self.thetai
        phii   = self.phii

        rr, tt, phph = np.meshgrid(ri, thetai, phii, indexing='ij')
        rxy = rr*np.sin(tt)      # radius in xy-plane, r*sin(theta)
        zz  = rr*np.cos(tt)      # z, r*cos(theta)
        #rr     = self.rr
        #phph   = self.phph
        #rxy    = self.rxy
        #zz     = self.zz

        rho_d  = self.rho_d
        nrho_g = self.nrho_g

        xx = rxy*np.cos(phph)
        yy = rxy*np.sin(phph)

        # parameters
        if len(rho_d_range) == 0:
            rho_d_max = np.nanmax(rho_d)
            rho_d_min = rho_d_max*1e-7  # g cm^-3, dynamic range of an order of five
        elif len(rho_d_range) == 2:
            rho_d_min, rho_d_max = rho_d_range
        else:
            print ('ERROR: rho_d_range must be given as [min value, max value].')
            rho_d_max = np.nanmax(rho_d)
            rho_d_min = rho_d_max*1e-7  # g cm^-3, dynamic range of an order of five

        if len(nrho_g_range) == 0:
            nrho_g_max = np.nanmax(nrho_g)
            nrho_g_min = nrho_g_max*1e-7 # dynamic range of an order of five
        elif len(nrho_g_range) == 2:
            nrho_g_min, nrho_g_max = nrho_g_range
        else:
            print ('ERROR: nrho_g_range must be given as [min value, max value].')
            nrho_g_max = np.nanmax(nrho_g)
            nrho_g_min = nrho_g_max*1e-7 # dynamic range of an order of five


        # dust disk
        fig1 = plt.figure(figsize=figsize)

        # plot 1; density in r vs z
        ax1     = fig1.add_subplot(121)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax1)
        cax1    = divider.append_axes('right', '3%', pad='0%')

        im1   = ax1.pcolormesh(rxy[:,:,nphi//2]/au, zz[:,:,nphi//2]/au, rho_d[:,:,nphi//2], cmap=cmap,
         norm = colors.LogNorm(vmin = rho_d_min, vmax=rho_d_max), rasterized=True)
        cbar1 = fig1.colorbar(im1, cax=cax1)

        ax1.set_xlabel('radius (au)')
        ax1.set_ylabel('z (au)')
        #cbar1.set_label(r'$\rho_\mathrm{dust}\ \mathrm{(g\ cm^{-3})}$')
        ax1.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
        ax1.set_aspect(1)


        # plot 2; density in r vs phi (xy-plane)
        ax2     = fig1.add_subplot(122)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax2)
        cax2    = divider.append_axes('right', '3%', pad='0%')

        im2   = ax2.pcolormesh(xx[:,-1,:]/au, yy[:,-1,:]/au, rho_d[:,-1,:],
         cmap=cmap, norm = colors.LogNorm(vmin = rho_d_min, vmax=rho_d_max), rasterized=True)
        #ax2.scatter(xx[:,-1,:]/au, yy[:,-1,:]/au, marker='x', s=5., color='k')
        cbar2 = fig1.colorbar(im2,cax=cax2)

        ax2.set_xlabel('x (au)')
        ax2.set_ylabel('y (au)')
        cbar2.set_label(r'$\rho_\mathrm{dust}\ \mathrm{(g\ cm^{-3})}$')
        ax2.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
        ax2.set_aspect(1)



        # gas disk
        fig2 = plt.figure(figsize=figsize)

        # plot 1; gas number density in r vs z
        ax3     = fig2.add_subplot(121)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax3)
        cax3    = divider.append_axes('right', '3%', pad='0%')

        im3   = ax3.pcolormesh(rxy[:,:,nphi//2]/au, zz[:,:,nphi//2]/au, nrho_g[:,:,nphi//2],
         cmap=cmap, norm = colors.LogNorm(vmin = nrho_g_min, vmax=nrho_g_max), rasterized=True)
        cbar3 = fig2.colorbar(im3,cax=cax3)

        ax3.set_xlabel('radius (au)')
        ax3.set_ylabel('z (au)')
        #cbar3.set_label(r'$n_\mathrm{gas}\ \mathrm{(cm^{-3})}$')
        ax3.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
        ax3.set_aspect(1)


        # plot 2; density in r vs phi (xy-plane)
        ax4     = fig2.add_subplot(122)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax4)
        cax4    = divider.append_axes('right', '3%', pad='0%')


        im4   = ax4.pcolormesh(xx[:,-1,:]/au, yy[:,-1,:]/au, nrho_g[:,-1,:], cmap=cmap,
         norm = colors.LogNorm(vmin = nrho_g_min, vmax=nrho_g_max), rasterized=True)
        #im4   = ax4.pcolor(rr[:,-1,:]/au, phph[:,-1,:], nrho_gas[:,-1,:], cmap=cm.coolwarm, norm = colors.LogNorm(vmin = 10., vmax=1.e4))

        cbar4 = fig2.colorbar(im4,cax=cax4)
        ax4.set_xlabel('x (au)')
        ax4.set_ylabel('y (au)')
        cbar4.set_label(r'$n_\mathrm{gas}\ \mathrm{(cm^{-3})}$')
        ax4.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
        ax4.set_aspect(1)


        # save figures
        fig1.subplots_adjust(wspace=wspace, hspace=hspace)
        fig1.savefig('dust_density_dist.pdf',transparent=True)

        fig2.subplots_adjust(wspace=wspace, hspace=hspace)
        fig2.savefig('gas_density_dist.pdf',transparent=True)
        plt.close()


    # plot velocity field
    def show_vfield(self, nrho_g_range=[], r_range=[], step=1,
     figsize=(11.69,8.27), vscale=3e2, width=10. ,cmap='coolwarm',
      fontsize=14, wspace=0.4, hspace=0.2):
        '''
        Visualize density distribution as 2-D slices.

        Args:
            rho_d_range:
            nrho_g_range:
        '''
        # modules
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib import pyplot as plt
        from matplotlib import cm
        import matplotlib.colors as colors
        import mpl_toolkits.axes_grid1

        # setting for figures
        #plt.rcParams['font.family']     = 'Arial'   # font (Times New Roman, Helvetica, Arial)
        plt.rcParams['xtick.direction'] = 'in'      # directions of x ticks ('in'), ('out') or ('inout')
        plt.rcParams['ytick.direction'] = 'in'      # directions of y ticks ('in'), ('out') or ('inout')
        plt.rcParams['font.size']       = fontsize  # fontsize

        # function for binning
        def binArray(data, axis, binstep, binsize, func=np.nanmean):
            data = np.array(data)
            dims = np.array(data.shape)
            argdims = np.arange(data.ndim)
            argdims[0], argdims[axis]= argdims[axis], argdims[0]
            data = data.transpose(argdims)
            data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
            data = np.array(data).transpose(argdims)
            return data

        # read model
        nr, ntheta, nphi = self.gridshape

        # edge of cells
        #  cuz the plot method, pcolormesh, requires the edge of each cell
        ri     = self.ri
        thetai = self.thetai
        phii   = self.phii

        # grid for plot
        rr_plt, tt_plt, phph_plt = np.meshgrid(ri, thetai, phii, indexing='ij')
        rxy_plt = rr_plt*np.sin(tt_plt)      # radius in xy-plane, r*sin(theta)
        zz_plt  = rr_plt*np.cos(tt_plt)      # z, r*cos(theta)
        xx_plt  = rxy_plt*np.cos(phph_plt)
        yy_plt  = rxy_plt*np.sin(phph_plt)


        # grid
        rr   = self.rr
        tt   = self.tt
        phph = self.phph
        rxy  = self.rxy
        zz   = self.zz

        # density
        nrho_g = self.nrho_g

        # velocity
        vr     = self.vr
        vtheta = self.vtheta
        vphi   = self.vphi


        vr_xy = vr*np.sin(tt)
        vr_zz = vr*np.cos(tt)
        v_xx = -vphi[:,-1,:]*np.sin(phph[:,-1,:]) + vr_xy[:,-1,:]*np.cos(phph[:,-1,:])
        v_yy = vphi[:,-1,:]*np.cos(phph[:,-1,:]) + vr_xy[:,-1,:]*np.sin(phph[:,-1,:])

        # parameters
        if len(nrho_g_range) == 0:
            nrho_g_max = np.nanmax(nrho_g)
            nrho_g_min = nrho_g_max*1e-7 # dynamic range of an order of five
        elif len(nrho_g_range) == 2:
            nrho_g_min, nrho_g_max = nrho_g_range
        else:
            print ('ERROR: nrho_g_range must be given as [min value, max value].')
            nrho_g_max = np.nanmax(nrho_g)
            nrho_g_min = nrho_g_max*1e-7 # dynamic range of an order of five

        if len(r_range) == 0:
            rmin = np.nanmin(rr)
            rmax = np.nanmax(rr)
        elif len(r_range) == 2:
            rmin, rmax = r_range
        else:
            print ('ERROR: r_range must be given as [min value, max value].')
            rmin = np.nanmin(rr)
            rmax = np.nanmax(rr)


        # gas disk
        fig2 = plt.figure(figsize=figsize)

        # plot 1; gas number density in r vs z
        ax3     = fig2.add_subplot(121)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax3)
        cax3    = divider.append_axes('right', '3%', pad='0%')

        im3   = ax3.pcolormesh(rxy_plt[:,:,nphi//2]/au, zz_plt[:,:,nphi//2]/au,
         nrho_g[:,:,nphi//2], cmap=cmap,
          norm = colors.LogNorm(vmin = nrho_g_min, vmax=nrho_g_max), rasterized=True)
        cbar3 = fig2.colorbar(im3,cax=cax3)

        # velocity vector
        # sampling
        vr_xy_vect = vr_xy[::step,::step,nphi//2]
        #vr_xy_vect = np.array([binArray(vr_xy_vect, i, binstep, binstep) for i in range(len(vr_xy_vect.shape))])
        vr_zz_vect = vr_zz[::step,::step,nphi//2]
        #vr_zz_vect = np.array([binArray(vr_zz_vect, i, binstep, binstep) for i in range(len(vr_zz_vect.shape))])
        rxy_vect = rxy[::step,::step,nphi//2]/au
        #rxy_vect = np.array([binArray(rxy_vect, i, binstep, binstep) for i in range(len(rxy_vect.shape))])
        zz_vect  = zz[::step,::step,nphi//2]/au
        #zz_vect  = np.array([binArray(zz_vect, i, binstep, binstep) for i in range(len(zz_vect.shape))])
        #print (rxy_vect.shape)

        # plot
        where_plt = np.where((rxy_vect >= rmin/au) & (rxy_vect <= rmax/au))
        #print (rxy_vect[where_plt]/au)
        ax3.quiver(rxy_vect[where_plt], zz_vect[where_plt],
         vr_xy_vect[where_plt], vr_zz_vect[where_plt],
          units='xy', scale = vscale, angles='uv', color='k', width=width)

        ax3.set_xlabel('radius (au)')
        ax3.set_ylabel('z (au)')

        ax3.set_xlim(0, rmax/au)
        ax3.set_ylim(0, rmax/au)
        #cbar3.set_label(r'$n_\mathrm{gas}\ \mathrm{(cm^{-3})}$')
        ax3.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        ax3.set_aspect(1.)


        # plot 2; density in r vs phi (xy-plane)
        ax4     = fig2.add_subplot(122)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax4)
        cax4    = divider.append_axes('right', '3%', pad='0%')

        xx = rxy*np.cos(phph)
        yy = rxy*np.sin(phph)

        im4   = ax4.pcolormesh(xx_plt[:,-1,:]/au, yy_plt[:,-1,:]/au,
         nrho_g[:,-1,:], cmap=cmap, norm = colors.LogNorm(vmin = nrho_g_min, vmax=nrho_g_max),
         rasterized=True)
        #im4   = ax4.pcolor(rr[:,-1,:]/au, phph[:,-1,:], nrho_gas[:,-1,:], cmap=cm.coolwarm, norm = colors.LogNorm(vmin = 10., vmax=1.e4))

        # velocity vector
        # sampling
        v_xx_vect = v_xx[::step,::step]
        v_yy_vect = v_yy[::step,::step]
        #v_xx_vect = np.array([binArray(v_xx, i, binstep, binstep) for i in range(len(v_xx.shape))])
        #v_yy_vect = np.array([binArray(v_yy, i, binstep, binstep) for i in range(len(v_yy.shape))])

        xx_vect = xx[::step,-1,::step]/au
        yy_vect = yy[::step,-1,::step]/au
        #xx_vect = np.array([binArray(xx_vect, i, binstep, binstep) for i in range(len(xx_vect.shape))])
        #yy_vect = np.array([binArray(yy_vect, i, binstep, binstep) for i in range(len(yy_vect.shape))])

        # plot
        where_plt = np.where((np.sqrt(xx_vect*xx_vect + yy_vect*yy_vect) >= rmin/au)\
         & (np.sqrt(xx_vect*xx_vect + yy_vect*yy_vect) <= rmax/au))
        ax4.quiver(xx_vect[where_plt], yy_vect[where_plt],
         v_xx_vect[where_plt], v_yy_vect[where_plt], units='xy',
          scale = vscale, angles='uv', color='k',width=width)

        cbar4 = fig2.colorbar(im4,cax=cax4)
        ax4.set_xlabel('x (au)')
        ax4.set_ylabel('y (au)')

        ax4.set_xlim(-rmax/au, rmax/au)
        ax4.set_ylim(-rmax/au, rmax/au)
        cbar4.set_label(r'$n_\mathrm{gas}\ \mathrm{(cm^{-3})}$')
        ax4.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        ax4.set_aspect(1.)


        # save figures
        fig2.subplots_adjust(wspace=wspace, hspace=hspace)
        fig2.savefig('gas_vfield.pdf',transparent=True)
        plt.close()


    # plot temperature profile
    def plot_temperature(self, infile='dust_temperature.dat',
     t_range=[], r_range=[], figsize=(11.69,8.27), cmap='coolwarm',
      fontsize=14, wspace=0.4, hspace=0.2, clevels=[10,20,30,40,50,60],
       aspect=1.):
        '''
        Plot temperature profile.

        Args:
        '''
        # modules
        import pandas as pd
        import os
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib import pyplot as plt
        from matplotlib import cm
        import matplotlib.colors as colors
        import mpl_toolkits.axes_grid1

        # setting for figures
        #plt.rcParams['font.family'] ='Arial'    # font (Times New Roman, Helvetica, Arial)
        plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
        plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
        plt.rcParams['font.size'] = fontsize    # fontsize

        # read model
        nr, ntheta, nphi = self.gridshape

        # edge of cells
        #  cuz the plot method, pcolormesh, requires the edge of each cell
        ri     = self.ri
        thetai = self.thetai
        phii   = self.phii

        rr, tt, phph = np.meshgrid(ri, thetai, phii, indexing='ij')
        rxy = rr*np.sin(tt)      # radius in xy-plane, r*sin(theta)
        zz  = rr*np.cos(tt)      # z, r*cos(theta)

        rho_d  = self.rho_d
        nrho_g = self.nrho_g

        xx = rxy*np.cos(phph)
        yy = rxy*np.sin(phph)


        # read file
        if os.path.exists(infile):
            pass
        else:
            print ('ERROR: Cannot find %s'%infile)
            return

        data = pd.read_csv('dust_temperature.dat', delimiter='\n', header=None).values
        iformat = data[0]
        imsize  = data[1]
        ndspc   = data[2]
        temp    = data[3:]

        #retemp = temp.reshape((nr,ntheta,nphi))
        retemp = temp.reshape((nphi,ntheta,nr)).T


        # setting for figure
        if len(r_range) == 0:
            rmin = np.nanmin(rr)
            rmax = np.nanmax(rr)
        elif len(r_range) == 2:
            rmin, rmax = r_range
        else:
            print ('ERROR: r_range must be given as [min value, max value].')
            rmin = np.nanmin(rr)
            rmax = np.nanmax(rr)

        # setting for figure
        if len(t_range) == 0:
            temp_min = 0.
            temp_max = np.nanmax(temp)
        elif len(t_range) == 2:
            temp_min, temp_max = t_range
        else:
            print ('ERROR: r_range must be given as [min value, max value].')
            temp_min = 0.
            temp_max = np.nanmax(temp)


        # figure
        fig = plt.figure(figsize=figsize)

        # plot #1: r-z plane
        ax1     = fig.add_subplot(121)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax1)
        cax1    = divider.append_axes('right', '3%', pad='0%')

        # plot
        im1   = ax1.pcolormesh(rxy[:,:,nphi//2]/au, zz[:,:,nphi//2]/au,
         retemp[:,:,nphi//2], cmap=cmap, vmin = temp_min, vmax=temp_max, rasterized=True)

        rxy_cont = (rxy[:nr,:ntheta,nphi//2] + rxy[1:nr+1,1:ntheta+1,nphi//2])*0.5
        zz_cont = (zz[:nr,:ntheta,nphi//2] + zz[1:nr+1,1:ntheta+1,nphi//2])*0.5
        im11  = ax1.contour(rxy_cont/au, zz_cont/au,
         retemp[:,:,nphi//2], colors='white', levels=clevels, linewidths=1.)

        cbar1 = fig.colorbar(im1, cax=cax1)
        ax1.set_xlabel('radius (au)')
        ax1.set_ylabel('z (au)')
        #cbar1.set_label(r'$T_\mathrm{dust}\ \mathrm{(K)}$')

        ax1.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        ax1.set_xlim(0,rmax/au)
        ax1.set_ylim(0,rmax/au)
        ax1.set_aspect(aspect)


        # plot #2: x-y plane
        ax2  = fig.add_subplot(122)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax2)
        cax2    = divider.append_axes('right', '3%', pad='0%')

        im2  = ax2.pcolormesh(xx[:,-1,:]/au, yy[:,-1,:]/au, retemp[:,-1,:],
         cmap=cmap, vmin = temp_min, vmax=temp_max, rasterized=True)

        cbar2 = fig.colorbar(im2, cax=cax2)
        ax2.set_xlabel('x (au)')
        ax2.set_ylabel('y (au)')
        cbar2.set_label(r'$T_\mathrm{dust}\ \mathrm{(K)}$')
        ax2.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)

        ax2.set_xlim(-rmax/au,rmax/au)
        ax2.set_ylim(-rmax/au,rmax/au)

        ax2.set_aspect(aspect)

        fig.subplots_adjust(wspace=wspace, hspace=hspace)
        fig.savefig('temp_dist.pdf', transparent=True)
        plt.close()

