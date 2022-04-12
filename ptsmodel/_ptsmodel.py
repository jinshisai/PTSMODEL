# -*- coding: utf-8 -*-
'''
Made and developed by J. Sai.

email: jn.insa.sai@gmail.com
'''

# modules
import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants, units
#import matplotlib
#matplotlib.use('TkAgg')

from .model_utils import read_lamda_moldata, image_contsub
from . import model_utils

# constants (in cgs)

Ggrav  = constants.G.cgs.value        # Gravitational constant
ms     = constants.M_sun.cgs.value    # Solar mass (g)
ls     = constants.L_sun.cgs.value    # Solar luminosity (erg s^-1)
rs     = constants.R_sun.cgs.value    # Solar radius (cm)
au     = units.au.to('cm')            # 1 au (cm)
pc     = units.pc.to('cm')            # 1 pc (cm)
clight = constants.c.cgs.value        # light speed (cm s^-1)
kb     = constants.k_B.cgs.value      # Boltzman coefficient
sigsb  = constants.sigma_sb.cgs.value # Stefan-Boltzmann constant (erg s^-1 cm^-2 K^-4)
mp     = constants.m_p.cgs.value      # Proton mass (g)

#au    = 1.49598e13      # Astronomical Unit       [cm]
#pc    = 3.08572e18      # Parsec                  [cm]
#ms    = 1.98892e33      # Solar mass              [g]
#ls    = 3.8525e33       # Solar luminosity        [erg/s]
#rs    = 6.96e10         # Solar radius            [cm]
#Ggrav = 6.67428e-8      # gravitational constant  [dyn cm^2 g^-2]
#mp    = 1.672621898e-24 # proton mass [g]
#sigsb = 5.670367e-5     # Stefan-Boltzmann constant [erg s^-1 cm^-2 K^-4]
#kb    = 1.38064852e-16  # Boltzman coefficient in cgs


# ProToStellar MODEL
class PTSMODEL():
    '''
    Build a model of a protostar, a disk, and an envelope system.

    '''

    def __init__(self, modelname, nr=None, ntheta=None, nphi=None ,rmin=None, rmax=None,
     thetamin=0., thetamax=np.pi*0.5, phimin=0., phimax=2.*np.pi, readfiles=False):
        self.modelname = modelname

        # initialize
        self.disk     = 0
        self.envelope = 0
        self.cavity   = 0
        self.rho_disk = np.array([])
        self.rho_env  = np.array([])
        self.rho_g    = np.array([])
        self.turbulence = False

        if readfiles:
            self.read_model()
            return

        # makegrid
        if all(np.array([nr, ntheta, nphi, rmin, rmax, thetamin, thetamax, phimin, phimax]) != None):
            print ('Making a grid..')
            self.makegrid(nr, ntheta, nphi ,rmin, rmax,
                thetamin=thetamin, thetamax=thetamax, phimin=phimin, phimax=phimax)


    def read_model(self):
        '''
        Reconstuct model by reading model files.

        '''

        import pandas as pd
        import os
        import glob

        # reading file
        # grid
        f = 'amr_grid.inp'
        if os.path.exists(f) == False:
            print ('ERROR\tread_model: amr_grid.inp cannot be found.')
            return

        # dimension
        nrtp             = np.genfromtxt(f, max_rows=1, skip_header=5, delimiter=' ',dtype=int)
        nr, ntheta, nphi = nrtp
        arraysize        = (nr,ntheta,nphi)
        self.nr     = nr
        self.ntheta = ntheta
        self.nphi   = nphi

        dread            = pd.read_csv(f, skiprows=6, comment='#', encoding='utf-8',header=None)
        coords           = dread.values
        ri, thetai, phii = np.split(coords,[nr+1,nr+ntheta+2])

        # centers of each cell
        rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )                 # centers of each cell
        thetac   = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
        phic     = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )

        # get grid
        qq           = np.meshgrid(rc,thetac,phic,indexing='ij') # (r, theta, phi) in the spherical coordinate
        rr, tt, phph = qq
        zr           = 0.5*np.pi - tt # angle from z axis (90deg - theta)
        rxy          = rr*np.sin(tt)  # r in xy-plane
        zz           = rr*np.cos(tt)  # z in xyz coordinate

        # save
        self.ri        = ri
        self.thetai    = thetai
        self.phii      = phii
        self.r         = rc
        self.theta     = thetac
        self.phi       = phic
        self.gridshape = arraysize
        self.grid      = qq
        self.rr        = rr
        self.tt        = tt
        self.phph      = phph
        self.rxy       = rxy
        self.zz        = zz


        # density
        # dust denisty [g/cm3]
        f = 'dust_density.inp'
        if os.path.exists(f) == False:
            print ('WARNING\t: dust_density.inp cannot be found. Put zero for dust density.')
            rho_d = np.zeros(arraysize)
        else:
            dread = pd.read_table(f, skiprows=3, comment='#', encoding='utf-8',header=None)
            rho_d = dread.values
            rho_d = np.reshape(rho_d.T,arraysize, order='F')
            #print drho_dust.shape


        # gas number denisty [/cm3]
        files = glob.glob('numberdens_*.inp')
        if len(files) == 0:
            print ("WARNING\t: numberdens_*.inp doesn't exist. Put zero for gas density.")
            nrho_g = np.zeros(arraysize)
            rho_g  = np.zeros(arraysize)
        elif len(files) == 1:
            f      = files[0]
            dread  = pd.read_table(f, skiprows=2, comment='#', encoding='utf-8',header=None)
            nrho_g = dread.values
            nrho_g = np.reshape(nrho_g.T,arraysize,order='F')
        else:
            f = files[0]
            print ("WARNING\t: More than two numberdens_*.inp files? Read '%s' for the moment."%f)
            dread  = pd.read_table(f, skiprows=2, comment='#', encoding='utf-8',header=None)
            nrho_g = dread.values
            nrho_g = np.reshape(nrho_g.T,arraysize,order='F')

        # save
        self.rho_d  = rho_d
        self.nrho_g = nrho_g


        # gas velocity [cm/s]
        f = 'gas_velocity.inp'
        if os.path.exists(f) == False:
            print ("WARNING\t: gas_velocity.inp doesn't exist. Put zero for gas v-field.")
            vr     = np.zeros(arraysize)
            vtheta = np.zeros(arraysize)
            vphi   = np.zeros(arraysize)
        else:
            dread = pd.read_csv(f, skiprows=2, comment='#', encoding='utf-8',header=None, delimiter=' ',parse_dates=True, keep_date_col=True, skipinitialspace=True)
            vrtp             = dread.values
            vr, vtheta, vphi = vrtp.T
            vr               = np.reshape(vr,arraysize,order='F')
            vtheta           = np.reshape(vtheta,arraysize,order='F')
            vphi             = np.reshape(vphi,arraysize,order='F')

        self.vr     = vr
        self.vtheta = vtheta
        self.vphi   = vphi


        # temperature if exists
        f = 'dust_temperature.dat'
        if os.path.exists(f):
            data = pd.read_csv(f, delimiter='\n', header=None).values
            iformat = data[0]
            imsize  = data[1]
            ndspc   = data[2]
            temp    = data[3:]

            retemp = temp.reshape((nphi,ntheta,nr)).T
            self.temp = retemp
        else:
            self.temp = None


    def makegrid(self, nr, ntheta, nphi ,rmin, rmax,
     thetamin=0., thetamax=np.pi*0.5, phimin=0., phimax=2.*np.pi, logr=True):
        '''
        Make a grid for model calculation. Currently, only polar coodinates are allowed.

        Args:
            nr, ntheta, nphi: number of grid
            xxxmin, xxxmax: minimum and maximum values of axes
        '''
        # dimension
        self.nr     = nr
        self.ntheta = ntheta
        self.nphi   = nphi

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

        self.r         = rc
        self.theta     = thetac
        self.phi       = phic

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
        # put a disk
        self.disk = 1

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
            self.sig0_disk(mdisk, rin, rout, plsig)
            sig0 = self.sig0

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
    def sig0_disk(self, mdisk, rin, rout, p):
        '''
        Calculate Sigma 0, column density of the disk at 1 au, from Mdisk.
        Disk is a cut-off disk.
        Mdisk = 2 pi int_rin^rout Sig0 (r/1 au)^p r dr

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

        self.sig0 = sig0


    # Density distribution of the envelope (Ulrich76)
    def rho_envUl76(self, rho0, rc, re_in=None, re_out=None):
        '''
        Calculate density distribution of the infalling envelope model proposed in Ulrich (1976).
        See Ulrich (1976) for the detail.

        Args:
            rc: centrifugal radius (cm)
            rho0: density at rc (g cm^-3)
            re_in, re_out: inner and outer radii of an envelope (cm)
        '''
        # put an envelope
        self.envelope = 1

        # parameters
        self.rho_e0 = rho0
        self.rc     = rc
        self.re_in  = re_in
        self.re_out = re_out

        r     = self.rr
        theta = self.tt
        costh = np.cos(theta)

        # solve costh0^3 + (-1 + r/rc) costh0 - costh*r/rc = 0
        costh0, sinth0 = der_th0ph0(r,theta,rc)
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
        self.rho_envUl76(rho0, rc, re_in=None, re_out=None)
        rho_u76 = self.rho_env
        costh0  = self.costh0
        sinth0  = self.sinth0

        cosh2   = np.cosh(eta*costh0)**2.
        sech2   = 1./cosh2

        rho_sheet = rho_u76 * sech2*(eta/np.tanh(eta))
        self.rho_env = rho_sheet


    # flow
    def rho_flow(self, rho0, rc, theta0_0, phi0_0, delth0, delph0, re_in=None, re_out=None):
        '''
        Calculate infalling flows using the model proposed in Ulrich (1976).
        See Ulrich (1976) for the detail.

        r: radius (any unit)
        rc: centrifugal radius (any unit)
        rho0: density at rc (any unit)
        theta: theta of spherical coordinates (rad)
        '''
        # deg --> rad
        theta0_0 = theta0_0*np.pi/180.
        phi0_0   = phi0_0*np.pi/180.
        delth0   = delth0*np.pi/180.
        delph0   = delph0*np.pi/180.


        # save parameters
        self.rho_e0 = rho0
        self.rc     = rc
        self.re_in  = re_in
        self.re_out = re_out

        r     = self.rr
        theta = self.tt
        phi   = self.phph

        # trig functions
        costh = np.cos(theta)
        sinth = np.sin(theta)
        tanth = np.tan(theta)
        cosph = np.cos(phi)
        sinph = np.sin(phi)

        # to treat pi/2 < theta < pi assuming symmetry
        #index = np.where(theta > np.pi*0.5)

        # solve costh0^3 + (-1 + r/rc) costh0 - costh*r/rc = 0
        costh0, sinth0 = der_th0ph0(r,theta,rc)
        self.costh0 = costh0
        self.sinth0 = sinth0
        tanth0 = sinth0/costh0
        theta0 = np.arccos(costh0)

        # derive phi0
        # From Ulrich76,
        #  tan(phi - phi_0)=tan(alpha)/sin(theta_0).
        #  cos(alpha) = cos(theta)/cos(theta_0)
        # We can also obtain following equations:
        #  sin(alpha) = sin(delphi) x sin(theta), where delphi = (phi - phi_0)
        #   from spherical trigonometry. alpha is within 0 to pi/2 from its geometry.
        # Thus,
        #  tan(delphi) = sin(alpha)/cos(alpha)/sin(theta_0)
        #              = sin(delphi) x sin(theta) x cos(theta_0)/cos(theta)/sin(theta_0)
        #  1/cos(delphi) = tan(theta)/tan(theta_0)
        #  cos(delphi) = tan(theta_0)/tan(theta)

        cosdphi = tanth0/tanth
        dphi    = np.arccos(cosdphi) # 0 < dphi < pi
        phi0    = phi - dphi
        #print (np.nanmin(phi0), np.nanmax(phi0))

        # put phi0 within 0 to 2 pi
        phi0[np.where(phi0 < 0.)]       = phi0[np.where(phi0 < 0.)] + 2.*np.pi
        phi0[np.where(phi0 > 2.*np.pi)] = phi0[np.where(phi0 > 2.*np.pi)] - 2.*np.pi

        # densities
        self.rho_envUl76(rho0, rc, re_in=None, re_out=None) # spherical envelope
        rho_sp = self.rho_env
        rho_fl = np.zeros(rho_sp.shape)                     # flow

        # flow
        #index_th0 = np.where((theta0 >= theta0_0-delth0) & (theta0 <= theta0_0+delth0))
        #index_ph0 = np.where((phi0 >= phi0_0-delph0) & (phi0 <= phi0_0+delph0))
        print ('range of theta_0: %.f--%.f deg'%( (theta0_0-delth0)*180./np.pi, (theta0_0+delth0)*180./np.pi))
        print ('range of phi_0: %.f--%.f deg'%( (phi0_0-delph0)*180./np.pi, (phi0_0+delph0)*180./np.pi))

        # if phi0 range across phi0=0.
        if phi0_0 - delph0 < 0.:
            index = np.where((theta0 >= theta0_0 - delth0) & (theta0 <= theta0_0 + delth0) & (phi0 >= phi0_0 - delph0 + 2.*np.pi))
            rho_fl[index] = rho_sp[index]
            #print (phi0_0-delph0 + 2.*np.pi)
            phirng_min    = np.nanmax(phi[index]*180./np.pi) - 2.*np.pi
            index = np.where((theta0 >= theta0_0 - delth0) & (theta0 <= theta0_0 + delth0) & (phi0 <= phi0_0 + delph0))
            rho_fl[index] = rho_sp[index]
            phirng_max    = np.nanmax(phi[index]*180./np.pi)
        else:
            index = np.where((theta0 >= theta0_0-delth0) & (theta0 <= theta0_0+delth0) & (phi0 >= phi0_0-delph0) & (phi0 <= phi0_0+delph0))
            rho_fl[index] = rho_sp[index]
            phirng_min, phirng_max = [np.nanmin(phi[index]*180./np.pi), np.nanmax(phi[index]*180./np.pi)]

            # rotate angle
            if np.abs(phirng_min - phirng_max) > 180.:
                phirng = phi[index]*180./np.pi
                phirng[np.where(phirng > 180.)] = phirng[np.where(phirng > 180.)]-360.
                phirng_min, phirng_max = [np.nanmin(phirng), np.nanmax(phirng)]
                #print (phirng)

        if re_in:
            rho_fl[np.where(r < re_in)] = 0.

        if re_out:
            rho_fl[np.where(r > re_out)] = 0.

        self.rho_env = rho_fl

        #rho_fl[index_th0] = rho_sp[index_th0]
        #rho_fl[index_ph0] = rho_sp[index_ph0]
        print ('range of theta: %.f--%.f deg'%(np.nanmin(theta[index]*180./np.pi), np.nanmax(theta[index]*180./np.pi)))
        print ('range of phi: %.f--%.f deg'%(phirng_min, phirng_max))
        #print (np.nanmin(theta), np.nanmax(theta))

        return rho_fl


    def outflow_cavity(self, ang, rho_cavity=0., func=None):
        '''
        Open an outflow cavity
        '''

        self.cavity = 1
        self.where_cavity = self.tt <= ang*np.pi/180.
        self.rho_cavity   = rho_cavity


    def rho_model(self, Xconv, mu=2.8, gtod_ratio = 100., disk_height=1):
        '''
        Make a density model

        Args:
            gtod_ratio: gas-to-dust mass ratio
            mu: mean molecular weight
            Xconv: Abandance of a molecule you want to model.
        '''
        rho = np.zeros(self.gridshape)

        # Density
        if (self.disk == 1) & (self.envelope == 1):
            # disk + envelope

            # assuming the disk height is one scale height
            where_disk = np.where((self.rxy <= self.rdisk_out) & (np.abs(self.zz) <= self.hh*disk_height))
            self.where_disk = where_disk

            rho = self.rho_env                          # Put an envelope
            rho[where_disk] = self.rho_disk[where_disk] # Put a disk within an envelope
        elif (self.disk == 1) & (self.envelope == 0):
            # disk only
            rho = self.rho_disk
        elif (self.disk == 0) & (self.envelope == 1):
            # envelope only
            rho = self.rho_env
        else:
            print('WARNING: No structure is input.')


        # Outflow cavity
        if self.cavity:
            rho[self.where_cavity] = self.rho_cavity

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

        if (self.disk ==1) & (self.envelope == 1):
            # v_envelope
            vr_env, vtheta_env, vphi_env = vinf_Ul76(rr, tt, self.mstar, self.rc)
            vr     = vr_env
            vtheta = vtheta_env
            vphi   = vphi_env

            # v_disk
            vr_disk, vtheta_disk, vphi_disk = v_steadydisk(rxy, self.mstar)
            where_disk = self.where_disk
            vr[where_disk]     = vr_disk[where_disk]
            vtheta[where_disk] = vtheta_disk[where_disk]
            vphi[where_disk]   = vphi_disk[where_disk]
        elif (self.disk == 1) & (self.envelope == 0):
            vr_disk, vtheta_disk, vphi_disk = v_steadydisk(rxy, self.mstar)
            vr     = vr_disk
            vtheta = vtheta_disk
            vphi   = vphi_disk
        elif (self.disk == 0) & (self.envelope == 1):
            vr_env, vtheta_env, vphi_env = vinf_Ul76(rr, tt, self.mstar, self.rc)
            vr     = vr_env
            vtheta = vtheta_env
            vphi   = vphi_env
        else:
            print('WARNING: No structure is input.')

        self.vr     = vr
        self.vtheta = vtheta
        self.vphi   = vphi


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
            #sigsb = 5.670367e-5     # Stefan-Boltzmann constant [erg s^-1 cm^-2 K^-4]
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
        #sigsb = 5.670367e-5     # Stefan-Boltzmann constant [erg s^-1 cm^-2 K^-4]
        tstar = (lstar/(4*np.pi*rstar*rstar)/sigsb)**0.25 # Stefan-Boltzmann
        self.tstar = tstar


    # Make input files for RADMC-3D
    def export_to_radmc3d(self, nphot, dustopac='nrmS03',line='c18o', iseed = -5415):
        nr, ntheta, nphi = self.gridshape
        self.dustopac = dustopac
        self.line     = line
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
            # Include r,theta, phi coordinates or not
            incl_rtp = [0 if n_i == 1 else 1 for n_i in [nr, ntheta, nphi]]
            f.write('%i %i %i\n'%(incl_rtp[0], incl_rtp[1], incl_rtp[2]))
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
            f.write('nphot = %d\n'%nphot)
            f.write('iseed = %d\n'%iseed)
            f.write('scattering_mode_max = 0\n')
            f.write('iranfreqmode = 1\n')
            f.write('tgas_eq_tdust = 1')
        ####################### End output ########################


    def run_mctherm(self, nthreads=1):
        '''
        Calculate temperature distribution with RADMC3D
        '''
        # files
        path_infiles = model_utils.__file__.split('ptsmodel')[0]+'infiles/'
        for fin in ['dustkappa_%s.inp'%self.dustopac, 'molecule_%s.inp'%self.line]:
            if os.path.exists(fin) == False:
                shutil.copy(path_infiles + fin, '.')

        print ('radmc3d mctherm setthreads %i'%nthreads)
        os.system('radmc3d mctherm setthreads %i'%nthreads)


    def solve_radtrans_line(self, npix, iline, sizeau,
        width_spw, nchan, pa, inc, contsub=True):
        '''
        Solve radiative transfer for line with RADMC3D.

        Parameters
        ----------
         npix: Pixel number of the output image. Output image will have npix x npix size.
         iline: Upper excitation level of line for which radiative transfer is solved.
         sizeau: Image size in diamiter in au.
         width_spw: Total width of the spectral window.
         nchan: Channel number of the spectral window.
         pa: Position angle of the object on the plane of sky.
         inc: Inclination angle of the object.

        Return
        ------
         None
        '''

        # read moldata
        _, weight, nlevels, EJ, gJ, J, ntrans, Jup, Acoeff, freq, delE =\
        read_lamda_moldata('molecule_'+self.line+'.inp')
        restfreq = freq[iline-1]*1e9 # rest frequency (Hz)

        run_radmc = 'radmc3d image npix %i phi 0 iline %i \
        sizeau %.f widthkms %.2f linenlam %i posang %.2f \
        incl %.2f'%(npix, iline, sizeau, width_spw, nchan, pa, inc)
        print ('Solve radiative transfer')
        print (run_radmc)
        os.system(run_radmc)

        # output
        fout_line = 'image_%s%i%i.out'%(self.line, iline, iline-1)
        print ('image.out --> '+fout_line)
        os.system('mv image.out '+fout_line)

        # obsinfo
        with open(fout_line.replace('.out','.obsinfo'),'w+') as f:
            f.write('# Information of observation to make image.out\n')
            f.write('# restfrequency inclination position_angle\n')
            f.write('# Hz deg deg\n')
            f.write('\n')
            f.write('%d %d %d'%(restfreq,inc,pa))

        # contsub
        if contsub:
            lam = clight*1e-2/restfreq*1e6 # micron
            print ('Solve radiative transfer for continuum\
                for continuum subtraction.')
            run_radmc = 'radmc3d image noline npix %i phi 0 \
            sizeau %.f posang %.2f incl %.2f lambda %.13e'\
            %(npix, sizeau, pa, inc, lam)
            print (run_radmc)
            os.system(run_radmc)
            # output
            fout_cont = 'image_%s%i%i_cont.out'%(self.line, iline, iline-1)
            print ('image.out --> '+fout_cont)
            os.system('cp image.out '+fout_cont)

            # contsub
            im_line_contsub = image_contsub(self.line, iline)


    def solve_radtrans_cont(npix, sizeau, pa, inc, lam):
        '''
        Solve radiative transfer for continuum with RADMC3D.

        Parameters
        ----------
         npix: Pixel number of the output image. Output image will have npix x npix size.
         sizeau: Image size in diamiter in au.
         pa: Position angle of the object on the plane of sky.
         inc: Inclination angle of the object.
         lam (float): lambda at which radiative transfer is solved (micron)

        Return
        ------
         None
        '''
        print ('Solve radiative transfer for continuum\
            for continuum subtraction.')
        run_radmc = 'radmc3d image npix %i phi 0 \
        sizeau %.f posang %.2f incl %.2f lambda %.13e'\
        %(npix, sizeau, pa, inc, lam)
        print (run_radmc)
        os.system(run_radmc)
        # output
        fout_cont = 'image_cont_%i.out'%lam
        print ('image.out --> '+fout_cont)
        os.system('cp image.out '+fout_cont)

        # obsinfo
        with open(fout_cont.replace('.out','.obsinfo'),'w+') as f:
            f.write('# Information of observation to make image.out\n')
            f.write('# restfrequency inclination position_angle\n')
            f.write('# Hz deg deg\n')
            f.write('\n')
            f.write('%d %d %d'%(clight/(lam*1e-4),inc,pa))


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

        ax1.set_xlabel('Radius (au)')
        ax1.set_ylabel('z (au)')
        #cbar1.set_label(r'$\rho_\mathrm{dust}\ \mathrm{(g\ cm^{-3})}$')
        ax1.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
        ax1.set_aspect(1)


        # plot 2; density in r vs phi (xy-plane)
        ax2     = fig1.add_subplot(122)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax2)
        cax2    = divider.append_axes('right', '3%', pad='0%')

        indx_mid = np.argmin(np.abs(self.tt[0,:,0] - np.pi*0.5)) # mid-plane
        im2   = ax2.pcolormesh(xx[:,indx_mid,:]/au, yy[:,indx_mid,:]/au, rho_d[:,indx_mid,:],
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

        im4   = ax4.pcolormesh(xx[:,indx_mid,:]/au, yy[:,indx_mid,:]/au, nrho_g[:,indx_mid,:], cmap=cmap,
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
        fig1.savefig('dust_density.pdf',transparent=True)

        fig2.subplots_adjust(wspace=wspace, hspace=hspace)
        fig2.savefig('gas_density.pdf',transparent=True)
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
        vr_zz_vect = vr_zz[::step,::step,nphi//2]
        rxy_vect = rxy[::step,::step,nphi//2]/au
        zz_vect  = zz[::step,::step,nphi//2]/au


        # plot
        where_plt = np.where((rxy_vect >= rmin/au) & (rxy_vect <= rmax/au))
        #print (rxy_vect[where_plt]/au)
        ax3.quiver(rxy_vect[where_plt], zz_vect[where_plt],
         vr_xy_vect[where_plt], vr_zz_vect[where_plt],
          units='xy', scale = vscale, angles='uv', color='k', width=width)

        ax3.set_xlabel(r'$r$ (au)')
        ax3.set_ylabel(r'$z$ (au)')

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

        indx_mid = np.argmin(np.abs(tt[0,:,0] - np.pi*0.5)) # mid-plane
        im4   = ax4.pcolormesh(xx_plt[:,indx_mid,:]/au, yy_plt[:,indx_mid,:]/au,
         nrho_g[:,indx_mid,:], cmap=cmap, norm = colors.LogNorm(vmin = nrho_g_min, vmax=nrho_g_max),
         rasterized=True)
        #im4   = ax4.pcolor(rr[:,-1,:]/au, phph[:,-1,:], nrho_gas[:,-1,:], cmap=cm.coolwarm, norm = colors.LogNorm(vmin = 10., vmax=1.e4))

        # velocity vector
        # sampling
        v_xx_vect = v_xx[::step,::step]
        v_yy_vect = v_yy[::step,::step]
        #v_xx_vect = np.array([binArray(v_xx, i, binstep, binstep) for i in range(len(v_xx.shape))])
        #v_yy_vect = np.array([binArray(v_yy, i, binstep, binstep) for i in range(len(v_yy.shape))])

        xx_vect = xx[::step,indx_mid,::step]/au
        yy_vect = yy[::step,indx_mid,::step]/au
        #xx_vect = np.array([binArray(xx_vect, i, binstep, binstep) for i in range(len(xx_vect.shape))])
        #yy_vect = np.array([binArray(yy_vect, i, binstep, binstep) for i in range(len(yy_vect.shape))])

        # plot
        where_plt = np.where((np.sqrt(xx_vect*xx_vect + yy_vect*yy_vect) >= rmin/au)\
         & (np.sqrt(xx_vect*xx_vect + yy_vect*yy_vect) <= rmax/au))
        ax4.quiver(xx_vect[where_plt], yy_vect[where_plt],
         v_xx_vect[where_plt], v_yy_vect[where_plt], units='xy',
          scale = vscale, angles='uv', color='k',width=width)

        cbar4 = fig2.colorbar(im4,cax=cax4)
        ax4.set_xlabel(r'$x$ (au)')
        ax4.set_ylabel(r'$y$ (au)')

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
    vphi   = Vkep(rxy, mstar)   # Keplerian rotation

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
    costh0, sinth0 = der_th0ph0(r,theta,rc)

    vr     = - cgrav*np.sqrt(1+(costh/costh0))
    vtheta = cgrav*(costh0 - costh)\
        *np.sqrt((costh0+costh)/(costh0*sinth*sinth))
    vphi = cgrav*(sinth0/sinth)\
        *np.sqrt(1.-costh/costh0)

    return vr, vtheta, vphi


# Derive theta0 and phi0
def der_th0ph0(r,theta,rc):
    '''
    Derive theta0 and phi0 from given r, theta, phi.
    See Ulrich (1976) for the detail. Here 0 < theta < pi/2.
    To get solutions, see also Mendoza (2004)
    '''
    # assuming symmetry about the xy-plane to treat pi/2 < theta < pi
    index = np.where(theta > np.pi*0.5)
    theta[index] = np.pi - theta[index]
    costh        = np.cos(theta)

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
    case = term_rcos_2**2. - term_1r3**3.
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

    # put theta back to input, 0 < theta < pi
    theta[index]  = np.pi - theta[index]
    costh0[index] = - costh0[index]

    sinth0 = np.sqrt(1.-costh0*costh0) # 0 < theta < pi/2, don't use here though

    return costh0, sinth0