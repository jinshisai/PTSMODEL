### import modules

import numpy as np
import pandas as pd
import os
import copy
from datetime import datetime
from astropy.io import fits
from datetime import datetime
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve_fft



### constants

au     = 1.49598e13      # Astronomical Unit       [cm]
kb     = 1.38064852e-16  # Boltzmann constant [erg K^-1]
hp     = 6.626070040e-27 # Planck constant [erg s]
clight = 2.99792458e10   # light speed [cm s^-1]




### functions
# convert Icgs --> Ibeam
def IcgsTObeam(Icgs,bmaj,bmin):
        '''
        Convert Intensity in cgs to in Jy/beam

        Icgs: intensity in cgs unit [erg s-1 cm-2 Hz-1 str-1]
        bmaj, bmin: FWHMs of beam major and minor axes [arcsec]
        Ibeam: intensity in Jy/beam
        '''

        # cgs --> Jy/str
        Imks = Icgs*1.e-7*1.e4   # cgs --> MKS
        Istr = Imks*1.0e26       # MKS --> Jy/str, 1 Jy = 10^-26 Wm-2Hz-1

        # Jy/sr -> Jy/beam(arcsec)
        # beam          = thmaj*thmin (arcsec^2) = (a_to_rad)^2*thmaj*thmin (rad^2)
        # Omg_beam (sr) = (pi/4ln(2))*beam (rad^2) = (pi/4ln2)*(a_to_rad)^2*thmaj*thmin
        # I [Jy/beam]   = I [Jy/sr] * Omg_beam
        C2              = np.pi/(4.*np.log(2.))
        radTOarcsec     = (60.0*60.0*180.0)/np.pi
        beam_th         = radTOarcsec*radTOarcsec/(C2*bmaj*bmin) # beam(sr) -> beam(arcsec), 1/beam_sr
        Ibeam           = Istr/beam_th
        return Ibeam


# equivalent brightness temperature
def IvTOJT(nu0, bmaj, bmin, Iv):
    # nu0 = header['RESTFRQ'] rest frequency [Hz]
    # bmaj = header['BMAJ'] # major beam size [deg]
    # bmin = header['BMIN'] # minor beam size [deg]
    # Iv: intensity [Jy/beam]
    # C2: coefficient to convert beam to str

    bmaj = bmaj*np.pi/180. # deg --> radian
    bmin = bmin*np.pi/180. # deg --> radian

    # Jy/beam -> Jy/sr
    # Omg_beam (sr) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/sr]
    C2 = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2  # beam --> str
    Istr = Iv/bTOstr # Jy/beam --> Jy/str
    Istr = Istr*1.0e-26 # Jy --> MKS (Jy = 10^-26 Wm-2Hz-1)
    Istr = Istr*1.e7*1.e-4 # MKS --> cgs

    JT = (clight*clight/(2.*nu0*nu0*kb))*Istr # equivalent brightness temperature
    return JT


# 2D Gaussian
def gaussian2D(x, y, A, mx, my, sigx, sigy, pa=0):
    # Generate normalized 2D Gaussian

    # x: x value (coordinate)
    # y: y value
    # A: Amplitude. Not a peak value, but the integrated value.
    # mx, my: mean values
    # sigx, sigy: standard deviations
    # pa: position angle [deg]. Counterclockwise is positive.
    x, y = rotate2d(x,y,pa)


    coeff = A/(2.0*np.pi*sigx*sigy)
    expx = np.exp(-(x-mx)*(x-mx)/(2.0*sigx*sigx))
    expy = np.exp(-(y-my)*(y-my)/(2.0*sigy*sigy))
    gauss=coeff*expx*expy
    return(gauss)


# 2D rotation
def rotate2d(x, y, angle, deg=True, coords=False):
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


# export
def export_radmc_tofits(outname, f='image.out', obsinfo='obsinfo.txt', restfreq=0, hdr=None, dist=140.,
    coordinate_center = '0h0m0.0s 0d0m0.0s', projection='SIN', frame = 'fk5', vsys=0,
    obname=None, beam_convolution=True, beam=[], Tb=False, add_noise=False, rms=None,
    noise_scale_factor=1.5, overwrite=False):
    '''
    Export a radmc output file into a fits file.

    '''
    # print
    print ('Export a radmc image to a fits file.')

    # reading file
    print ('reading files...')
    iformat = np.genfromtxt(f, dtype=None, max_rows=1)
    imsize  = np.genfromtxt(f, delimiter='     ',max_rows=1, skip_header=1, dtype=int)
    nlam    = np.genfromtxt(f, max_rows=1, skip_header=2, dtype=int)
    pixsize = np.genfromtxt(f, delimiter='     ',max_rows=1, skip_header=3, dtype=float)
    lam     = np.genfromtxt(f, max_rows=nlam, skip_header=4, dtype=float)
    data    = pd.read_csv(f, comment='#', encoding='utf-8', header=None, dtype=float, skiprows=5+nlam)
    image   = data.values
    #print iformat, imsize,nlam,pixsize,lam

    if obsinfo:
        inc, pa = np.genfromtxt(obsinfo,unpack=True,delimiter=' ',usecols=(0,3))
    else:
        print ('ERROR\texport_radmc_tofits: No observing information file.')
        return

    # grid
    nx, ny     = imsize
    delx, dely = pixsize/au/dist/3600.             # cm --> au --> arcsec --> degree
    delx       = - delx                            # observe object on a sky plane from the Earth, so that the x axis is inversed
    nlam       = int(nlam)
    freq       = clight/(lam*1.e-6*1.e2)
    reimage    = image.reshape((1,nlam,ny,nx))#,order='F')
    freq_vsys  = - vsys*1.e5*restfreq/clight
    #print len(reimage.shape), type(nlam)
    #print (clight*(1.-(freq[0]+freq_vsys)/restfreq)*1e-5)


    # set coordinates
    # image center is (0.0), meaning the origin of the coordinate is between pixels
    # all in degree for header
    xref, yref       = [0+delx*0.5,0+dely*0.5]
    crpix_x, crpix_y = [nx//2+1,ny//2+1]

    xmax = xref + delx*(nx - crpix_x)
    xmin = xref + delx*(1 - crpix_x)
    ymax = yref + dely*(ny - crpix_y)
    ymin = yref + dely*(1 - crpix_y)
    #print xmin, xmax, ymin, ymax

    # projection
    ra  = 'RA---'+projection
    dec = 'DEC--'+projection

    # coordinate center
    c_ra, c_dec = coordinate_center.split(' ')
    c_center    = SkyCoord(c_ra,c_dec,frame=frame) # ra,dec in [deg]
    alpha0, delta0 = [c_center.ra.deg, c_center.dec.deg]

    # reference: Calabretta and Greisen (2002)
    if projection == 'SIN':
        # Native longitude and latitude of the fiducial point
        phi0, theta0 = [0.,90.] # Zenithal (azimuthal) projections
        latpol = delta0
    elif projection == 'TAN':
        phi0, theta0 = [0.,90.]
        latpol = delta0
    elif projection == 'SFL':
        phi0, theta0 = [0.,0.]
        latpol = 90.-delta0
    elif projection == 'GLS':
        phi0, theta0 = [0.,0.]
        latpol = 90.-delta0
    else:
        print ('ERROR\texport_radmc_tofits: Currently there is no position for input projection.\
            Please choose from SIN, TAN, SFL or GLS.')

    if delta0 >= theta0:
        lonpole = 0.
    elif delta0 < theta0:
        lonpole = 180.

    # date
    today = datetime.today()
    today = today.strftime("%Y/%m/%d %H:%M:%S")


    # header
    if hdr:
        pass
    else:
        # open new fits file
        hdr   = fits.Header()

        # header info.
        hdr['SIMPLE']   = ('T', 'Standard FITS')
        hdr['BITPIX']   = ('-32', 'Floating point (32 bit)')

    # add header information
    hdr['NAXIS']    = len(reimage.shape)
    hdr['NAXIS1']   = nx
    hdr['NAXIS2']   = ny
    hdr['NAXIS3']   = nlam
    hdr['NAXIS4']   = 1
    hdr['EXTEND']   = 'T'
    hdr['BSCALE']   = (1.000000000000E+00, 'PHYSICAL = PIXEL*BSCALE + BZERO')
    hdr['BZERO']    = 0.000000000000E+00
    hdr['BTYPE']    = 'Intensity'
    if obname:
        hdr['OBJECT']   = obname
    hdr['BUNIT']    = ('Jy/beam ', 'Brightness (pixel) unit')
    hdr['EQUINOX']  = 2.000000000000E+03
    hdr['RADESYS']  = 'FK5'
    hdr['LONPOLE']  = lonpole
    hdr['LATPOLE']  = latpol
    hdr['PC1_1']    = 1.000000000000E+00
    hdr['PC2_1']    = 0.000000000000E+00
    hdr['PC3_1']    = 0.000000000000E+00
    hdr['PC4_1']    = 0.000000000000E+00
    hdr['PC1_2']    = 0.000000000000E+00
    hdr['PC2_2']    = 1.000000000000E+00
    hdr['PC3_2']    = 0.000000000000E+00
    hdr['PC4_2']    = 0.000000000000E+00
    hdr['PC1_3']    = 0.000000000000E+00
    hdr['PC2_3']    = 0.000000000000E+00
    hdr['PC3_3']    = 1.000000000000E+00
    hdr['PC4_3']    = 0.000000000000E+00
    hdr['PC1_4']    = 0.000000000000E+00
    hdr['PC2_4']    = 0.000000000000E+00
    hdr['PC3_4']    = 0.000000000000E+00
    hdr['PC4_4']    = 1.000000000000E+00
    hdr['CTYPE1']   = ra
    hdr['CRVAL1']   = xref + c_center.ra.deg
    hdr['CDELT1']   = delx
    hdr['CRPIX1']   = nx//2
    hdr['CUNIT1']   = 'deg'
    hdr['CTYPE2']   = dec
    hdr['CRVAL2']   = yref + c_center.dec.deg
    hdr['CDELT2']   = dely
    hdr['CRPIX2']   = ny//2
    hdr['CUNIT2']   = 'deg'
    hdr['CTYPE3']   = 'FREQ'
    hdr['CRVAL3']   = freq[0]+freq_vsys if nlam >= 2 else freq
    hdr['CDELT3']   = freq[1] - freq[0] if nlam >= 2 else 1.
    hdr['CRPIX3']   = 1.000000000000E+00
    hdr['CUNIT3']   = 'Hz'
    hdr['CTYPE4']   = 'STOKES'
    hdr['CRVAL4']   = 1.000000000000E+00
    hdr['CDELT4']   = 1.000000000000E+00
    hdr['CRPIX4']   = 1.000000000000E+00
    hdr['CUNIT4']   = ''
    hdr['PV2_1']    = 0.000000000000E+00
    hdr['PV2_2']    = 0.000000000000E+00
    hdr['RESTFRQ']  = (restfreq, 'Rest Frequency (Hz)')
    hdr['SPECSYS']  = ('LSRK', 'Spectral reference frame')
    hdr['ALTRVAL']  = (1.519999627520E+03, 'ALTernate frequency reference value')
    hdr['ALTRPIX']  = (1.000000000000E+00, 'ALTernate frequency reference pixel')
    hdr['VELREF']   = (257, '1 LSR, 2 HEL, 3 OBS, +256 Radio')
    hdr['TELESCOP'] = 'None'
    hdr['OBSERVER'] = 'Jinshi Sai/RADMC3D'
    hdr['DATE']     = (today, 'Date FITS file was written')
    hdr['INC_MDL']  = inc
    hdr['PA_MDL']   = pa
    #print hdr


    # output
    outimage = copy.deepcopy(reimage)


    # add noise
    if add_noise:
        if rms:
            if len(beam)==3:
                bmaj, bmin, bpa = beam
                f_cgstobeam     = IcgsTObeam(1.,bmaj,bmin)                     # factor = Ibeam/Icgs
                f_tbtoIbeam     = IvTOJT(restfreq, bmaj/3600., bmin/3600., 1.) # factor = Tmb/Ibeam

                # unit
                if Tb:
                    rms = rms/f_tbtoIbeam # Tb --> Jy/beam
                rms = rms/f_cgstobeam # Jy/beam --> in cgs

                # scaling to add noise before convolution
                s_ang = np.pi/(4.*np.log(2.))*bmaj*bmin/3600./3600.  # solid angle (deg^2)
                ratio = s_ang/np.abs(delx*dely)                      # pixels
                rms   = rms*np.sqrt(ratio)*noise_scale_factor        # scale factor for fine tuning
                #print (ratio)
            else:
                pass

            # noise maps
            noise = np.random.normal(loc=0., scale=rms, size=(outimage.shape))
        else:
            print ('ERROR\texport_radmc_tofits: The parameter rms is not found, which is necessary to add noise.')
            return

        print ('adding noise...')
        outimage = outimage + noise


    # beam convolution
    if beam_convolution:
        print ('convolving a beam...')
        if len(beam)==0:
            print ('ERROR\texport_radmc_tofits: The parameter beam is not found.\
                 It is necessary if you want to perform beam convolution.')
            return
        elif len(beam)==3:
            bmaj, bmin, bpa = beam
        else:
            print ('ERROR\texport_radmc_tofits: The input beam is wrong.\
                 It must be [bmaj, bmin, bpa]. The units will be arcsec, arcsec, and degree, respectively.')
            return

        # convolution
        xc    = np.arange(xmin,xmax+delx,delx)
        yc    = np.arange(ymin,ymax+dely,dely)
        xx,yy = np.meshgrid(xc, yc)

        sigx = 0.5*bmin/np.sqrt(2.*np.log(2.))/3600.            # in degree
        sigy = 0.5*bmaj/np.sqrt(2.*np.log(2.))/3600.            # in degree
        area = 1.
        beam = gaussian2D(xx, yy, area, 0., 0., sigx, sigy, pa = bpa)


        '''
        for ichan in range(nlam):
            imconv = outimage[0,ichan,:,:]
            outimage[0,ichan,:,:] = convolve_fft(imconv, beam, nan_treatment='fill')
        '''
        outimage = np.array([
            convolve_fft(outimage[0,ichan,:,:], beam, nan_treatment='fill')
            for ichan in range(nlam) ])
        outimage = outimage.reshape(1, nlam, ny, nx)

        # in cgs --> Jy/beam
        outimage = IcgsTObeam(outimage,bmaj,bmin)

        # header
        hdr['BMAJ'] = bmaj/3600. # degree
        hdr['BMIN'] = bmin/3600. # degree
        hdr['BPA']  = bpa


    # Jy/beam --> Tb
    if Tb:
        print ('converting unit from Jy/beam to Kelvine...')
        outimage = np.array([IvTOJT(freq[i], bmaj/3600., bmin/3600., outimage[0,i,:,:])
            for i in range(nlam)])
        hdr['BUNIT'] = ('K ', 'Tb')


    # write to fits file
    print ('writing fits file...')
    if os.path.exists(outname+'.fits'):
        os.system('rm -r '+outname+'.fits')

    fits.writeto(outname+'.fits', outimage, header=hdr)
    return