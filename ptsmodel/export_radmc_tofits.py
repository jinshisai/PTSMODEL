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

# convert Icgs --> Iv Jy/pixel
def IcgsTOjpp(Icgs, px, py ,dist):
    '''
    Convert Intensity in cgs to in Jy/beam

    Icgs: intensity in cgs unit [erg s-1 cm-2 Hz-1 str-1]
    psize: pixel size (au)
    dist: distance to the object (pc)
    '''

    # cgs --> Jy/str
    Imks = Icgs*1.e-7*1.e4   # cgs --> MKS
    Istr = Imks*1.0e26       # MKS --> Jy/str, 1 Jy = 10^-26 Wm-2Hz-1

    # Jy/sr -> Jy/pixel
    px = np.radians(px/dist/3600.) # au --> radian
    py = np.radians(py/dist/3600.) # au --> radian
    # one_pixel_area = pixel*pixel (rad^2)
    # Exactly, one_pixel_area = 4.*np.arcsin(np.sin(psize*0.5)*np.sin(psize*0.5))
    #  but the result is almost the same pixel cuz pixel area is much small.
    # (When psize = 20 au and dist = 140 pc, S_apprx/S_acc = 1.00000000000004)
    # I [Jy/pixel]   = I [Jy/sr] * one_pixel_area
    one_pixel_area  = px*py
    Ijpp            = Istr*one_pixel_area # Iv (Jy per pixel)
    return Ijpp



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
def export_radmc_tofits(outname, f='image.out', obsinfo=None, restfreq=None, hdr=None, dist=140.,
    coordinate_center = '0h0m0.0s 0d0m0.0s', projection='SIN', frame = 'fk5', vsys=0,
    obname=None, beam_convolution=True, beam=[], Tb=False, add_noise=False, rms=None,
    noise_scale_factor=1.5, overwrite=False, units='Jy/beam'):
    '''
    Export a radmc output file into a fits file.

    Parameters
    ----------
     outname (str): Output name. outname.fits will be the one you get.
     f (str): radmc image file.
     obsinfo (str): Name of obsinfo file. If there is any input,
                    rest frequency, inclination, pa will be read from it.
     restfreq (float): Rest frequency in Hz. If obsinfo file is provided,
                       this parameter is not used.
     hdr (header): Header information can be directly given. Default None.
     dist (float): Distance to the object in pc.
     coordinate_center (str): Coordinate of the target source. Must be given
                              in a format of 'hms dms'.
     projection (str): Projection to the plane of sky.
     frame (str): Coordinate frame.
     vsys (float): Systemic velocity in km/s.
     obname (str): Object name written in header if given.
     beam_convolution (bool): Beam will be convolved if True.
     beam (list): Beam properties for beam convolution. Must be given
                  as [bmaj, bmin, bpa].
     Tb (bool): Output unit will be in K if True.
     add_noise (bool): Add noise if True.
     rms (float): RMS noise level.
     noise_scale_factor (float): Unit conversion factor for noise.
     units (str): Output unit of the intensity. Current option is Jy/pixel,
                  which is for CASA simulation.
     overwrite (bool): Overwrite existing fits file if True.
    '''
    print ('Export a radmc image to a fits file.')

    # obs. info.
    if obsinfo is None:
        # strip contsub
        if '_contsub' in f:
            obsinfo = f.replace('_contsub', '').replace('.out', '.obsinfo')
        elif '_cont' in f:
            obsinfo = f.replace('_cont', '').replace('.out', '.obsinfo')
        else:
            obsinfo = f.replace('.out', '.obsinfo')

    # read image file
    print ('reading files...')
    iformat = np.genfromtxt(f, dtype=None, max_rows=1)
    imsize  = np.genfromtxt(f, delimiter='     ',max_rows=1, skip_header=1, dtype=int)
    nlam    = np.genfromtxt(f, max_rows=1, skip_header=2, dtype=int)
    pixsize = np.genfromtxt(f, delimiter='     ',max_rows=1, skip_header=3, dtype=float)
    lam     = np.genfromtxt(f, max_rows=nlam, skip_header=4, dtype=float)
    data    = pd.read_csv(f, comment='#', encoding='utf-8', header=None, dtype=float, skiprows=5+nlam)
    image   = data.values
    #print iformat, imsize,nlam,pixsize,lam


    # rest frequency
    if os.path.exists(obsinfo):
        restfreq, inc, pa = np.genfromtxt(obsinfo, unpack=True, delimiter=' ')
    elif restfreq is None:
        print ('ERROR\texport_radmc_tofits: Rest frequency is not found.')
        return
    else:
        inc = None
        pa  = None

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
    hdr['CRPIX1']   = float(nx//2) + 1
    hdr['CUNIT1']   = 'deg'
    hdr['CTYPE2']   = dec
    hdr['CRVAL2']   = yref + c_center.dec.deg
    hdr['CDELT2']   = dely
    hdr['CRPIX2']   = float(ny//2) + 1
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
    hdr['ALTRVAL']  = (0.000000000000E+00, 'Alternate frequency reference value')
    hdr['ALTRPIX']  = (1.000000000000E+00, 'Alternate frequency reference pixel')
    hdr['VELREF']   = (257, '1 LSR, 2 HEL, 3 OBS, +256 Radio')
    hdr['TELESCOP'] = 'None'
    hdr['OBSERVER'] = 'RADMC3D'
    hdr['DATE']     = (today, 'Date FITS file was written')
    if inc is not None: hdr['INC'] = inc
    if pa is not None: hdr['PA']   = pa
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


    # unit
    #if units == 'Jy/pixel':
    #    # in cgs --> Jy/pixel
    #    outimage = IcgsTOjpp(outimage,pixsize[0]/au,pixsize[1]/au,dist)
    #    hdr['BUNIT'] = ('Jy/pixel', 'Brightness (pixel) unit')
    #    if beam_convolution:
    #        print ('Currently beam convolution is not supported when unit \
    #            of Jy/pixel is used.')
    #        beam_convolution = False
    #    Tb = False


    # beam convolution
    if beam_convolution:
        if len(beam)==3:
            print ('Convolving a beam...')
            # beam
            bmaj, bmin, bpa = beam

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
        else:
            print ('ERROR\texport_radmc_tofits: The format of the input beam is wrong.\
                 beam must be given as [bmaj (arcsec), bmin (arcsec), bpa (deg)].')
            print ('ERROR\texport_radmc_tofits: Output units will be Jy/pixel.')
            beam_convolution = False
            # in cgs --> Jy/pixel
            outimage = IcgsTOjpp(outimage,pixsize[0]/au,pixsize[1]/au,dist)
            hdr['BUNIT'] = ('Jy/pixel', 'Brightness (pixel) unit')
    else:
        # in cgs --> Jy/pixel
        outimage = IcgsTOjpp(outimage,pixsize[0]/au,pixsize[1]/au,dist)
        hdr['BUNIT'] = ('Jy/pixel', 'Brightness (pixel) unit')


    # Jy/beam --> Tb
    if beam_convolution & Tb:
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