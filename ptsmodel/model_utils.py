import os
import numpy as np
import pandas as pd


### functions

# read
def read_grid(f='amr_grid.inp', outpixel='center'):
    # grid
    if os.path.exists(f) == False:
        print ('ERROR\tread_model: amr_grid.inp cannot be found.')
        return 0

    # dimension
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
    #qq           = np.meshgrid(rc,thetac,phic,indexing='ij') # (r, theta, phi) in the spherical coordinate
    #rr, tt, phph = qq
    #zr           = 0.5*np.pi - tt # angle from z axis (90deg - theta)
    #rxy          = rr*np.sin(tt)  # r in xy-plane
    #zz           = rr*np.cos(tt)  # z in xyz coordinate

    if outpixel == 'center':
        return rc, thetac, phic
    elif outpixel == 'edge':
        return ri, thetai, phii
    else:
        print('WARNING\tread_grid: outpixel must be center or edge.')
        print('WARNING\tread_grid: Ignore input value and return pixel centers.')
        return rc, thetac, phic

# read dust_temperature.dat
def read_temperature(f='dust_temperature.dat'):
    '''
    Read a RADMC-3D temperature file.
    '''
    # grid
    if os.path.exists('amr_grid.inp') == False:
        print ('ERROR\tread_temperature: amr_grid.inp cannot be found.')
        return
    else:
        nrtp = np.genfromtxt('amr_grid.inp', max_rows=1, skip_header=5, 
            delimiter=' ',dtype=int)
        nr, ntheta, nphi = nrtp

    # temperature
    if os.path.exists(f):
        data = pd.read_csv(f, delimiter='\n', header=None).values
        iformat = data[0]
        imsize  = data[1]
        ndspc   = data[2]
        temp    = data[3:]

        retemp = temp.reshape((nphi,ntheta,nr)).T
        return retemp
    else:
        print ('Found no temperature file.')
        return

def write_temperature(temp, f='dust_temperature.dat', overwrite=False):
    '''
    Write out xxx_temperature.dat file for RADMC-3D by hand.
    '''
    #retemp

# read LAMDA file
def read_lamda_moldata(infile):
    '''
    Read a molecular data file from LAMDA (Leiden Atomic and Molecular Database).

    Parameters
    ----------
     infile (str): Input LAMDA file.

    Return
    ------
    '''
    data = pd.read_csv(infile, comment='!', delimiter='\n', header=None)

    # get
    # line name, weight, nlevels
    line, weight, nlevels = data[0:3][0].values
    weight  = float(weight)
    nlevels = int(nlevels)

    # energy on each excitation level
    elevels = data[3:3+nlevels].values
    elevels = np.array([ elevels[i][0].split() for i in range(nlevels)])
    lev, EJ, gJ, J = elevels.T
    lev = np.array([ int(lev[i]) for i in range(nlevels)])
    EJ  = np.array([ float(EJ[i]) for i in range(nlevels)])
    gJ  = np.array([ float(gJ[i]) for i in range(nlevels)])
    J   = np.array([ int(J[i]) for i in range(nlevels)])

    # number of transition
    ntrans = data[0][3+nlevels].strip()
    ntrans = int(ntrans)

    # Einstein A coefficient
    vtrans = data[3+nlevels+1:3+nlevels+1+ntrans].values
    vtrans = np.array([vtrans[i][0].split() for i in range(ntrans)])

    itrans, Jup, Jlow, Acoeff, freq, delE = vtrans.T
    itrans = np.array([ int(itrans[i]) for i in range(ntrans)])
    Jup    = np.array([ int(Jup[i]) for i in range(ntrans)])
    Jlow   = np.array([ int(Jlow[i]) for i in range(ntrans)])
    Acoeff = np.array([ float(Acoeff[i]) for i in range(ntrans)])
    freq   = np.array([ float(freq[i]) for i in range(ntrans)])
    delE   = np.array([ float(delE[i]) for i in range(ntrans)])

    return line, weight, nlevels, EJ, gJ, J, ntrans, Jup, Jlow, Acoeff, freq, delE


def image_contsub(line, iline, filehead='image_'):
    '''
    Continuum subtraction.
    '''
    # image files
    f_line = filehead + '%s%i%i.out'%(line, iline, iline-1)
    f_cont = filehead + '%s%i%i_cont.out'%(line, iline, iline-1)

    # read file
    # line
    nx, ny = np.genfromtxt(f_line, delimiter='     ',max_rows=1, skip_header=1, dtype=int)
    nchan  = np.genfromtxt(f_line, max_rows=1, skip_header=2, dtype=int)
    d_line = pd.read_csv(f_line, comment='#', encoding='utf-8',
        header=None, dtype=float, skiprows=5+nchan)
    im_line = d_line.values
    im_line = im_line.reshape((1, nchan, ny, nx))#,order='F')
    # cont
    d_cont = pd.read_csv(f_cont, comment='#', encoding='utf-8',
        header=None, dtype=float, skiprows=5+1)
    im_cont   = d_cont.values
    im_cont   = im_cont.reshape((1, 1, ny, nx))#,order='F')

    # contsub
    im_line_contsub = im_line - im_cont

    # save file
    with open(f_line.replace('.out', '_contsub.out'), 'w') as f:
        f_i    = open(f_line, 'r')
        header = f_i.readlines()[0:5+nchan]
        header = ''.join(header)
        f.write(header)
        d_out = im_line_contsub.ravel() # Create a 1-D view, fortran-style indexing order='F'
        np.savetxt(f,d_out,fmt=['%13.6e'])

    return im_line_contsub

