import os
import numpy as np
import pandas as pd


### functions

class AMRGrid(object):
    """docstring for AMRGrid"""
    def __init__(self, f = 'amr_grid.inp'):
        super(AMRGrid, self).__init__()
        self.fgrid = f

        # grid
        if os.path.exists(f) == False:
            print ('ERROR\tAMRGrid: amr_grid.inp cannot be found.')
            return 0

        # format
        iformat, gridstyle, coordsys, gridinfo = np.genfromtxt(f, max_rows=4, delimiter='\n',dtype=int)
        incl_x, incl_y, incl_z = np.genfromtxt(f, max_rows=1, skip_header=4, delimiter=' ',dtype=int)
        self.iformat = iformat
        self.gridstyle = gridstyle
        self.coordsystem = coordsys
        self.gridinfo = gridinfo
        self.incl_x, self.incl_y, self.incl_z = incl_x, incl_y, incl_z
        
        if gridstyle != 0:
            print('ERROR\tAMRGrid: gridstyle != 0 is found.')
            print('ERROR\tAMRGrid: Currently only regular grid is supported.')
            return 0

        # read grid
        if coordsys < 100:
            # Cartesian

            # dimension
            nx, ny, nz = np.genfromtxt(f, max_rows=1, skip_header=5, delimiter=' ',dtype=int)
            arraysize = (nx,ny,nz)


            dread            = pd.read_csv(f, skiprows=6, comment='#', 
                encoding='utf-8', header=None, dtype=float, sep = '\s+')
            coords = dread.values
            xi, yi, zi = coords #np.split(coords,[nx+1,nx+ny+2])

            # centers of each cell
            xc = 0.5 * ( xi[0:nx] + xi[1:nx+1] ) # centers of each cell
            yc = 0.5 * ( yi[0:ny] + yi[1:ny+1] )
            zc = 0.5 * ( zi[0:nz] + zi[1:nz+1] )

            # save
            self.nx, self.ny, self.nz = nx, ny, nz
            self.xi = xi
            self.yi = yi
            self.zi = zi
            self.x = xc
            self.y = yc
            self.z = zc

        elif 100 <= coordsys < 200:
            # spherical
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

            # save
            self.nr, self.ntheta, self.nphi = nr, ntheta, nphi
            self.ri        = ri
            self.thetai    = thetai
            self.phii      = phii
            self.r         = rc
            self.theta     = thetac
            self.phi       = phic

            # get grid
            #qq           = np.meshgrid(rc,thetac,phic,indexing='ij') # (r, theta, phi) in the spherical coordinate
            #rr, tt, phph = qq
            #zr           = 0.5*np.pi - tt # angle from z axis (90deg - theta)
            #rxy          = rr*np.sin(tt)  # r in xy-plane
            #zz           = rr*np.cos(tt)  # z in xyz coordinate

        else:
            print('ERROR\tread_grid: Currently no support for coordsystem >= 200.')
            print('ERROR\tread_grid: Check you input of coordsystem.')
            return 0


# read
def read_grid(f='amr_grid.inp', outpixel='center'):
    # grid
    if os.path.exists(f) == False:
        print ('ERROR\tread_model: amr_grid.inp cannot be found.')
        return 0


    # format
    iformat, grid_style, coordsys, gridinfo = np.genfromtxt(f, max_rows=4, delimiter='\n',dtype=int)
    incl_x, incl_y, incl_z = np.genfromtxt(f, max_rows=1, skip_header=4, delimiter=' ',dtype=int)

    if coordsys < 100:
        # Cartesian

        # dimension
        nxyz             = np.genfromtxt(f, max_rows=1, skip_header=5, delimiter=' ',dtype=int)
        nx, ny, nz = nxyz
        arraysize        = (nx,ny,nz)

        dread            = pd.read_csv(f, skiprows=6, comment='#', encoding='utf-8',header=None)
        coords           = dread.values
        xi, yi, zi = np.split(coords,[nx+1,nx+ny+2])

        # centers of each cell
        xc = 0.5 * ( xi[0:nx] + xi[1:nx+1] ) # centers of each cell
        yc = 0.5 * ( yi[0:ny] + yi[1:ny+1] )
        zc = 0.5 * ( zi[0:nz] + zi[1:nz+1] )

        if outpixel == 'center':
            return xc, yc, zc
        elif outpixel == 'edge':
            return xi, yi, zi
        else:
            print('WARNING\tread_grid: outpixel must be center or edge.')
            print('WARNING\tread_grid: Ignore input value and return pixel centers.')
            return xc, yc, zc
    elif 100 <= coordsys < 200:
        # spherical
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
    else:
        print('ERROR\tread_grid: Currently no support for coordsystem >= 200.')
        print('ERROR\tread_grid: Check you input of coordsystem.')
        return 0


# read dust_temperature.dat
def read_temperature(f='dust_temperature.dat', fgrid = 'amr_grid.inp'):
    '''
    Read a RADMC-3D temperature file.
    '''
    # grid
    if os.path.exists(fgrid) == False:
        print ('ERROR\tread_temperature: amr_grid.inp cannot be found.')
        return
    else:
        nrtp = np.genfromtxt(fgrid, max_rows=1, skip_header=5, 
            delimiter=' ',dtype=int)
        nr, ntheta, nphi = nrtp

    # temperature
    if os.path.exists(f):
        data = pd.read_csv(f, delimiter='\s+', header=None).values
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
    #data = pd.read_csv(infile, comment='!', delimiter='\n', header=None)
    with open(infile, 'r') as f:
        data = f.read()
    data = data.split('\n')
    data = [i for i in data[:-1] if (len(i) >= 1) & (i[0] != '!')]

    # get
    # line name, weight, nlevels
    line, weight, nlevels = data[0:3] #[0].values
    weight  = float(weight)
    nlevels = int(nlevels)

    # energy on each excitation level
    elevels = data[3:3+nlevels] #.values
    elevels = np.array([ elevels[i].split() for i in range(nlevels)])
    lev, EJ, gJ, J = elevels.T
    lev = np.array([ int(lev[i]) for i in range(nlevels)])
    EJ  = np.array([ float(EJ[i]) for i in range(nlevels)])
    gJ  = np.array([ float(gJ[i]) for i in range(nlevels)])
    J   = np.array([ int(J[i]) for i in range(nlevels)])

    # number of transition
    ntrans = data[3+nlevels].strip()
    ntrans = int(ntrans)

    # Einstein A coefficient
    vtrans = data[3+nlevels+1:3+nlevels+1+ntrans] #.values
    vtrans = np.array([vtrans[i].split() for i in range(ntrans)])

    itrans, Jup, Jlow, Acoeff, freq, delE = vtrans.T
    itrans = np.array([ int(itrans[i]) for i in range(ntrans)])
    Jup    = np.array([ int(Jup[i]) for i in range(ntrans)])
    Jlow   = np.array([ int(Jlow[i]) for i in range(ntrans)])
    Acoeff = np.array([ float(Acoeff[i]) for i in range(ntrans)])
    freq   = np.array([ float(freq[i]) for i in range(ntrans)])
    delE   = np.array([ float(delE[i]) for i in range(ntrans)])

    # transitions
    trans = [ str(J[ int(Jup[i] - 1)]) + '-' \
    + str( J[ int(Jlow[i] - 1)]) for i in range(len(itrans))]

    return line, weight, nlevels, EJ, gJ, J, ntrans, trans, Jup, Jlow, Acoeff, freq, delE


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

