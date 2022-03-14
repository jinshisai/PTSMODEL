import numpy as np
import pandas as pd


### functions
# read file
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

    return line, weight, nlevels, EJ, gJ, J, ntrans, Jup, Acoeff, freq, delE


def image_contsub(line, iline, filehead='image_'):
    '''
    Continuum subtraction.
    '''
    # image files
    f_line = filehead + '%s%i%i.out'%(self.line, iline, iline-1)
    f_cont = filehead + '%s%i%i_cont.out'%(self.line, iline, iline-1)

    # read file
    # line
    nx, ny = np.genfromtxt(f_line, delimiter='     ',max_rows=1, skip_header=1, dtype=int)
    nchan  = np.genfromtxt(f_line, max_rows=1, skip_header=2, dtype=int)
    d_line = pd.read_csv(f_line, comment='#', encoding='utf-8',
        header=None, dtype=float, skiprows=5+nchan)
    im_line = d_line.values.values
    im_line = im_line.reshape((1, nchan, ny, nx))#,order='F')
    # cont
    d_cont = pd.read_csv(f_cont, comment='#', encoding='utf-8',
        header=None, dtype=float, skiprows=5+1)
    im_cont   = d_cont.values.values
    im_cont   = im_cont.reshape((1, 1, npix, npix))#,order='F')

    # contsub
    im_line_contsub = im_line - im_cont

    # save file
    with open(f_line.replace('.out', '_contsub.out'), 'w') as f:
        f_i    = open(f_line, 'r')
        header = f_i.readlines()[0:5+nchan]
        header = ''.join(header)
        f.write(header)
        f.write('\n')
        d_out = im_line_contsub.ravel(order='F') # Create a 1-D view, fortran-style indexing
        np.savetxt(f,d_out.T,fmt=['%13.6e'])

    return im_line_contsub

