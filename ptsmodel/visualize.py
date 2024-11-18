import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import ptsmodel



### constants
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
ms  = 1.98892e33     # Solar mass              [g]
ts  = 5.78e3         # Solar temperature       [K]
ls  = 3.8525e33      # Solar luminosity        [erg/s]
rs  = 6.96e10        # Solar radius            [cm]


def change_aspect_ratio(ax, ratio, plottype='linear'):
    '''
    This function change aspect ratio of figure.
    Parameters:
        ax: ax (matplotlit.pyplot.subplots())
            Axes object
        ratio: float or int
            relative x axis width compared to y axis width.
    '''
    if plottype == 'linear':
        aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    elif plottype == 'loglog':
        aspect = (1/ratio) *(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])) / (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    elif plottype == 'linearlog':
        aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / np.log10(ax.get_ylim()[1]/ax.get_ylim()[0])
    elif plottype == 'loglinear':
        aspect = (1/ratio) *(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    else:
        print('ERROR\tchange_aspect_ratio: plottype must be choosen from the types below.')
        print('   plottype can be linear or loglog.')
        print('   plottype=loglinear and linearlog is being developed.')
        return

    aspect = np.abs(aspect)
    aspect = float(aspect)
    ax.set_aspect(aspect)


# functions
def dust_density(model, outname = None,
	rho_range = None, xlim = None, ylim = None,
	rlim = None, zlim = None,
	figsize = (11.69,8.27), cmap='coolwarm',
	fontsize = 14, wspace = 0.4, hspace = 0.2,
	cbaroptions = ['right', '3%', '3%'], cbarlabel = None,
	drange = 1.e-5):
	'''
	Visualize density distribution as 2-D slices.

	Args:
	    rho_d_range:
	    nrho_g_range:
	'''

	# check input
	if type(model) == ptsmodel.PTSMODEL:
		pass
	else:
		print ("ERROR\tvisualize: input must be PTSMODEL object.")

	# setting for figures
	#plt.rcParams['font.family'] ='Arial'    # font (Times New Roman, Helvetica, Arial)
	plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
	plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
	plt.rcParams['font.size'] = fontsize    # fontsize

	# read model
	# dimension
	nr, ntheta, nphi = model.gridshape

	# edge of cells
	#  cuz the plot method, pcolormesh, requires the edge of each cell
	ri     = model.ri
	thetai = model.thetai
	phii   = model.phii
	theta_c = (thetai[0:ntheta] + thetai[1:ntheta+1])*0.5 # cell center
	# cylindarical
	rr, tt, phph = np.meshgrid(ri, thetai, phii, indexing='ij')
	rxy = rr*np.sin(tt)      # radius in xy-plane, r*sin(theta)
	zz  = rr*np.cos(tt)      # z, r*cos(theta)
	# Cartesian
	xx = rxy*np.cos(phph)
	yy = rxy*np.sin(phph)
	indx_mid = np.argmin(np.abs(theta_c - np.pi*0.5)) # mid-plane
	# density
	rho_d  = model.rho_d

	# for plot
	rho_d[np.where(rho_d <= 0.)] = np.nan
	rho_range = rho_range if rho_range is not None \
	else [np.nanmax(rho_d) * drange, np.nanmax(rho_d)]
	cbarlabel = r'$\rho_\mathrm{dust}\ \mathrm{(g\ cm^{-3})}$' if cbarlabel is None \
	else cbarlabel

	xlim = xlim if xlim is not None else [np.nanmin(xx)/au, np.nanmax(xx)/au]
	ylim = ylim if ylim is not None else [np.nanmin(yy)/au, np.nanmax(yy)/au]
	rlim = rlim if rlim is not None else [np.nanmin(rr)/au, np.nanmax(rr)/au]
	zlim = zlim if zlim is not None else [np.nanmin(zz)/au, np.nanmax(zz)/au]


	# dust disk
	fig = plt.figure(figsize=figsize)
	if nphi <= 1:
		ax1 = fig.add_subplot(111)
		cbarlabel1 = cbarlabel
	else:
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)
		cbarlabel1 = ''

	# r-z plot
	colorplot(rxy[:,:,nphi//2]/au, 
		zz[:,:,nphi//2]/au, 
		rho_d[:,:,nphi//2], ax = ax1,
		xlim = rlim, ylim = zlim, dlim = rho_range,
		cmap = cmap, colorscale = 'log', xlabel = r'$R$ (au)',
		ylabel = r'$z$ (au)', cbarlabel = cbarlabel1)
	ax1.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
	ax1.set_aspect(1)

	# x-y plot
	if nphi > 1:
		colorplot(rxy[:,indx_mid,:]/au, 
		zz[:,indx_mid,:]/au, 
		rho_d[:,indx_mid,:], ax = ax2,
		xlim = xlim, ylim = ylim, dlim = rho_range,
		cmap = cmap, colorscale = 'log', xlabel = r'$x$ (au)',
		ylabel = r'$y$ (au)', cbarlabel = cbarlabel)
	ax2.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
	ax2.set_aspect(1)

	# save figures
	fig.subplots_adjust(wspace=wspace, hspace=hspace)
	if outname:
		fig.savefig(outname + '.pdf', transparent=True)
	else:
		fig.savefig('dust_density.pdf', transparent=True)


def gas_density(model, outname = None, 
	nrho_range = None, xlim = None, ylim = None,
	rlim = None, zlim = None,
	figsize=(11.69,8.27), cmap='coolwarm',
	fontsize=14, wspace=0.4, hspace=0.2, imol=0,
	cbaroptions = ['right', '3%', '3%'], cbarlabel = None,
	drange = 1.e-5):
	'''
	Visualize density distribution as 2-D slices.

	Args:
	    rho_d_range:
	    nrho_g_range:
	'''

	# check input
	if type(model) == ptsmodel.PTSMODEL:
		pass
	else:
		print ("ERROR\tvisualize: input must be PTSMODEL object.")

	# setting for figures
	#plt.rcParams['font.family'] ='Arial'    # font (Times New Roman, Helvetica, Arial)
	plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
	plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
	plt.rcParams['font.size'] = fontsize    # fontsize

	# read model
	# dimension
	nr, ntheta, nphi = model.gridshape

	# edge of cells
	#  cuz the plot method, pcolormesh, requires the edge of each cell
	ri     = model.ri
	thetai = model.thetai
	phii   = model.phii
	theta_c = (thetai[0:ntheta] + thetai[1:ntheta+1])*0.5 # cell center

	rr, tt, phph = np.meshgrid(ri, thetai, phii, indexing='ij')
	rxy = rr*np.sin(tt)      # radius in xy-plane, r*sin(theta)
	zz  = rr*np.cos(tt)      # z, r*cos(theta)
	xx = rxy*np.cos(phph)
	yy = rxy*np.sin(phph)
	indx_mid = np.argmin(np.abs(theta_c - np.pi*0.5)) # mid-plane
	# density
	nrho_g = model.nrho_g[model.line[imol]]


	# for plot
	nrho_g[np.where(nrho_g <= 0.)] = np.nan
	nrho_range = nrho_range if nrho_range is not None \
	else [np.nanmax(nrho_g) * drange, np.nanmax(nrho_g)]
	cbarlabel = r'$n_\mathrm{%s}\ \mathrm{(cm^{-3})}$'%model.line[imol] if cbarlabel is None \
	else cbarlabel

	xlim = xlim if xlim is not None else [np.nanmin(xx)/au, np.nanmax(xx)/au]
	ylim = ylim if ylim is not None else [np.nanmin(yy)/au, np.nanmax(yy)/au]
	rlim = rlim if rlim is not None else [np.nanmin(rr)/au, np.nanmax(rr)/au]
	zlim = zlim if zlim is not None else [np.nanmin(zz)/au, np.nanmax(zz)/au]


	# dust disk
	fig = plt.figure(figsize=figsize)
	if nphi <= 1:
		ax1 = fig.add_subplot(111)
		cbarlabel1 = cbarlabel
	else:
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)
		cbarlabel1 = ''

	# r-z plot
	colorplot(rxy[:,:,nphi//2]/au, 
		zz[:,:,nphi//2]/au, 
		nrho_g[:,:,nphi//2], ax = ax1,
		xlim = rlim, ylim = zlim, dlim = nrho_range,
		cmap = cmap, colorscale = 'log', xlabel = r'$R$ (au)',
		ylabel = r'$z$ (au)', cbarlabel = cbarlabel1)
	ax1.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
	ax1.set_aspect(1)

	# x-y plot
	if nphi > 1:
		print(indx_mid, nrho_g[:,indx_mid,:].ravel())
		colorplot(xx[:,indx_mid,:]/au, 
		yy[:,indx_mid,:]/au, 
		nrho_g[:,indx_mid,:], ax = ax2,
		xlim = xlim, ylim = ylim, dlim = nrho_range,
		cmap = cmap, colorscale = 'log', xlabel = r'$x$ (au)',
		ylabel = r'$y$ (au)', cbarlabel = cbarlabel)
		ax2.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
		ax2.set_aspect(1)

		fig.subplots_adjust(wspace=wspace, hspace=hspace)

	if outname:
		fig.savefig(outname + '.pdf', transparent=True)
	else:
		fig.savefig('gas_density.pdf',transparent=True)
	plt.close()


def density_profile(model, 
	outname = None, 
	rho_range = None, rlim = None,
	figsize=(11.69,8.27),
	fontsize=14, wspace=0.4, hspace=0.2, imol=0,
	cbaroptions = ['right', '3%', '3%'], cbarlabel = None,
	drange = 1.e-5, kind = 'gas'):
	'''
	Visualize density distribution as 2-D slices.

	Args:
	    rho_d_range:
	    nrho_g_range:
	'''

	# check input
	if type(model) == ptsmodel.PTSMODEL:
		pass
	else:
		print ("ERROR\tvisualize: input must be PTSMODEL object.")

	# setting for figures
	#plt.rcParams['font.family'] ='Arial'    # font (Times New Roman, Helvetica, Arial)
	plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
	plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
	plt.rcParams['font.size'] = fontsize    # fontsize

	# read model
	# dimension
	nr, ntheta, nphi = model.gridshape

	# grid
	r = model.r
	theta = model.theta
	indx_mid = np.argmin(np.abs(theta - np.pi*0.5)) # mid-plane
	# density
	if kind == 'gas':
		rho = model.nrho_g[model.line[imol]][:,indx_mid, nphi//2]
		ylabel = r'$n_\mathrm{%s}\ \mathrm{(cm^{-3})}$'%model.line[imol]
	elif kind == 'dust':
		rho = model.rho_d[:,indx_mid, nphi//2]
		ylabel = r'$\rho_\mathrm{dust}\ \mathrm{(g\ cm^{-3})}$'
	else:
		print('WARNING\tdensity_profile: input type is wrong.')
		print('WARNING\tdensity_profile: type must be gas or dust.')
		print('WARNING\tdensity_profile: ignore input and plot gas density profile.')
		rho = model.nrho_g[model.line[imol]][:,indx_mid, nphi//2]
		kind = 'gas'


	# for plot
	rho_range = rho_range if rho_range is not None \
	else [np.nanmax(rho) * drange, np.nanmax(rho) * 1.2]

	rlim = rlim if rlim is not None else [np.nanmin(r) / au, np.nanmax(r) / au]

	# dust disk
	fig = plt.figure(figsize=figsize)
	ax1 = fig.add_subplot(111)

	# r-z plot
	ax1.plot(r/au, rho, color = 'k', ls= '-', lw = 1.)
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.set_xlim(rlim[0], rlim[1])
	ax1.set_ylim(rho_range[0], rho_range[1])
	ax1.set_xlabel(r'$R$ (au)')
	ax1.set_ylabel(ylabel)
	ax1.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
	change_aspect_ratio(ax1, 1, plottype = 'loglog')

	if outname:
		fig.savefig(outname + '.pdf', transparent=True)
	else:
		fig.savefig(kind + '_density_profile_%s.pdf'%model.line[imol],transparent=True)
	#plt.show()
	plt.close()


# plot temperature profile
def temperature(model, infile='dust_temperature.dat',
	outname = None,
	t_range = None, xlim = None, ylim = None, rlim = None, zlim = None, 
	figsize=(11.69,8.27), 
	cmap='coolwarm', fontsize=14, wspace=0.4, hspace=0.2, 
	cbaroptions = ['right', '3%', '3%'], cbarlabel = None,
	clevels = [10,20,30,40,50,60]):
	'''
	Plot temperature profile.

	Args:
	'''

	# check input
	if type(model) == ptsmodel.PTSMODEL:
		pass
	else:
		print ("ERROR\ttemperature: input must be PTSMODEL object.")


	# setting for figures
	#plt.rcParams['font.family'] ='Arial'    # font (Times New Roman, Helvetica, Arial)
	plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
	plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
	plt.rcParams['font.size'] = fontsize    # fontsize

	# read model
	nr, ntheta, nphi = model.gridshape

	# edge of cells
	#  cuz the plot method, pcolormesh, requires the edge of each cell
	ri     = model.ri
	thetai = model.thetai
	phii   = model.phii
	theta_c = (thetai[0:ntheta] + thetai[1:ntheta+1])*0.5 # cell center
	# Cylindarical
	rr, tt, phph = np.meshgrid(ri, thetai, phii, indexing='ij')
	rxy = rr*np.sin(tt)      # radius in xy-plane, r*sin(theta)
	zz  = rr*np.cos(tt)      # z, r*cos(theta)
	# Cartesian
	xx = rxy*np.cos(phph)
	yy = rxy*np.sin(phph)
	# mid-plane
	indx_mid = np.argmin(np.abs(theta_c - np.pi*0.5)) # mid-plane

	# for contour plot
	rr_c, tt_c, phph_c = np.meshgrid(model.r, model.theta, model.phi, indexing='ij')
	rxy_c = rr_c*np.sin(tt_c)      # radius in xy-plane, r*sin(theta)
	zz_c  = rr_c*np.cos(tt_c)      # z, r*cos(theta)
	# Cartesian
	xx_c = rxy_c*np.cos(phph_c)
	yy_c = rxy_c*np.sin(phph_c)


	# read file
	if os.path.exists(infile):
	    pass
	else:
	    print ('ERROR: Cannot find %s'%infile)
	    return

	data = pd.read_csv(infile, delimiter='\s+', header=None).values
	iformat = data[0,0]
	imsize  = data[1,0]
	ndspc   = data[2,0]
	temp    = data[3:,0]

	#retemp = temp.reshape((nr,ntheta,nphi))
	retemp = temp.reshape((nphi,ntheta,nr)).T


	# for plot
	t_range = t_range if t_range is not None \
	else [0., np.nanmax(temp)]
	cbarlabel = r'$T\ \mathrm{(K)}$' if cbarlabel is None \
	else cbarlabel

	xlim = xlim if xlim is not None else [np.nanmin(xx)/au, np.nanmax(xx)/au]
	ylim = ylim if ylim is not None else [np.nanmin(yy)/au, np.nanmax(yy)/au]
	rlim = rlim if rlim is not None else [np.nanmin(rr)/au, np.nanmax(rr)/au]
	zlim = zlim if zlim is not None else [np.nanmin(zz)/au, np.nanmax(zz)/au]


	# plot
	fig = plt.figure(figsize=figsize)
	if nphi <= 1:
		ax1 = fig.add_subplot(111)
		cbarlabel1 = cbarlabel
	else:
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)
		cbarlabel1 = ''

	# r-z plot
	colorplot(rxy[:,:,nphi//2]/au, 
		zz[:,:,nphi//2]/au, 
		retemp[:,:,nphi//2], ax = ax1,
		xlim = rlim, ylim = zlim, dlim = t_range,
		cmap = cmap, colorscale = 'linear', xlabel = r'$R$ (au)',
		ylabel = r'$z$ (au)', cbarlabel = cbarlabel1)
	ax1.contour(rxy_c[:,:,nphi//2]/au, zz_c[:,:,nphi//2]/au, retemp[:,:,nphi//2], 
		levels = clevels, colors = 'white', linewidths = 1.)
	ax1.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
	ax1.set_aspect(1)

	# x-y plot
	if nphi > 1:
		# color
		colorplot(rxy[:,indx_mid,:]/au, 
		zz[:,indx_mid,:]/au, 
		retemp[:,indx_mid,:], ax = ax2,
		xlim = xlim, ylim = ylim, dlim = t_range,
		cmap = cmap, colorscale = 'linear', xlabel = r'$x$ (au)',
		ylabel = r'$y$ (au)', cbarlabel = cbarlabel)
		# contour
		ax2.contour(rxy_c[:,indx_mid,:]/au, zz_c[:,indx_mid,:]/au, retemp[:,indx_mid,:], 
			levels = clevels, colors = 'white', linewidths = 1.)
		# ticks
		ax2.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
		ax2.set_aspect(1)

		fig.subplots_adjust(wspace=wspace, hspace=hspace)

	if outname is not None:
		fig.savefig(outname + '.pdf', transparent = True)
	else:
		fig.savefig('dust_temperature.pdf', transparent=True)
	return fig


# plot temperature profile
def plot_temperature_xy(model, infile='dust_temperature.dat', fig=None, ax=None,
	t_range=[], x_range=[], y_range=[], figsize=(8.27, 8.27), cmap='coolwarm',
	fontsize=14, clevels=[10,20,30,40,50,60],
	aspect=1., shrink=None, savefig=True, imol=0):
	'''
	Plot temperature profile.

	Args:
	'''

	# check input
	if type(model) == ptsmodel.PTSMODEL:
		pass
	else:
		print ("ERROR\tvisualize: input must be PTSMODEL object.")


	# setting for figures
	#plt.rcParams['font.family'] ='Arial'    # font (Times New Roman, Helvetica, Arial)
	plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
	plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
	plt.rcParams['font.size'] = fontsize    # fontsize

	# read model
	nr, ntheta, nphi = model.gridshape

	# edge of cells
	#  cuz the plot method, pcolormesh, requires the edge of each cell
	ri     = model.ri
	thetai = model.thetai
	phii   = model.phii
	theta_c = (thetai[0:ntheta] + thetai[1:ntheta+1])*0.5 # cell center

	rr, tt, phph = np.meshgrid(ri, thetai, phii, indexing='ij')
	rxy = rr*np.sin(tt)      # radius in xy-plane, r*sin(theta)
	zz  = rr*np.cos(tt)      # z, r*cos(theta)

	rho_d  = model.rho_d
	nrho_g = model.nrho_g[model.line[imol]]

	xx = rxy*np.cos(phph)
	yy = rxy*np.sin(phph)


	# read file
	if os.path.exists(infile):
	    pass
	else:
	    print ('ERROR: Cannot find %s'%infile)
	    return

	data = pd.read_csv(infile, delimiter='\s+', header=None).values
	iformat = data[0,0]
	imsize  = data[1,0]
	ndspc   = data[2,0]
	temp    = data[3:,0]

	#retemp = temp.reshape((nr,ntheta,nphi))
	retemp = temp.reshape((nphi,ntheta,nr)).T


	# setting for figure
	x_range = x_range if len(x_range) == 2 else [-np.nanmax(rr)/au, np.nanmax(rr)/au]
	y_range = y_range if len(y_range) == 2 else [-np.nanmax(rr)/au, np.nanmax(rr)/au]
	t_range = t_range if len(t_range) == 2 else [0., np.nanmax(temp)]


	# figure
	if fig:
		pass
	else:
		fig = plt.figure(figsize=figsize)
		ax = fig.add_subplot(111)
		#divider = make_axes_locatable(ax)
		#cax1    = divider.append_axes('right', '3%', pad='0%')

	# plot
	indx_mid = np.argmin(np.abs(theta_c - np.pi*0.5)) # mid-plane
	im = ax.pcolormesh(xx[:,indx_mid,:]/au, yy[:,indx_mid,:]/au, 
		retemp[:,indx_mid,:], cmap=cmap, vmin = t_range[0], vmax=t_range[1], 
		rasterized=True)

	shrink = shrink if shrink else aspect*0.8
	cbar = fig.colorbar(im, ax=ax, shrink=shrink, pad=0.03)
	ax.set_xlabel(r'$x$ (au)')
	ax.set_ylabel(r'$y$ (au)')
	cbar.set_label(r'$T_\mathrm{dust}\ \mathrm{(K)}$')
	ax.tick_params(which='both', direction='in',
		bottom=True, top=True, left=True, right=True)

	ax.set_xlim(*x_range)
	ax.set_ylim(*y_range)
	#ax.set_aspect(aspect)
	change_aspect_ratio(ax, aspect)

	if savefig:
		#fig.subplots_adjust(wspace=wspace, hspace=hspace)
		fig.savefig('dust_temperature_xy.pdf', transparent=True)
	return fig



def plot_temperature_rz(model, infile='dust_temperature.dat', fig=None, ax=None,
	t_range=[], r_range=[], z_range=[], figsize=(8.27, 8.27), cmap='coolwarm',
	fontsize=14, clevels=[10,20,30,40,50,60],
	aspect=1., shrink=None, savefig=True, imol=0):
	'''
	Plot temperature profile.

	Args:
	'''

	# check input
	if type(model) == ptsmodel.PTSMODEL:
		pass
	else:
		print ("ERROR\tvisualize: Input must be PTSMODEL object.")


	# setting for figures
	#plt.rcParams['font.family'] ='Arial'    # font (Times New Roman, Helvetica, Arial)
	plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
	plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
	plt.rcParams['font.size'] = fontsize    # fontsize

	# read model
	nr, ntheta, nphi = model.gridshape

	# edge of cells
	#  cuz the plot method, pcolormesh, requires the edge of each cell
	ri     = model.ri
	thetai = model.thetai
	phii   = model.phii
	theta_c = (thetai[0:ntheta] + thetai[1:ntheta+1])*0.5 # cell center

	rr, tt, phph = np.meshgrid(ri, thetai, phii, indexing='ij')
	rxy = rr*np.sin(tt)      # radius in xy-plane, r*sin(theta)
	zz  = rr*np.cos(tt)      # z, r*cos(theta)

	rho_d  = model.rho_d
	nrho_g = model.nrho_g[model.line[imol]]

	xx = rxy*np.cos(phph)
	yy = rxy*np.sin(phph)


	# read file
	if os.path.exists(infile):
	    pass
	else:
	    print ('ERROR: Cannot find %s'%infile)
	    return

	data = pd.read_csv(infile, delimiter='\n', header=None).values
	iformat = data[0]
	imsize  = data[1]
	ndspc   = data[2]
	temp    = data[3:]

	#retemp = temp.reshape((nr,ntheta,nphi))
	retemp = temp.reshape((nphi,ntheta,nr)).T


	# setting for figure
	r_range = r_range if len(r_range) == 2 else [np.nanmin(rr)/au, np.nanmax(rr)/au]
	t_range = t_range if len(t_range) == 2 else [0., np.nanmax(temp)]
	z_range = z_range if len(z_range) == 2 else [0., r_range[1]]


	# figure
	if fig:
		pass
	else:
		fig = plt.figure(figsize=figsize)
		ax = fig.add_subplot(111)
		#divider = make_axes_locatable(ax1)
		#cax1    = divider.append_axes('right', '3%', pad='0%')

	# plot
	im1   = ax.pcolormesh(rxy[:,:,nphi//2]/au, zz[:,:,nphi//2]/au,
	 retemp[:,:,nphi//2], cmap=cmap, 
	 vmin = t_range[0], vmax=t_range[1], rasterized=True)

	rxy_cont = (rxy[:nr, :ntheta, nphi//2] + rxy[1:nr+1,1:ntheta+1,nphi//2])*0.5
	zz_cont = (zz[:nr,:ntheta,nphi//2] + zz[1:nr+1,1:ntheta+1,nphi//2])*0.5
	im11  = ax.contour(rxy_cont/au, zz_cont/au,
	 retemp[:,:,nphi//2], colors='white', levels=clevels, linewidths=1.)

	shrink = shrink if shrink else aspect*0.8
	cbar = fig.colorbar(im1, ax=ax, shrink=shrink, pad=0.03)
	ax.set_xlabel(r'$r$ (au)')
	ax.set_ylabel(r'$z$ (au)')
	cbar.set_label(r'$T_\mathrm{dust}\ \mathrm{(K)}$')

	ax.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True)
	ax.set_xlim(*r_range)
	ax.set_ylim(*z_range)
	#ax1.set_aspect(aspect)
	change_aspect_ratio(ax, aspect)

	#fig.subplots_adjust(wspace=wspace, hspace=hspace)
	if savefig:
		fig.savefig('dust_temperature_rz.pdf', transparent=True)
	return fig


def colorplot(x, y, d, 
	xlim = None, ylim = None, dlim = None,
	fig = None, ax = None, iaxis = 0, figsize = None,
	colorscale = 'linear', cmap = 'coolwarm', norm = None,
	colorbar = True, cbaroptions = ['right', '3%', '0%'],
	cbarlabel = '', xlabel = '(au)', ylabel = '(au)'):
	# figure
	if (fig is not None) & (ax is None):
		ax = fig.axes[iaxis]
	elif ax is not None:
		pass
	else:
		fig = plt.figure(figsize)
		ax = fig.add_subplot(111)

	# setting for figure
	xlim = xlim if xlim is not None else [np.nanmin(x), np.nanmax(x)]
	ylim = ylim if ylim is not None else [np.nanmin(y), np.nanmax(y)]
	dlim = dlim if dlim is not None else [np.nanmin(d), np.nanmax(d)]

	if norm is None:
		if colorscale == 'log':
			norm = colors.LogNorm(vmin = dlim[0], vmax=dlim[1])
		else:
			norm = colors.Normalize(vmin=dlim[0], vmax=dlim[1])


	im = ax.pcolormesh(x, y, d,
		norm = norm, cmap = cmap, rasterized = True)
	if colorbar:
		cax, cbar = add_colorbar_toaxis(im, ax, cbarlabel = cbarlabel, 
			cbaroptions = cbaroptions)

	ax.set_xlim(xlim[0], xlim[1])
	ax.set_ylim(ylim[0], ylim[1])

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	return im, ax



def add_colorbar_toaxis(
	cim, ax,
    cbarlabel: str='', 
    cbaroptions: list = ['right', '3%', '0%'],
    ticks: list = None,
    tickcolor: str = 'k', 
    axiscolor: str = 'k', 
    labelcolor: str = 'k'):
    # parameter
    orientations = {
    'right': 'vertical',
    'left': 'vertical',
    'top': 'horizontal',
    'bottom': 'horizontal'}

    # setting for a color bar
    if len(cbaroptions) == 3:
        cbar_loc, cbar_wd, cbar_pad = cbaroptions
    elif len(cbaroptions) == 4:
        cbar_loc, cbar_wd, cbar_pad, cbarlabel = cbaroptions
    else:
        print('WARNING\tadd_colorbar_toaxis: cbaroptions must have three or four elements. \
        Input is ignored.')
    cbar_loc, cbar_wd, cbar_pad = cbaroptions

    # inset axes
    if cbar_loc == 'right':
        width = cbar_wd
        height = '100%'
        bbox_to_anchor = (1. + float(cbar_pad.strip('%'))*0.01, 0., 1., 1.)
    elif cbar_loc == 'top':
        width = '100%'
        height = cbar_wd
        bbox_to_anchor = (0., 1. + float(cbar_pad.strip('%'))*0.01, 1., 1.)
    else:
        print("ERROR\tadd_colorbar: cbar_loc must be 'right' or 'top'.")
        return 0
    cax = inset_axes(ax,
        width = width,
        height = height,
        loc = 'lower left',
        bbox_to_anchor = bbox_to_anchor,
        bbox_transform = ax.transAxes,
        borderpad = 0.)
    # add a color bar
    cbar = plt.colorbar(cim, cax=cax, ticks=ticks, 
        orientation=orientations[cbar_loc], ticklocation=cbar_loc)
    cbar.set_label(cbarlabel)
    return cax, cbar



def gasdensity3d_faceon(model, step=1,
 nrho_g_min=None,  nrho_g_max=None, xlim=[], ylim=[]):

	# check input
	if type(model) == ptsmodel.PTSMODEL:
		pass
	else:
		print ("ERROR\tvisualize: input must be PTSMODEL object.")

	# grid
	rr, tt, phph = model.grid
	rxy = rr*np.sin(tt)      # r in xy-plane
	zz  = rr*np.cos(tt)      # z in xyz coordinate
	xx  = rxy*np.cos(phph)   # x in xyz coordinate
	yy  = rxy*np.sin(phph)   # y in xyz coordinate


	# density
	#rho_d = model.rho_d
	rho_g = model.rho_g
	nrho_g = model.nrho_g


	# velocity
	vr     = model.vr
	vtheta = model.vtheta
	vphi   = model.vphi

	# vx, vy, vz
	vx = vr*np.sin(tt)*np.cos(phph) + vtheta*np.cos(tt)*np.cos(phph) - vphi*np.sin(phph)
	vy = vr*np.sin(tt)*np.sin(phph) + vtheta*np.cos(tt)*np.sin(phph) + vphi*np.cos(phph)
	vz = vr*np.cos(tt) - vtheta*np.sin(tt)
	v_scalar = np.sqrt(vr*vr + vtheta*vtheta + vphi*vphi)


	# 0 --> nan
	vx[np.where(nrho_g <= 0.)] = np.nan
	vy[np.where(nrho_g <= 0.)] = np.nan
	vz[np.where(nrho_g <= 0.)] = np.nan
	nrho_g[np.where(nrho_g <= 0.)] = np.nan


	# sampling for beauty
	#vr = vr[::step,::step,::step]
	#vtheta = vtheta[::step,::step,::step]
	#vphi   = vphi[::step,::step,::step]
	vx     = vx[::step,::step,::step]
	vy     = vy[::step,::step,::step]
	xx_smpl = xx[::step,::step,::step]
	yy_smpl = yy[::step,::step,::step]
	zz_smpl = zz[::step,::step,::step]



	# for plot
	nrho_g_min = nrho_g_min if nrho_g_min else np.nanmin(nrho_g)
	nrho_g_max = nrho_g_max if nrho_g_max else np.nanmax(nrho_g)
	#print (nrho_g_min, nrho_g_max)

	xlim = xlim if len(xlim) == 2 else [np.nanmin(xx)/au, np.nanmax(xx)/au]
	ylim = ylim if len(ylim) == 2 else [np.nanmin(yy)/au, np.nanmax(yy)/au]


	# plot
	fig = plt.figure(figsize=(11.69,8.27))
	ax  = fig.add_subplot(111)

	# density
	im = ax.scatter(xx.ravel()/au, yy.ravel()/au, c=nrho_g.ravel(),
		norm = colors.LogNorm(vmin = nrho_g_min, vmax=nrho_g_max), cmap=cm.coolwarm,
		rasterized=True, alpha=0.7)

	# vector
	ax.quiver(xx_smpl.ravel()/au, yy_smpl.ravel()/au, vx.ravel(), vy.ravel(),
		color='k', angles='uv') #scale=vscale, width=width, units='xy'

	ax.tick_params(which='both', direction='in',bottom=True, top=True,
	 left=True, right=True, pad=9.)
	ax.set_aspect(1)
	#ax.set_aspect('equal')
	#ax.set_box_aspect((1,1,1))

	ax.set_xlabel('x (au)')
	ax.set_ylabel('y (au)')

	ax.set_xlim(xlim[0], xlim[1])
	ax.set_ylim(ylim[0], ylim[1])

	# color bar
	divider = make_axes_locatable(ax)
	cax  = divider.append_axes("right", size="3%", pad=0.)
	cbar = fig.colorbar(im, cax=cax, label=r'$n_\mathrm{mol}$ (cm$^{-3}$)')#, pad=0., aspect=30., shrink=3./7.)

	print ('saving plots...')
	plt.savefig('gas_density_3d-faceon.pdf',transparent=True, dpi=120)
	print ('saved.')
	plt.close()


# functions
def gasdensity3d_obconfg(model, step=1,
 nrho_g_min=None,  nrho_g_max=None, xlim=[], ylim=[], zlim=[], inc=None):

	# check input
	if type(model) == ptsmodel.PTSMODEL:
		pass
	else:
		print ("ERROR\tvisualize: input must be PTSMODEL object.")

	# params
	phiob = 270. # from minus along y-axis
	if inc:
		inc = inc
	else:
		try:
			inc = self.inc
		except:
			print ('WARNING\tgasdensity3d_obconfg: No readable inclination angle.\
			 inc = 0 is assumed')
			inc = 0.

	# grid
	rr, tt, phph = model.grid
	rxy = rr*np.sin(tt)      # r in xy-plane
	zz  = rr*np.cos(tt)      # z in xyz coordinate
	xx  = rxy*np.cos(phph)   # x in xyz coordinate
	yy  = rxy*np.sin(phph)   # y in xyz coordinate


	# density
	#rho_d = model.rho_d
	rho_g = model.rho_g
	nrho_g = model.nrho_g


	# velocity
	vr     = model.vr
	vtheta = model.vtheta
	vphi   = model.vphi

	# vx, vy, vz
	vx = vr*np.sin(tt)*np.cos(phph) + vtheta*np.cos(tt)*np.cos(phph) - vphi*np.sin(phph)
	vy = vr*np.sin(tt)*np.sin(phph) + vtheta*np.cos(tt)*np.sin(phph) + vphi*np.cos(phph)
	vz = vr*np.cos(tt) - vtheta*np.sin(tt)
	v_scalar = np.sqrt(vr*vr + vtheta*vtheta + vphi*vphi)


	# 0 --> nan
	vx[np.where(nrho_g <= 0.)] = np.nan
	vy[np.where(nrho_g <= 0.)] = np.nan
	vz[np.where(nrho_g <= 0.)] = np.nan
	nrho_g[np.where(nrho_g <= 0.)] = np.nan


	# sampling for beauty
	#vr = vr[::step,::step,::step]
	#vtheta = vtheta[::step,::step,::step]
	#vphi   = vphi[::step,::step,::step]
	vx     = vx[::step,::step,::step]
	vy     = vy[::step,::step,::step]
	vz     = vz[::step,::step,::step]
	xx_smpl = xx[::step,::step,::step]
	yy_smpl = yy[::step,::step,::step]
	zz_smpl = zz[::step,::step,::step]


	# for plot
	nrho_g_min = nrho_g_min if nrho_g_min else np.nanmin(nrho_g)
	nrho_g_max = nrho_g_max if nrho_g_max else np.nanmax(nrho_g)
	#print (nrho_g_min, nrho_g_max)

	xlim = xlim if len(xlim) == 2 else [np.nanmin(xx)/au, np.nanmax(xx)/au]
	ylim = ylim if len(ylim) == 2 else [np.nanmin(yy)/au, np.nanmax(yy)/au]
	zlim = zlim if len(zlim) == 2 else [np.nanmin(zz)/au, np.nanmax(zz)/au]


	# plot
	fig = plt.figure(figsize=(11.69,8.27))
	ax  = fig.add_subplot(111, projection='3d')

	# density
	im = ax.scatter(xx.ravel()/au, yy.ravel()/au, zz.ravel()/au, c=nrho_g.ravel(),
		norm = colors.LogNorm(vmin = nrho_g_min, vmax=nrho_g_max), cmap=cm.coolwarm,
		rasterized=True, alpha=0.7)

	# vector
	ax.quiver(xx_smpl.ravel()/au, yy_smpl.ravel()/au, zz_smpl.ravel()/au,
	 vx.ravel(), vy.ravel(), vz.ravel(), color='k', normalize=True, arrow_length_ratio=0.3)# angles='uv', scale=vscale, width=width, units='xy'

	ax.tick_params(which='both', direction='in',bottom=True, top=True,
	 left=True, right=True, pad=9.)
	#ax.set_aspect(1)
	#ax.set_aspect('equal')
	#ax.set_box_aspect((1,1,1))

	ax.set_xlabel('x (au)')
	ax.set_ylabel('y (au)')
	ax.set_zlabel('z (au)')

	ax.set_xlim(xlim[0], xlim[1])
	ax.set_ylim(ylim[0], ylim[1])
	ax.set_ylim(zlim[0], zlim[1])

	ax.view_init(elev=90.+inc, azim=phiob) # elev=0 means edge-on view
	                                       # elev=90. means face-on view

	cbar = fig.colorbar(im, use_gridspec=True)
	print ('saving plots...')
	plt.savefig('gas_density_3d-obconfg.pdf',transparent=True, dpi=120)
	print ('saved.')
	plt.close()
