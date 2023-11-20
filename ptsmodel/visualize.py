import numpy as np
import pandas as pd
import matplotlib
import os
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import ptsmodel



### constants
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
ms  = 1.98892e33     # Solar mass              [g]
ts  = 5.78e3         # Solar temperature       [K]
ls  = 3.8525e33      # Solar luminosity        [erg/s]
rs  = 6.96e10        # Solar radius            [cm]



# functions
def dust_density(model, outname = None,
	rho_range=[], xlim=[], ylim=[],
	figsize=(11.69,8.27), cmap='coolwarm',
	 fontsize=14, wspace=0.4, hspace=0.2):
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
	#rr     = self.rr
	#phph   = self.phph
	#rxy    = self.rxy
	#zz     = self.zz

	rho_d  = model.rho_d

	xx = rxy*np.cos(phph)
	yy = rxy*np.sin(phph)

	# for plot
	rho_d[np.where(rho_d <= 0.)] = np.nan
	rho_range = rho_range if len(rho_range) ==2 else [np.nanmin(rho_d), np.nanmax(rho_d)]

	xlim = xlim if len(xlim) == 2 else [np.nanmin(xx)/au, np.nanmax(xx)/au]
	ylim = ylim if len(ylim) == 2 else [np.nanmin(yy)/au, np.nanmax(yy)/au]


	# dust disk
	fig1 = plt.figure(figsize=figsize)

	# plot 1; density in r vs z
	ax1     = fig1.add_subplot(121)
	divider = make_axes_locatable(ax1)
	cax1    = divider.append_axes('right', '3%', pad='0%')

	im1   = ax1.pcolormesh(rxy[:,:,nphi//2]/au, zz[:,:,nphi//2]/au, rho_d[:,:,nphi//2], cmap=cmap,
	 norm = colors.LogNorm(vmin = rho_range[0], vmax=rho_range[1]), rasterized=True)
	cbar1 = fig1.colorbar(im1, cax=cax1)

	ax1.set_xlabel(r'$r$ (au)')
	ax1.set_ylabel(r'$z$ (au)')
	ax1.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
	ax1.set_aspect(1)


	# plot 2; density in r vs phi (xy-plane)
	ax2     = fig1.add_subplot(122)
	divider = make_axes_locatable(ax2)
	cax2    = divider.append_axes('right', '3%', pad='0%')

	indx_mid = np.argmin(np.abs(theta_c - np.pi*0.5)) # mid-plane
	im2   = ax2.pcolormesh(xx[:,indx_mid,:]/au, yy[:,indx_mid,:]/au, rho_d[:,indx_mid,:],
	 cmap=cmap, norm = colors.LogNorm(vmin = rho_range[0], vmax=rho_range[1]), rasterized=True)
	cbar2 = fig1.colorbar(im2,cax=cax2)

	ax2.set_xlabel(r'$x$ (au)')
	ax2.set_ylabel(r'$y$ (au)')
	cbar2.set_label(r'$\rho_\mathrm{dust}\ \mathrm{(g\ cm^{-3})}$')
	ax2.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
	ax2.set_aspect(1)

	# save figures
	fig1.subplots_adjust(wspace=wspace, hspace=hspace)
	if outname:
		fig1.savefig(outname + '.pdf', transparent=True)
	else:
		fig1.savefig('dust_density.pdf',transparent=True)


def gas_density(model, outname = None, nrho_range=[], xlim=[], ylim=[],
	rlim=[], zlim=[],
	figsize=(11.69,8.27), cmap='coolwarm',
	 fontsize=14, wspace=0.4, hspace=0.2):
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
	#rr     = self.rr
	#phph   = self.phph
	#rxy    = self.rxy
	#zz     = self.zz

	nrho_g = model.nrho_g

	xx = rxy*np.cos(phph)
	yy = rxy*np.sin(phph)

	# for plot
	nrho_g[np.where(nrho_g <= 0.)] = np.nan
	nrho_range = nrho_range if len(nrho_range) == 2 else [np.nanmin(nrho_g), np.nanmax(nrho_g)]

	xlim = xlim if len(xlim) == 2 else [np.nanmin(xx)/au, np.nanmax(xx)/au]
	ylim = ylim if len(ylim) == 2 else [np.nanmin(yy)/au, np.nanmax(yy)/au]
	rlim = rlim if len(rlim) == 2 else [0, np.nanmax(rr)/au]
	zlim = zlim if len(zlim) == 2 else [0, np.nanmax(zz)/au]



	# gas disk
	fig2 = plt.figure(figsize=figsize)

	# plot 1; gas number density in r vs z
	ax3     = fig2.add_subplot(121)
	divider = make_axes_locatable(ax3)
	cax3    = divider.append_axes('right', '3%', pad='0%')

	im3   = ax3.pcolormesh(rxy[:,:,nphi//2]/au, zz[:,:,nphi//2]/au, nrho_g[:,:,nphi//2],
	 cmap=cmap, norm = colors.LogNorm(vmin = nrho_range[0], vmax=nrho_range[1]), rasterized=True)
	cbar3 = fig2.colorbar(im3,cax=cax3)

	ax3.set_xlabel(r'$r$ (au)')
	ax3.set_ylabel(r'$z$ (au)')
	ax3.set_xlim(rlim)
	ax3.set_ylim(zlim)
	#cbar3.set_label(r'$n_\mathrm{gas}\ \mathrm{(cm^{-3})}$')
	ax3.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
	ax3.set_aspect(1)


	# plot 2; density in r vs phi (xy-plane)
	ax4     = fig2.add_subplot(122)
	divider = make_axes_locatable(ax4)
	cax4    = divider.append_axes('right', '3%', pad='0%')

	indx_mid = np.argmin(np.abs(theta_c - np.pi*0.5)) # mid-plane
	im4   = ax4.pcolormesh(xx[:,indx_mid,:]/au, yy[:,indx_mid,:]/au, nrho_g[:,indx_mid,:], cmap=cmap,
	 norm = colors.LogNorm(vmin = nrho_range[0], vmax=nrho_range[1]), rasterized=True)

	cbar4 = fig2.colorbar(im4,cax=cax4)
	ax4.set_xlabel(r'$x$ (au)')
	ax4.set_ylabel(r'$y$ (au)')
	ax4.set_xlim(xlim)
	ax4.set_ylim(ylim)
	cbar4.set_label(r'$n_\mathrm{gas}\ \mathrm{(cm^{-3})}$')
	ax4.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
	ax4.set_aspect(1)


	fig2.subplots_adjust(wspace=wspace, hspace=hspace)
	if outname:
		fig2.savefig(outname + '.pdf', transparent=True)
	else:
		fig2.savefig('gas_density.pdf',transparent=True)
	plt.close()


# plot temperature profile
def plot_temperature(model, infile='dust_temperature.dat',
	t_range=[], r_range=[], figsize=(11.69,8.27), cmap='coolwarm',
	fontsize=14, wspace=0.4, hspace=0.2, clevels=[10,20,30,40,50,60],
	aspect=1.):
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
	nrho_g = model.nrho_g

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


	# figure
	fig = plt.figure(figsize=figsize)

	# plot #1: r-z plane
	ax1     = fig.add_subplot(121)
	divider = make_axes_locatable(ax1)
	cax1    = divider.append_axes('right', '3%', pad='0%')

	# plot
	im1   = ax1.pcolormesh(rxy[:,:,nphi//2]/au, zz[:,:,nphi//2]/au,
	 retemp[:,:,nphi//2], cmap=cmap, vmin = temp_min, vmax=temp_max, rasterized=True)

	rxy_cont = (rxy[:nr,:ntheta,nphi//2] + rxy[1:nr+1,1:ntheta+1,nphi//2])*0.5
	zz_cont = (zz[:nr,:ntheta,nphi//2] + zz[1:nr+1,1:ntheta+1,nphi//2])*0.5
	im11  = ax1.contour(rxy_cont/au, zz_cont/au,
	 retemp[:,:,nphi//2], colors='white', levels=clevels, linewidths=1.)

	cbar1 = fig.colorbar(im1, cax=cax1)
	ax1.set_xlabel(r'$r$ (au)')
	ax1.set_ylabel(r'$z$ (au)')
	#cbar1.set_label(r'$T_\mathrm{dust}\ \mathrm{(K)}$')

	ax1.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
	ax1.set_xlim(0,r_range[0])
	ax1.set_ylim(0,r_range[1])
	ax1.set_aspect(aspect)


	# plot #2: x-y plane
	ax2  = fig.add_subplot(122)
	divider = make_axes_locatable(ax2)
	cax2    = divider.append_axes('right', '3%', pad='0%')

	indx_mid = np.argmin(np.abs(theta_c - np.pi*0.5)) # mid-plane
	im2  = ax2.pcolormesh(xx[:,indx_mid,:]/au, yy[:,indx_mid,:]/au, retemp[:,indx_mid,:],
	 cmap=cmap, vmin = t_range[0], vmax=t_range[1], rasterized=True)

	cbar2 = fig.colorbar(im2, cax=cax2)
	ax2.set_xlabel(r'$x$ (au)')
	ax2.set_ylabel(r'$y$ (au)')
	cbar2.set_label(r'$T_\mathrm{dust}\ \mathrm{(K)}$')
	ax2.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)

	ax2.set_xlim(-r_range[1],r_range[1])
	ax2.set_ylim(-r_range[1],r_range[1])

	ax2.set_aspect(aspect)

	fig.subplots_adjust(wspace=wspace, hspace=hspace)
	fig.savefig('temperature.pdf', transparent=True)
	plt.close()




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
