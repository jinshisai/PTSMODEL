Protostellar System Model
---------------------------
Build a kinematic model of a protostellar system, which consists of a protostar(s), a disk(s) and an envelope, and calculate the model images. Use the open code [RADMC-3D](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/) to calculate a tempearture distribution and the solve radiative transfer.

Currently, coordinates are limitied only to spherical coordinates. A density profile for a disk is cut-off disk. The density and velocity distributions for an envelope model is calculated based on Ulrich 1976.


**Containts**
- ptsmodel: Module of the protostellar system model.
- infiles: Input files necessary for calculations with RADMC-3D.
- run_models.py: A script to build models and calculate model images.


**References**
- Ulrich (1976)
- Mendoza et al. (2004)


**Contact**  
E-mail: jn.insa.sai@gmail.com  
Jinshi Sai (Insa Choi)  
Depertment of Astronomy, the University of Tokyo



How to use
------------------
The script run_models.py contains all needed processes. Run run_models.py in a directory containing ptsmodel and infiles. I recommend you to generate one directory for one model, and add a line to change directory inside a script to start from the directory where ptsmodel located but put all calculation results in another directory. Or you can use ptsmodel in a more interactive way.

The model requires many input parameters. Look into run_models.py to check what parameters are needed. The modeling is done with the following steps:

1. Give a grid
1. Put a protostar
1. Calculate density distributions of a disk and an envelope
1. Calculate velocity field
1. (Visualize density and velocity distributions)
1. Export the model into RADMC-3D input files
1. Calculate the temperature distribution with RADMC-3D
1. Solve the radiative transfer with RADMC-3D
1. Export into a fits file

See below for the detail of each step. Constants like ```au``` must be given in a script. Here is an example of constants:

```python
# -------------- constants -----------------
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
clight = 2.99792458e10  # light speed [cm s^-1]
# ------------------------------------------
```

**Steps**
1. Build an empty model and grid for a model.

```python
# Import
import ptsmodel # in a directory where you put ptsmodel

# Model name
modelname = 'test_model'

# Grid parameters (spherical coordiante)
nr       = 32               # radius from center
ntheta   = 32               # angle from z axis, 90 deg is disk plane
nphi     = 32               # azimuthal
rmin     = 1.*au            # minimum radius of the domein
rmax     = 4000.*au         # maximum radius of the domein
thetamin = 0.               # from z axis

# Make a model
model_i = ptsmodel.PTSMODEL(modelname, nr, ntheta, nphi ,rmin, rmax, thetamin=0., thetamax=np.pi*0.5, phimin=0., phimax=2.*np.pi)
```

2. Put a protostar

```python
# Stellar parameters
mstar = 1.6                                       # stellar mass [Msun]
mstar = mstar*ms                                  # Msun --> g
lstar = 3.5*ls                                    # Steller Luminosity
rstar = 4.*rs                                     # Stahler, Shu and Taam (1980), for protostars
pstar = np.array([0.,0.,0.])                      # position

# Put a protostar
model_i.prop_star(mstar, lstar, rstar, pstar, tstar=None)
```

3. Make density distributions.

Calculate density distributions for a disk and an envelope, and run ```.rho_model```. Then, if you input both, envelope density within the disk radius and the scale height is replaced with disk density. You can put either only if you don't need to put both.

```python
#  ---------------- Input -------------------
# Disk
mdisk     = 1.8e-2*ms                     # Gas mass of a disk
plsig     = -0.5e0                        # Power-law index of the surface density
rd_in     = 1.*au                         # Dust & gas disk inner radius [cm]
rd_out    = 600.*au                       # Dust & gas disk outer radius [cm]
Tdisk_1au = 400.                          # Disk temperature at 1 au [K]
cs_1au    = np.sqrt(kb*Tdisk_1au/(mu*mp)) # Isothermal sound speed at 1 au [cm/s]
Omg_1au   = np.sqrt(Ggrav*mstar/au**3)    # Keplerian angular velocity at 1 au[/s]
hr0       = cs_1au/Omg_1au/au             # H/r at 1au, assuming hydrostatic equillibrium
plh       = 0.25                          # Powerlaw of flaring h/r, assuming T prop r^-0.5 & Keplerian rotation

# Envelope
rcent  = rd_out   # Centrifugal radius [cm]
re_in  = rd_in    # Envelope inner radius [cm]
re_out = 5000.*au # Envelope outer radius [cm]
rho0_e = 1.4e-18  # rho at rcent [g]

# Global
mu         = 2.8    # mean molecular weight. Kauffman (2008)
abunc18o   = 1.7e-7 # C18O abundance. Frerking, Langer and Wilson (1982), for rho-Oph and Taurus
gtod_ratio = 100.   # gas to dust mass ratio
# --------------------------------------------

# Density distributions
# For an envelope
model_i.rho_envUl76(rho0_e, rcent, re_in=re_in, re_out=re_out)

# For a disk
model_i.rho_cutoffdisk(mdisk, rd_in, rd_out, plsig, hr0, plh, sig0=False)

# Density distribution
model_i.rho_model(abunc18o, mu, gtod_ratio)
```

4. Calculate velocity distribution
After put a protostar and made a density distribution, you can calculate the velocity distributions from given information; the protostellar mass and the centrifugal radius.

```python
# velocity field
model_i.vfield_model()
```

If you want to add more options, you can modify the velocity field as follows, for example.

```python
# Modify velocity field
# Suppress the radial velocity
vr = model_i.vr # v_r
vr = vr*0.5     # Reduce v_r by a factor of 2
model_i.vr = vr # Update
```


5. (Visualize the density and velocity distributions)

Density and velocity field is easily visualized if you want.

```python
model_i.show_density() # Visualize the density distribution
model_i.show_vfield(r_range=[100*au, 4000*au]) # Visualize v-field

# Options:
#  - nrho_g_range: nrho range for the coloring. Must be given [min, max].
#  - r_range: Radial range to be shown. Must be given [min, max].
#  - binstep: Binning step to make the velocity field sparse. Default = 1.
#  - figsize: Figure size. Default (11.69,8.27), which is A4 size.
#  - vscale: Parameter to determine lengths of arrows to show v-field. Default 3e2. Large number makes arrows shorter.
#  - width: Widths of arrows for v-field. Default 10.
#  - cmap: Color map. Default 'coolwarm'.
#  - fontsize: Fontsize in pt. Default 14.
#  - wspace: Horizontal space between subplots. Default 0.4.
#  - hspace: Vertical space between subplots. Default 0.2
```


6. Export the model into RADMC-3D input files

```python
# Input
nphot    = 10000000 # Number of photon for Monte Carlo calculations.
dustopac = 'nrmS03' # Dust opacity. Here adopt opacity table calculated in Semenov et al. (2003).
line     = 'c18o'   # Line. Here, C18O.

# Export
model_i.export_to_radmc3d(nphot, dustopac, line) # All information including info. of a protostar, density and v-field is exported.
```


7. Calculate the temperature distribution

RADMC-3D has to be called out of Python but you may want to put all procedures in one script. Here, system module is used to run RADMC-3D inside a python script.

```python
# Calculate temperature profile
print ('Calculate temperature profile')
print ('radmc3d mctherm')
os.system('radmc3d mctherm') # Run RADMC-3D

# Visualize. t_range is an option to determine the coloring range.
model_i.plot_temperature(t_range=[0.,120.])
```


8. Solve the radiative transfer

Produce images of the model by solving the radiative transfer. system module is used again to run RADMC-3D inside a python script.

```python
# Input
# Example for C18O 2--1
inc      = 73.                     # [deg]
inc      = 180.-inc                # Make lower side face us
restfreq = 219.56035e9             # Rest frequency of C18O 2--1 [Hz]
sizeau   = 8000.                   # Image size [au]
pa       = 54.-90.                 # Position angle [deg]
vmin     = -5                      # Minimum velocity to be imaged
vmax     = 5                       # Maximum velocity to be imaged
vrange   = np.array([vmin,vmax])   # Velocity range
nchan    = 70                      # Channel number
iline    = 2                       # The upper exciation level of the radiation. 2 means J=2--1.
npix     = 1000                    # Pixel size of image

# Frequency --> Wavelength
freqrange      = restfreq*(1. - vrange*1.e5/clight)
lambdarange    = clight/freqrange       # in cm
lambdarange    = lambdarange*1.e-2*1.e6 # cm --> micron
lammin, lammax = lambdarange

# Command
run_radmc = 'radmc3d image npix %i phi 0 iline %i\
sizeau %.f lambdarange %.15e %.15e posang %.2f\
incl %.2f nlam %.0f'%(npix, iline, sizeau, lammin,lammax,pa,inc,nchan)

# Check
print ('Solve radiative transfer')
print (run_radmc)

# Run RADMC-3D
os.system(run_radmc)
```


9. Export into a fits file

Finally, obtained image is exported into a fits file. Then, you can handle the model image like a real observed map. You can perform beam convolution and add noise if you necessary.


```python
# Input for export
obname            = 'L1489 IRS'                   # Object name
outname           = model_i.modelname + '_c18o21' # Output fits name
dist              = 140.                          # Distance to object [pc]
coordinate_center = '4h4m43.07s +26d18m56.20s'    # RA Dec
vsys              = 7.22                          # km/s
beam_convolution  = True                          # Convolve a beam?
beam              = [7.7, 6.4, -85.]              # bmaj, bmin, bpa
Tb                = False                         # In a unit of Tb?
add_noise         = True                          # Add noise
rms               = 0.22                          # Jy/beam
frame             = 'fk5'                         # Frame
projection        = 'SFL'                         # Projection

ptsmodel.export_radmc_tofits.export_radmc_tofits(outname,
 dist=dist, obname=obname, vsys=vsys,
 coordinate_center=coordinate_center, beam_convolution=beam_convolution,
 beam=beam, Tb=Tb, add_noise=add_noise, rms=rms,
 frame=frame, projection=projection)
```