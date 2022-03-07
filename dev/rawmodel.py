import numpy as np
import pandas as pd
import matplotlib
import copy
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ptsmodel import PTSMODEL



class RawModel(PTSMODEL):

	def __init__(self, model):

		# check input
		if type(model) == PTSMODEL:
			pass
		else:
			print ("ERROR\tRawModel: input must be PTSMODEL object.")

		self = copy.copy(model)

		if 'xx' in self.__dict__:
			rr, tt, phph = self.rr, self.tt, self.phph
			self.xx = rr*np.sin(tt)*np.cos(phph)
			self.yy = rr*np.sin(tt)*np.sin(phph)
			self.zz = rr*np.cos(tt)


	def rotate_coordinates(self, angle, axis='x', degree=True):
		'''
		Rotate coordinates around an axis.

		Parameters
		----------
		 - angle: Rotation angle.
		 - axis: Axis around which coordinates are rotated.
		'''

		if degree:
			angle *= np.pi/180.

		if axis == 'x':
			self.xx_rot = self.xx
			self.yy_rot = self.yy*np.cos(self.tt) + self.zz*np.sin(self.tt)
			self.zz_rot = -self.zz*np.sin(self.tt) + self.zz*np.cos(self.tt)

	def project_to(self, plane='xy'):

		if 'xx_rot' in self.__dict__:
			self.project_x = self.xx_rot
			self.project_y = self.zz_rot
			pass
		else:
			print ('No coordinate rotation is found.')
			print ('No rotation is adopted.')

			self.project_x = self.xx
			self.project_y = self.zz
