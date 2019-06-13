
import numpy as np
import keras

import os
from keras.applications.vgg19 import preprocess_input as pp_vgg19
from keras.applications.inception_v3 import preprocess_input as pp_inception

import processing
import ops

import imageio

import time

import warnings

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, clusterIDs, batch_size=32, dim_in=(32,32,32), n_channels_in=1, 
				dim_out = (32,32,32), n_channels_out=1, out_scale=5,shuffle=True, camnames = [],
				crop_width = (0, 1024), crop_height = (20, 1300), tilefac=1, bbox_dim=(32,32,32), samples_per_cluster=0, training=True, vidreaders=None):
		'Initialization'
		self.dim_in = dim_in
		self.dim_out = dim_out
		self.batch_size = batch_size
		self.labels = labels
		self.vidreaders = vidreaders
		self.list_IDs = list_IDs
		self.n_channels_in = n_channels_in
		self.n_channels_out = n_channels_out
		self.shuffle = shuffle
		self.out_scale = out_scale #sigma for the ground truth joint probability map Gaussians
		self.camnames = camnames
		self.crop_width = crop_width
		self.crop_height = crop_height
		self.tilefac = tilefac
		self.bbox_dim = bbox_dim #This is only used when serving/targeting 2D, and even so is deprecated
		self.clusterIDs = clusterIDs
		self.samples_per_cluster = samples_per_cluster
		self.training = training
		self.on_epoch_end()

		if self.vidreaders is not None:
			self.extension = '.' + list(vidreaders[camnames[0][0]].keys())[0].rsplit('.')[-1]

		assert len(self.list_IDs) == len(self.clusterIDs)

	def __len__(self):
		'Denotes the number of batches per epoch'
		if self.training:
			return self.samples_per_cluster*len(np.unique(self.clusterIDs))//self.batch_size
		else:
			return int(np.floor(len(self.list_IDs) / self.batch_size))

	def on_epoch_end(self):
		'Updates indexes after each epoch'

		if self.training:
			print('Regnerating samples from clusters...')
			self.indexes = []

			uClusterID = np.unique(self.clusterIDs)
			for i in range(len(uClusterID)):
				clusterID = uClusterID[i]
				inds = np.where(self.clusterIDs==clusterID)[0]

				# draw randomly from the cluster
				inds_ = np.random.choice(inds,size=(self.samples_per_cluster,))

				self.indexes = self.indexes + list(inds_)
			self.indexes = np.array(self.indexes)
		else:
			self.indexes = np.arange(len(self.list_IDs))

		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def load_vid_frame(self, ind, camname, preload=True, extension='.mp4'):
		# This is currently implemented for handling only one camera as input
		fname = str(3500*int(np.floor(ind/3500))) + extension
		frame_num = ind % 3500

		#camname = self.camnames[0]

		#print(fname)
		keyname = os.path.join(camname,fname)
		#print(keyname)
		if preload:
			return self.vidreaders[camname][keyname].get_data(frame_num).astype('float32')
		else:
			vid = imageio.get_reader(self.vidreaders[camname][keyname])
			im = vid.get_data(frame_num).astype('float32')
			vid.close()
			return im


class DataGenerator_3Dconv_kmeans(DataGenerator):
	"""
	Updated generator class that with resample from kmeans clusters at the end of each epoch. Also handles data across multiple experiments
	"""
	def __init__(self, list_IDs, labels, labels_3d, camera_params, clusterIDs, com3d, tifdirs, batch_size=32, dim_in=(32,32,32), n_channels_in=1, 
				dim_out = (32,32,32), n_channels_out=1, out_scale=5,shuffle=True, camnames = [],
				crop_width = (0, 1024), crop_height = (20, 1300), tilefac=1, bbox_dim=(32,32,32), 
				vmin=-100,vmax=100,nvox=32,interp='linear',depth=False,channel_combo=None,mode='3dprob',preload=True, 
				samples_per_cluster=0, immode='tif', training=True,rotation=False,pregrid = None,pre_projgrid = None, stamp=False, 
				vidreaders = None, distort=False, expval = False, multicam = True, var_reg=False, COM_aug = None, crop_im = True,
				norm_im = True):
		DataGenerator.__init__(self, list_IDs,labels, clusterIDs, batch_size,dim_in,n_channels_in,
			dim_out, n_channels_out, out_scale, shuffle, camnames, crop_width, crop_height, tilefac, bbox_dim, samples_per_cluster, training, vidreaders)
		#self.projected_grid = projected_grid
		self.vmin= vmin
		self.vmax = vmax
		self.nvox = nvox
		self.vsize = (vmax-vmin)/nvox
		self.dim_out_3d = (nvox, nvox, nvox)
		self.labels_3d = labels_3d
		self.camera_params = camera_params
		self.interp = interp
		self.depth = depth
		self.channel_combo = channel_combo
		print(self.channel_combo)
		self.mode = mode
		self.preload = preload
		self.immode = immode		
		self.tifdirs = tifdirs
		self.com3d = com3d
		self.rotation = rotation
		self.pregrid = pregrid
		self.pre_projgrid = pre_projgrid
		self.stamp=stamp
		self.distort = distort
		self.expval = expval
		self.multicam = multicam
		self.var_reg = var_reg
		self.COM_aug = COM_aug 
		self.crop_im = crop_im
		self.norm_im = norm_im #If saving npy as uint8 rather than training directly, dont normalize

		if self.pregrid is not None:
			#Then we need to save our world size for later use
			# we expect pregride to be (coord_x,coord_y,coord_z)
			self.worldsize = np.min(pregrid[0])

		if self.stamp:
			# To save time, "stamp" a 3d gaussian at each marker position
			(x_coord_3d, y_coord_3d, z_coord_3d) = np.meshgrid(np.arange(self.worldsize,-self.worldsize,self.vsize), 
												   np.arange(self.worldsize,-self.worldsize,self.vsize),
												   np.arange(self.worldsize,-self.worldsize,self.vsize))

			self.stamp_ = np.exp(-((y_coord_3d-0)**2 + (x_coord_3d-0)**2 + (z_coord_3d-0)**2)/(2*self.out_scale**2))


	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def rot90(self,X):
		#Rotate 90
		X = np.transpose(X,[1,0,2,3])
		X = X[:,::-1,:,:]

		return X

	def rot180(self,X):
		#Rotate 180
		X = X[::-1,::-1,:,:]

		return X

	def random_rotate(self,X, y_3d, log = False):
		"""
		Rotate each sample by 0, 90, 180, or 270 degrees

		log indicates whether to return the rotation pattern (for saving) as well
		"""

		rots = np.random.choice(np.arange(4),X.shape[0])
		for i in range(X.shape[0]):
			if rots[i]==0:
				pass
			elif rots[i]==1:
				#Rotate180
				X[i] = self.rot180(X[i])
				y_3d[i] = self.rot180(y_3d[i])
			elif rots[i]==2:
				#Rotate90
				X[i] = self.rot90(X[i])
				y_3d[i] = self.rot90(y_3d[i])
			elif rots[i]==3:
				#Rotate -90/270
				X[i] = self.rot90(X[i])
				X[i] = self.rot180(X[i])
				y_3d[i] = self.rot90(y_3d[i])
				y_3d[i] = self.rot180(y_3d[i])

		if log:
			return X, y_3d, rots
		else:
			return X, y_3d

	def fetch_grid(self,c):
		"""
		return ROI from pregrid
		"""
		c0 = int((c[0]-(self.worldsize))/self.vsize)
		c1 = int((c[1]-(self.worldsize))/self.vsize)
		c2 = int((c[2]-(self.worldsize))/self.vsize)

		x_coord_3d = self.pregrid[0][c0-self.nvox//2:c0+self.nvox//2,
				   c0-self.nvox//2:c0+self.nvox//2,
				   c0-self.nvox//2:c0+self.nvox//2]
		y_coord_3d = self.pregrid[1][c1-self.nvox//2:c1+self.nvox//2,
				   c1-self.nvox//2:c1+self.nvox//2,
				   c1-self.nvox//2:c1+self.nvox//2]
		z_coord_3d = self.pregrid[2][c2-self.nvox//2:c2+self.nvox//2,
				   c2-self.nvox//2:c2+self.nvox//2,
				   c2-self.nvox//2:c2+self.nvox//2]

		return x_coord_3d, y_coord_3d, z_coord_3d

	def fetch_projgrid(self,c,e,cam):
		"""
		return ROI from pre_projgrid
		"""
		c0 = int((c[0]-(self.worldsize))/self.vsize)
		c1 = int((c[1]-(self.worldsize))/self.vsize)
		c2 = int((c[2]-(self.worldsize))/self.vsize)

		proj_grid = self.pre_projgrid[e][cam][c1-self.nvox//2:c1+self.nvox//2,
										  c0-self.nvox//2:c0+self.nvox//2,
										  c2-self.nvox//2:c2+self.nvox//2].copy()
		proj_grid = np.reshape(proj_grid,[-1,3])

		return proj_grid

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization

		first_exp = int(self.list_IDs[0].split('_')[0])

		X = np.zeros((self.batch_size*len(self.camnames[first_exp]), *self.dim_out_3d, self.n_channels_in + self.depth),dtype='float32')
		if self.mode == '3dprob':
			y_3d = np.zeros((self.batch_size, self.n_channels_out, *self.dim_out_3d), dtype='float32')
		elif self.mode == 'coordinates':
			y_3d = np.zeros((self.batch_size, 3, self.n_channels_out), dtype='float32')
		else:
			raise Exception("not a valid generator mode")

		if self.expval:
			X_grid = np.zeros((self.batch_size,self.dim_out_3d[0]*self.dim_out_3d[1]*self.dim_out_3d[2], 3),dtype='float32')   

		# Generate data
		cnt = 0
		for i, ID in enumerate(list_IDs_temp):

			sampleID = int(ID.split('_')[1])
			experimentID = int(ID.split('_')[0])

			# For 3D ground truth
			this_y_3d = self.labels_3d[ID]

			this_COM_3d = self.com3d[ID]

			if self.COM_aug is not None:
				this_COM_3d = this_COM_3d.copy().ravel()
				this_COM_3d = this_COM_3d + self.COM_aug*2*np.random.rand(len(this_COM_3d)) - self.COM_aug


			#Actually we need to create and project the grid here, relative to the reference point (SpineM). 

			if self.pregrid is None:
				(x_coord_3d, y_coord_3d, z_coord_3d) = np.meshgrid(np.arange(self.vmin + this_COM_3d[0] + self.vsize/2,this_COM_3d[0]+self.vmax,self.vsize), 
																   np.arange(self.vmin + this_COM_3d[1] + self.vsize/2,this_COM_3d[1]+self.vmax,self.vsize), 
																   np.arange(self.vmin + this_COM_3d[2] + self.vsize/2,this_COM_3d[2]+self.vmax,self.vsize))
			else:
				
				(x_coord_3d, y_coord_3d, z_coord_3d) = self.fetch_grid(this_COM_3d)
				

			if self.mode == '3dprob':
				
				for j in range(self.n_channels_out):
					if self.stamp:
						# these are coordinates of each marker relative to COM
						c = this_y_3d[:,j] - this_COM_3d
						c0 = int((c[0]-(self.worldsize))/self.vsize)
						c1 = int((c[1]-(self.worldsize))/self.vsize)
						c2 = int((c[2]-(self.worldsize))/self.vsize)
						y_3d[i,j] = self.stamp_[c1-self.nvox//2:c1+self.nvox//2,c0+self.nvox//2:c0-self.nvox//2:-1,c2+self.nvox//2:c2-self.nvox//2:-1]
					else:
						y_3d[i,j] = np.exp(-((y_coord_3d-this_y_3d[1,j])**2 + (x_coord_3d-this_y_3d[0,j])**2 + (z_coord_3d-this_y_3d[2,j])**2)/(2*self.out_scale**2))
						# When the voxel grid is coarse, we will likely miss the peak of the probability distribution, as it will lie somewhere in the middle of a 
						# large voxel. So here we renormalize to [~, 1]
					#y_3d[i,j] = y_3d[i,j]/np.max(y_3d[i,j])
				
			if self.mode == 'coordinates':
				#this_y_3d = this_y_3d - this_y_3d[:,4:5]
				if this_y_3d.shape == y_3d[i].shape:
					y_3d[i] = this_y_3d#.ravel()
				else:
					warnings.warn("Note: ignoring dimension mismatch in 3D labels")

			if self.expval:
				X_grid[i] = np.stack((x_coord_3d.ravel(),y_coord_3d.ravel(),z_coord_3d.ravel()),axis=1)

			for camname in self.camnames[experimentID]:


				# !!!! Note that this unnecessary round actually saved me - otherwise this_y would be decreasing every iteration. Should probably replace by a copy here
				# Added .copy() 11/15/18
				this_y = np.round(self.labels[ID]['data'][camname]).copy()

				#print(this_y)

				if np.all(np.isnan(this_y)):
					com_precrop = np.zeros_like(this_y[:,0])*np.nan
				else:
					com_precrop = np.nanmean(this_y,axis=1) # For projecting points, we should not use this offset

				# Store sample
				if self.immode == 'tif': # for pre-cropped tifs
					
					thisim = imageio.imread(os.path.join(self.tifdirs[experimentID],camname,'{}.tif'.format(sampleID)))
					
				elif self.immode == 'vid': # from raw video, need to crop
					thisim = self.load_vid_frame(self.labels[ID]['frames'][camname], camname, self.preload, extension=self.extension)[self.crop_height[0]:self.crop_height[1],self.crop_width[0]:self.crop_width[1]]
					

				elif self.immode == 'arb_ims':
					# load in the image file at the specified path
					thisim = imageio.imread(self.tifdirs[experimentID] + self.labels[ID]['frames'][camname][0] +'.jpg')

				if self.immode == 'vid' or self.immode == 'arb_ims':
					this_y[0,:] = this_y[0,:] - self.crop_width[0]
					this_y[1,:] = this_y[1,:] - self.crop_height[0]

					com = np.nanmean(this_y,axis=1)

					if self.crop_im:
						
						if np.all(np.isnan(com)):
							thisim = np.zeros((self.dim_in[1],self.dim_in[0],self.n_channels_in))
						else:
							thisim = processing.cropcom(thisim,com, size=self.dim_in[0])



				# Project de novo or load in approximate (faster)
				if self.pre_projgrid is None:
					proj_grid = ops.project_to2d(np.stack((x_coord_3d.ravel(),y_coord_3d.ravel(),z_coord_3d.ravel()),axis=1),
						self.camera_params[experimentID][camname]['K'], self.camera_params[experimentID][camname]['R'], self.camera_params[experimentID][camname]['t'])
				else:
					proj_grid = self.fetch_projgrid(this_COM_3d,experimentID,camname)


				if self.depth:
					d = proj_grid[:,2]


				if self.distort:
					"""
					Distort points using lens distortion parameters
					"""
					proj_grid = ops.distortPoints(proj_grid[:,:2],
						self.camera_params[experimentID][camname]['K'],
						np.squeeze(self.camera_params[experimentID][camname]['RDistort']), 
						np.squeeze(self.camera_params[experimentID][camname]['TDistort'])).T

				if self.crop_im:			
					proj_grid = proj_grid[:,:2] - com_precrop + self.dim_in[0]//2 
					# Now all coordinates should map properly to the image cropped around the COM
				else:
					#Then the only thing we need to correct for is crops at the borders
					proj_grid = proj_grid[:,:2]
					proj_grid[:,0] = proj_grid[:,0] - self.crop_width[0]
					proj_grid[:,1] = proj_grid[:,1] - self.crop_height[0]
				

				(r, g, b) = ops.sample_grid(thisim,proj_grid,method=self.interp)

				if ~np.any(np.isnan(com_precrop)) or self.channel_combo == 'avg' or not self.crop_im:
				
					X[cnt,:,:,:,0] = np.reshape(r,(self.nvox,self.nvox,self.nvox))
					X[cnt,:,:,:,1] = np.reshape(g,(self.nvox,self.nvox,self.nvox))
					X[cnt,:,:,:,2] = np.reshape(b,(self.nvox,self.nvox,self.nvox))
					if self.depth:
						X[cnt,:,:,:,3] = np.reshape(d,(self.nvox,self.nvox,self.nvox))
				

				# if this_y.shape[1] != self.n_channels_out and self.mode != 'coordinates':
				# 	raise Exception("Desired Label channels and ground truth channels do not agree")


				cnt = cnt + 1

		if self.multicam:
			X = np.reshape(X,(self.batch_size,len(self.camnames[first_exp]),X.shape[1],X.shape[2],X.shape[3],X.shape[4]))
			X = np.transpose(X,[0,2,3,4,5,1])
			if self.channel_combo == 'avg':
				X = np.nanmean(X,axis=-1)
			elif self.channel_combo == 'random': #randomly reorder the cameras fed into the first layer
				X = X[:,:,:,:,:,np.random.permutation(X.shape[-1])]
				X = np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],X.shape[3], X.shape[4]*X.shape[5]),order='F')
			else:
				X = np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],X.shape[3], X.shape[4]*X.shape[5]),order='F')
		else:
			# Then leave the batch_size and num_cams combined
			y_3d = np.tile(y_3d,[len(self.camnames[experimentID]),1,1,1,1])

		if self.mode =='3dprob':
			y_3d = np.transpose(y_3d,[0,2,3,4,1])

		if self.rotation:
			if self.expval:
				#First make X_grid 3d
				X_grid = np.reshape(X_grid,(self.batch_size,self.nvox,self.nvox,self.nvox,3))

				if self.norm_im:
					X, X_grid = self.random_rotate(X, X_grid)
				else:
					X, X_grid, rotate_log = self.random_rotate(X, X_grid,log=True)
				# Need to reshape back to raveled version
				X_grid = np.reshape(X_grid,(self.batch_size,-1,3))
			else:
				if self.norm_im:
					X, y_3d = self.random_rotate(X, y_3d)
				else:
					X, y_3d, rotate_log = self.random_rotate(X, y_3d,log=True)
		
		if self.expval: #Then we also need to return the 3d grid center coordinates, for calculating a spatial expected value
		# Xgrid is typically symmetric for 90 and 180 degree rotations (when vmax and vmin are symmetric) 
		# around the z-axis, so no need to rotate X_grid.
			if self.var_reg:
				return [processing.preprocess_3d(X), X_grid], [y_3d,np.zeros((self.batch_size,1))]

			if self.norm_im: 
				return [processing.preprocess_3d(X), X_grid], y_3d #y_3d is in coordinates here.
			else:
				return [X, X_grid], [y_3d, rotate_log]
		else:
			if self.norm_im:
				return processing.preprocess_3d(X), y_3d
			else:
				return X, [y_3d, rotate_log]

class DataGenerator_3Dconv_frommem(keras.utils.Sequence):

	def __init__(self, list_IDs, data, labels, batch_size, rotation=True, random=True, chan_num=3, shuffle=True, 
			expval=False, xgrid = None, var_reg = False):
		self.list_IDs = list_IDs
		self.data = data
		self.labels = labels
		self.rotation = rotation
		self.batch_size = batch_size
		self.random = random
		self.chan_num = 3
		self.shuffle = shuffle
		self.expval = expval
		self.var_reg = var_reg

		if self.expval:
			self.xgrid = xgrid


		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def rot90(self,X):
		#Rotate 90
		X = np.transpose(X,[1,0,2,3])
		X = X[:,::-1,:,:]

		return X

	def rot180(self,X):
		#Rotate 180
		X = X[::-1,::-1,:,:]

		return X

	def random_rotate(self,X, y_3d):
		"""
		Rotate each sample by 0, 90, 180, or 270 degrees
		"""

		rots = np.random.choice(np.arange(4),X.shape[0])
		for i in range(X.shape[0]):
			if rots[i]==0:
				pass
			elif rots[i]==1:
				#Rotate180
				X[i] = self.rot180(X[i])
				y_3d[i] = self.rot180(y_3d[i])
			elif rots[i]==2:
				#Rotate90
				X[i] = self.rot90(X[i])
				y_3d[i] = self.rot90(y_3d[i])
			elif rots[i]==3:
				#Rotate -90/270
				X[i] = self.rot90(X[i])
				X[i] = self.rot180(X[i])
				y_3d[i] = self.rot90(y_3d[i])
				y_3d[i] = self.rot180(y_3d[i])

		return X, y_3d

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples'
		# Initialization

		X = np.zeros((self.batch_size,*self.data.shape[1:]))
		y_3d = np.zeros((self.batch_size,*self.labels.shape[1:]))

		if self.expval:
			X_grid = np.zeros((self.batch_size,*self.xgrid.shape[1:]))

		for i, ID in enumerate(list_IDs_temp):
			X[i] = self.data[ID].copy()
			y_3d[i] = self.labels[ID]
			if self.expval:
				X_grid[i] = self.xgrid[ID]

		if self.rotation:
			X, y_3d = self.random_rotate(X.copy(), y_3d.copy())

		# Randomly re-order, if desired
		if self.random:
			#pass
			X = np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],X.shape[3], self.chan_num,-1),order='F')
			X = X[:,:,:,:,:,np.random.permutation(X.shape[-1])]
			X = np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],X.shape[3], X.shape[4]*X.shape[5]),order='F')

		if self.expval:
			if self.var_reg:
				return [X, X_grid], [y_3d, np.zeros(self.batch_size)]
			return [X, X_grid], y_3d
		else:
			return X, y_3d
