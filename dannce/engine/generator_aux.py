"""Generator for 3d video images."""
import numpy as np
import keras
import os
from keras.applications.vgg19 import preprocess_input as pp_vgg19
import imageio
from dannce.engine import processing as processing
import scipy.io as sio
_DEFAULT_CAM_NAMES = [
	'CameraR', 'CameraL', 'CameraU', 'CameraU2', 'CameraS', 'CameraE']
_EXEP_MSG = "Desired Label channels and ground truth channels do not agree"


class DataGenerator_downsample(keras.utils.Sequence):
	"""Generate data for Keras."""

	def __init__(
		self, list_IDs, labels, vidreaders, batch_size=32,
		dim_in=(32, 32, 32), n_channels_in=1,
		dim_out=(32, 32, 32), n_channels_out=1,
		out_scale=5, shuffle=True,
		camnames=_DEFAULT_CAM_NAMES,
		crop_width=(0, 1024), crop_height=(20, 1300),
		tilefac=1, bbox_dim=(32, 32, 32), downsample=1, immode='video',
		labelmode='prob', preload=True, dsmode='dsm', chunks=3500):
		"""Initialize generator.

		TODO(params_definitions)
		"""
		self.dim_in = dim_in
		self.dim_out = dim_out
		self.batch_size = batch_size
		self.labels = labels
		self.vidreaders = vidreaders
		self.list_IDs = list_IDs
		self.n_channels_in = n_channels_in
		self.n_channels_out = n_channels_out
		self.shuffle = shuffle
		# sigma for the ground truth joint probability map Gaussians
		self.out_scale = out_scale
		self.camnames = camnames
		self.crop_width = crop_width
		self.crop_height = crop_height
		self.tilefac = tilefac
		self.bbox_dim = bbox_dim
		self.downsample = downsample
		self.preload = preload
		self.dsmode = dsmode
		self.on_epoch_end()

		if immode == 'video':
			self.extension = \
				'.' + list(vidreaders[camnames[0]].keys())[0].rsplit('.')[-1]

		self.immode = immode
		self.labelmode = labelmode
		self.chunks = chunks

	def __len__(self):
		"""Denote the number of batches per epoch."""
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		"""Generate one batch of data."""
		# Generate indexes of the batch
		indexes = \
			self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		"""Update indexes after each epoch."""
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def load_vid_frame(self, ind, camname, preload=True, extension='.mp4'):
		"""Load the video frame from a single camera."""
		fname = str(self.chunks * int(np.floor(ind / self.chunks))) + extension
		frame_num = ind % self.chunks

		keyname = os.path.join(camname, fname)

		if preload:
			return self.vidreaders[camname][keyname].get_data(
				frame_num).astype('float32')
		else:
			vid = imageio.get_reader(self.vidreaders[camname][keyname])
			im = vid.get_data(frame_num).astype('float32')
			vid.close()
			return im

	def load_tif_frame(self, ind, camname):
		"""Load frames in tif mode."""
		# In tif mode, vidreaders should just be paths to the tif directory
		return imageio.imread(
			os.path.join(self.vidreaders[camname], '{}.tif'.format(ind)))

	def __data_generation(self, list_IDs_temp):
		"""Generate data containing batch_size samples.

		# X : (n_samples, *dim, n_channels)
		"""
		# Initialization
		X = np.empty(
			(self.batch_size * len(self.camnames),
				*self.dim_in, self.n_channels_in),
			dtype='float32')

		# We'll need to transpose this later such that channels are last,
		# but initializaing the array this ways gives us
		# more flexibility in terms of user-defined array sizes
		y = np.empty(
			(self.batch_size * len(self.camnames),
				self.n_channels_out, *self.dim_out),
			dtype='float32')

		# Generate data
		cnt = 0
		for i, ID in enumerate(list_IDs_temp):
			for camname in self.camnames:
				# Store sample
				# TODO(Refactor): This section is tricky to read
				if self.immode == 'video':
					X[cnt] = self.load_vid_frame(
						self.labels[ID]['frames'][camname],
						camname, self.preload, self.extension
					)[self.crop_height[0]:self.crop_height[1],
					  self.crop_width[0]:self.crop_width[1]]
				elif self.immode == 'tif':
					X[cnt] = self.load_tif_frame(
						self.labels[ID]['frames'][camname], camname
					)[self.crop_height[0]:self.crop_height[1],
					  self.crop_width[0]:self.crop_width[1]]
				else:
					raise Exception('Not a valid image reading mode')

				# Labels will now be the pixel positions of each joint.
				# Here, we convert them to
				# probability maps with a numpy meshgrid operation
				this_y = np.round(self.labels[ID]['data'][camname])
				if self.immode == 'video':
					this_y[0, :] = this_y[0, :] - self.crop_width[0]
					this_y[1, :] = this_y[1, :] - self.crop_height[0]
				elif self.immode == 'tif':
					# DIRTY: HERE WE ASSUME TIF MODE IS ONLY
					# FOR PRE-DOWNSAMPLED SAVED TIFS
					this_y[0, :] = this_y[0, :] - self.crop_width[0] * 2
					this_y[1, :] = this_y[1, :] - self.crop_height[0] * 2

				(x_coord, y_coord) = np.meshgrid(
					np.arange(self.dim_out[1]), np.arange(self.dim_out[0]))

				# For 2D, this_y should be size (2, 20)
				if this_y.shape[1] != self.n_channels_out:
					# TODO(shape_exception):This should probably be its own
					# class that inherits from base exception
					raise Exception(_EXEP_MSG)

				if self.labelmode == 'prob':
					# Only do this if we actually need the labels --
					# this is too slow otherwise
					for j in range(self.n_channels_out):
						# I tested a version of this with numpy broadcasting,
						# and looping was ~100ms seconds faster for making
						# 20 maps
						# In the future, a shortcut might be to "stamp" a
						# truncated Gaussian pdf onto the images, centered
						# at the peak
						y[cnt, j] = np.exp(
							-((y_coord - this_y[1, j])**2 +
							  (x_coord - this_y[0, j])**2) /
							(2 * self.out_scale**2))

				cnt = cnt + 1

		# After we downsample to probabiltiy distributions,
		# we should rescale the maximum back to 1
		y = np.transpose(y, [0, 2, 3, 1])

		if self.downsample > 1:
			X = processing.downsample_batch(
				X, fac=self.downsample, method=self.dsmode)
			if self.labelmode == 'prob':
				y = processing.downsample_batch(
					y, fac=self.downsample, method=self.dsmode)
				y /= np.max(np.max(y, axis=1), axis=1)[
					:, np.newaxis, np.newaxis, :]
		# Again, we sloppily assume that tifs are already downsampled
		if self.downsample == 1:
			if self.labelmode == 'prob' and self.immode == 'tif':
				y = processing.downsample_batch(
					y, fac=self.downsample * 2, method=self.dsmode)
				y /= np.max(np.max(y, axis=1), axis=1)[
					:, np.newaxis, np.newaxis, :]

		if self.tilefac > 1:
			return (
				processing.return_tile(pp_vgg19(X), fac=self.tilefac),
				processing.return_tile(y, fac=self.tilefac))
		else:
			return pp_vgg19(X), y

	def save_for_dlc(self, imfolder, ext='.png', full_data=True):
		"""Generate data.

		# The full_data flag is used so that one can
		# write only the coordinates and not the images, if desired.
		"""
		cnt = 0
		list_IDs_temp = self.list_IDs
		dsize = self.labels[list_IDs_temp[0]]['data'][self.camnames[0]].shape
		allcoords = np.zeros(
			(len(list_IDs_temp) * len(self.camnames), dsize[1], 3), dtype='int')
		fnames = []

		# Load in a sample so that size can be found when full_data=False
		camname = self.camnames[0]
		# TODO(refactor): Hard to read
		X = self.load_vid_frame(
			self.labels[list_IDs_temp[0]]['frames'][camname],
			camname,
			self.preload,
			self.extension)[
			self.crop_height[0]:self.crop_height[1],
			self.crop_width[0]:self.crop_width[1]]

		for i, ID in enumerate(list_IDs_temp):
			for camname in self.camnames:
				if full_data:
					X = self.load_vid_frame(
						self.labels[ID]['frames'][camname],
						camname,
						self.preload,
						self.extension)[
						self.crop_height[0]:self.crop_height[1],
						self.crop_width[0]:self.crop_width[1]]

				# Labels will now be the pixel positions of each joint.
				# Here, we convert them to probability maps with a numpy
				# meshgrid operation
				this_y = self.labels[ID]['data'][camname].copy()
				this_y[0, :] = this_y[0, :] - self.crop_width[0]
				this_y[1, :] = this_y[1, :] - self.crop_height[0]

				if self.downsample > 1:
					X = processing.downsample_batch(
						X[np.newaxis, :, :, :],
						fac=self.downsample, method='dsm')
					this_y = np.round(this_y / 2).astype('int')
					if full_data:
						imageio.imwrite(
							imfolder + 'sample{}_'.format(ID) + camname + ext,
							X[0].astype('uint8'))
				else:
					if full_data:
						imageio.imwrite(
							imfolder + 'sample{}_'.format(ID) + camname + ext,
							X.astype('uint8'))

				allcoords[cnt, :, 0] = np.arange(dsize[1])
				allcoords[cnt, :, 1:] = this_y.T

				relpath = imfolder.split('/')[-2]
				relpath = \
					'../' + relpath + '/sample{}_'.format(ID) + camname + ext
				fnames.append(relpath)

				cnt = cnt + 1

		sio.savemat(
			imfolder + 'allcoords.mat',
			{'allcoords': allcoords,
			 'imsize': [X.shape[-1], X.shape[0], X.shape[1]],
			 'filenames': fnames})
