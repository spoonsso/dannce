"""Define routines for reading/structuring input data for DANNCE."""
import numpy as np
import scipy.io as sio
from dannce.engine import ops as ops
import os
from six.moves import cPickle
from scipy.special import comb
import warnings


def prepare_data(CONFIG_PARAMS, com_flag=True, nanflag=True):
	"""Assemble necessary data structures given a set of config params.

	Given a set of config params, assemble necessary data structures and
	return them -- tailored to center of mass finding
	That is, we are refactoring to get rid of unneeded data structures
	(i.e. data 3d)
	"""
	data = sio.loadmat(
		os.path.join(CONFIG_PARAMS['datadir'], CONFIG_PARAMS['datafile'][0]))
	samples = np.squeeze(data['data_sampleID'])

	# Collect data labels and matched frames info. We will keep the 2d labels
	# here just because we could in theory use this for training later.
	# No need to collect 3d data but it sueful for checking predictions
	if len(CONFIG_PARAMS['CAMNAMES']) != len(CONFIG_PARAMS['datafile']):
		raise Exception("need a datafile for every cameras")

	framedict = {}
	ddict = {}
	for i in range(len(CONFIG_PARAMS['datafile'])):
		fr = sio.loadmat(
			os.path.join(CONFIG_PARAMS['datadir'], CONFIG_PARAMS['datafile'][i]))
		framedict[CONFIG_PARAMS['CAMNAMES'][i]] = np.squeeze(fr['data_frame'])
		data = fr['data_2d']

		# reshape data_2d so that it is shape (time points, 2, 20)
		data = np.transpose(np.reshape(data, [data.shape[0], -1, 2]), [0, 2, 1])

		# Correct for Matlab "1" indexing
		data = data - 1

		if com_flag:
			# Convert to COM only
			if nanflag:
				data = np.mean(data, axis=2)
			else:
				data = np.nanmean(data, axis=2)
			data = data[:, :, np.newaxis]
		ddict[CONFIG_PARAMS['CAMNAMES'][i]] = data

	data_3d = fr['data_3d']
	data_3d = np.transpose(
		np.reshape(data_3d, [data_3d.shape[0], -1, 3]), [0, 2, 1])

	datadict = {}
	datadict_3d = {}
	for i in range(len(samples)):
		frames = {}
		data = {}
		for j in range(len(CONFIG_PARAMS['CAMNAMES'])):
			frames[CONFIG_PARAMS['CAMNAMES'][j]] = \
				framedict[CONFIG_PARAMS['CAMNAMES'][j]][i]
			data[CONFIG_PARAMS['CAMNAMES'][j]] = \
				ddict[CONFIG_PARAMS['CAMNAMES'][j]][i]
		datadict[samples[i]] = {'data': data, 'frames': frames}
		datadict_3d[samples[i]] = data_3d[i]

	if 'calib_file' in list(CONFIG_PARAMS.keys()):
		cameras = {}
		for i in range(len(CONFIG_PARAMS['CAMNAMES'])):
			test = sio.loadmat(
				os.path.join(CONFIG_PARAMS['CALIB_DIR'], CONFIG_PARAMS['calib_file'][i]))
			cameras[CONFIG_PARAMS['CAMNAMES'][i]] = {
				'K': test['K'], 'R': test['r'], 't': test['t']}
			if 'RDistort' in list(test.keys()):
				# Added Distortion params on Dec. 19 2018
				cameras[CONFIG_PARAMS['CAMNAMES'][i]]['RDistort'] = test['RDistort']
				cameras[CONFIG_PARAMS['CAMNAMES'][i]]['TDistort'] = test['TDistort']

		return samples, datadict, datadict_3d, fr['data_3d'], cameras
	else:
		return samples, datadict, datadict_3d, fr['data_3d']


def do_retriangulate(this_com, j, k, uCamnames, camera_mats):
	"""Retriangulate COM.

	If cameras parameters have been updated since finding COM,
	COM should be re-trianngulated
	"""
	pts1 = this_com[uCamnames[j]]['COM']
	pts2 = this_com[uCamnames[k]]['COM']
	pts1 = pts1[np.newaxis, :]
	pts2 = pts2[np.newaxis, :]

	test = camera_mats[uCamnames[j]]
	cam1 = ops.camera_matrix(test['K'], test['R'], test['t'])

	test = camera_mats[uCamnames[k]]
	cam2 = ops.camera_matrix(test['K'], test['R'], test['t'])

	test3d = np.squeeze(ops.triangulate(pts1, pts2, cam1, cam2))
	this_com['triangulation']['{}_{}'.format(uCamnames[j], uCamnames[k])] = test3d


def prepare_COM(
	comfile, datadict, comthresh=0.01, weighted=True, retriangulate=False,
	camera_mats=None, conf_rescale=None, method='mean'):
	"""Replace 2d coords with preprocessed COM coords, return 3d COM coords.

	Loads COM file, replaces 2D coordinates in datadict with the preprocessed
	COM coordinates, returns dict of 3d COM coordinates

	Thresholds COM predictions at comthresh w.r.t. saved pred_max values.
	Averages only the 3d coords for camera pairs that both meet thresh.
	Returns nan for 2d COM if camera does not reach thresh. This should be
	detected by the generator to return nans such that bad camera
	frames do not get averaged in to image data
	"""
	if camera_mats is None and retriangulate:
		raise Exception("Need camera matrices to retriangulate")

	camnames = np.array(list(datadict[list(datadict.keys())[0]]['data'].keys()))

	# Because I repeat cameras to fill up 6 camera quota, I need grab only
	# the unique names
	_, idx = np.unique(camnames, return_index=True)
	uCamnames = camnames[np.sort(idx)]

	with open(comfile, 'rb') as f:
		com = cPickle.load(f)
	com3d_dict = {}

	if method == 'mean':
		print('using mean to get 3D COM')

	elif method == 'median':
		print('using median to get 3D COM')

	for key in com.keys():
		this_com = com[key]

		if key in datadict.keys():
			for k in range(len(camnames)):
				datadict[key]['data'][camnames[k]] = \
					this_com[camnames[k]]['COM'][:, np.newaxis].astype('float32')

				# Quick & dirty way to dynamically scale the confidence map output
				if conf_rescale is not None and camnames[k] in conf_rescale.keys():
					this_com[camnames[k]]['pred_max'] *= conf_rescale[camnames[k]]

				# then, set to nan
				if this_com[camnames[k]]['pred_max'] <= comthresh:
					datadict[key]['data'][camnames[k]][:] = np.nan

			com3d = np.zeros((3, int(comb(len(uCamnames), 2)))) * np.nan
			weights = np.zeros((int(comb(len(uCamnames), 2)),))
			cnt = 0
			for j in range(len(uCamnames)):
				for k in range(j + 1, len(uCamnames)):
					if retriangulate:
						do_retriangulate(this_com, j, k, uCamnames, camera_mats)
					if (this_com[uCamnames[j]]['pred_max'] > comthresh) and (
						this_com[uCamnames[k]]['pred_max'] > comthresh):
						com3d[:, cnt] = \
							this_com['triangulation']['{}_{}'.format(uCamnames[j], uCamnames[k])]
						weights[cnt] = \
							this_com[uCamnames[j]]['pred_max'] * this_com[uCamnames[k]]['pred_max']
					cnt += 1

			# weigts produces a weighted average of COM based on our overall confidence
			if weighted:
				if np.sum(weights) != 0:
					weights = weights / np.sum(weights)
					com3d = np.nansum(com3d * weights[np.newaxis, :], axis=1)
				else:
					com3d = np.zeros((3,)) * np.nan
			else:
				if method == 'mean':
					com3d = np.nanmean(com3d, axis=1)
				elif method == 'median':
					com3d = np.nanmedian(com3d, axis=1)
				else:
					raise Exception("Uknown 3D COM method")

			com3d_dict[key] = com3d
		else:
			warnings.warn("Key in COM file but not in datadict")

	return datadict, com3d_dict


def prepare_com3ddict(datadict_3d):
	"""Take the mean of the 3d data.

	Call this when using ground truth 3d anchor points that do not need to be
	loaded in via a special com file -- just need to take the mean
	of the 3d data with the 3d datadict
	"""
	com3d_dict = {}
	for key in datadict_3d.keys():
		com3d_dict[key] = np.nanmean(datadict_3d[key], axis=-1)
	return com3d_dict


def addCOM(d3d_dict, c3d_dict):
	"""Add COM back in to data.

	For JDM37 data and its ilk, the data are loaded in centered to the
	animal center of mass (Because they were predictions from the network)
	We need to add the COM back in, because durign training everything gets
	centered to the true COM again
	"""
	for key in c3d_dict.keys():
		d3d_dict[key] = d3d_dict[key] + c3d_dict[key][:, np.newaxis]
	return d3d_dict


def remove_samples(s, d3d, mode='clean', auxmode=None):
	"""Filter data structures for samples that meet inclusion criteria (mode).

	mode == 'clean' means only use samples in which all ground truth markers
			 are recorded
	mode == 'SpineM' means only remove data where SpineM is missing
	mode == 'liberal' means include any data that isn't *all* nan
	aucmode == 'JDM52d2' removes a really bad marker period -- samples 20k to 32k
	I need to cull the samples array (as this is used to index eveyrthing else),
	but also the
	data_3d_ array that is used to for finding clusters
	"""
	sample_mask = np.ones((len(s),), dtype='bool')

	if mode == 'clean':
		for i in range(len(s)):
			if np.isnan(np.sum(d3d[i])):
				sample_mask[i] = 0
	elif mode == 'liberal':
		for i in range(len(s)):
			if np.all(np.isnan(d3d[i])):
				sample_mask[i] = 0

	if auxmode == 'JDM52d2':
		print('removing bad JDM52d2 frames')
		for i in range(len(s)):
			if s[i] >= 20000 and s[i] <= 32000:
				sample_mask[i] = 0

	s = s[sample_mask]
	d3d = d3d[sample_mask]

	# zero the 3d data to SpineM
	d3d[:, ::3] -= d3d[:, 12:13]
	d3d[:, 1::3] -= d3d[:, 13:14]
	d3d[:, 2::3] -= d3d[:, 14:15]
	return s, d3d


def remove_samples_com(s, d3d, com3d_dict, cthresh=350, rmc=False):
	"""Remove any remaining samples in which the 3D COM estimates are nan.

	(i.e. no camera pair above threshold for a given frame)
	Also, let's remove any sample where abs(COM) is > 350
	"""
	sample_mask = np.ones((len(s),), dtype='bool')

	for i in range(len(s)):
		if np.isnan(np.sum(com3d_dict[s[i]])):
			sample_mask[i] = 0
		if rmc:
			if np.any(np.abs(com3d_dict[s[i]]) > cthresh):
				sample_mask[i] = 0

	s = s[sample_mask]
	d3d = d3d[sample_mask]
	return s, d3d


def add_experiment(
	experiment, samples_out, datadict_out, datadict_3d_out, com3d_dict_out,
	samples_in, datadict_in, datadict_3d_in, com3d_dict_in):
	"""Change variable names to satisfy naming convention.

	Append *_in variables to out variables, after appending the experiment
	number to the front of keys
	"""
	samples_in = [str(experiment) + '_' + str(x) for x in samples_in]
	samples_out = samples_out + samples_in

	for key in datadict_in.keys():
		datadict_out[str(experiment) + '_' + str(key)] = datadict_in[key]

	for key in datadict_3d_in.keys():
		datadict_3d_out[str(experiment) + '_' + str(key)] = datadict_3d_in[key]

	for key in com3d_dict_in.keys():
		com3d_dict_out[str(experiment) + '_' + str(key)] = com3d_dict_in[key]

	return samples_out, datadict_out, datadict_3d_out, com3d_dict_out
