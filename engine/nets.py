import tensorflow as tf
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Conv3D, Lambda, MaxPooling3D, Conv3DTranspose, Dense
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, GlobalAveragePooling3D, Add
from keras.layers.core import Activation, Permute, Reshape, Flatten
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, mean_squared_error
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.applications.vgg19 import VGG19

from keras.utils import multi_gpu_model

import ops

import gc

def refine_autoencoder(lossfunc,lr,input_dim,feature_num):
	inputs = Input((input_dim,))
	dense = Dense(320,activation='relu')(inputs)
	dense = Dense(160,activation='relu')(BatchNormalization(momentum=0.5)(dense))
	dense = Dense(80,activation='relu')(BatchNormalization(momentum=0.5)(dense))
	dense = Dense(20,activation='relu')(BatchNormalization(momentum=0.5)(dense))
	dense = Dense(80,activation='relu')(BatchNormalization(momentum=0.5)(dense))
	dense = Dense(160,activation='relu')(BatchNormalization(momentum=0.5)(dense))
	dense = Dense(320,activation='relu')(BatchNormalization(momentum=0.5)(dense))
	out = Dense(feature_num)(dense)

	model = Model(inputs=[inputs], outputs=[out])

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def refine_autoencoder_CNN(lossfunc,lr,input_dim,feature_num, input_length):
	inputs = Input((None,input_dim))
	conv = Conv1D(32,7,activation='relu',padding='same')(inputs)
	conv = MaxPooling1D(2)(conv)
	conv = Conv1D(64,5,activation='relu',padding='same')(BatchNormalization()(conv))
	conv = MaxPooling1D(2)(conv)
	conv = Conv1D(128,3,activation='relu',padding='same')(BatchNormalization()(conv))
	conv = MaxPooling1D(2)(conv)
	conv = Conv1D(256,3,activation='relu',padding='same')(BatchNormalization()(conv))
	conv = UpSampling1D(2)(conv)
	conv = Conv1D(128,3,activation='relu',padding='same')(BatchNormalization()(conv))
	conv = UpSampling1D(2)(conv)
	conv = Conv1D(64,5,activation='relu',padding='same')(BatchNormalization()(conv))
	conv = UpSampling1D(2)(conv)
	conv = Conv1D(32,7,activation='relu',padding='same')(BatchNormalization()(conv))
	out = Conv1D(feature_num,1)(conv)

	model = Model(inputs=[inputs], outputs=[out])

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet2d(lossfunc, lr, input_dim, feature_num):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')(conv1)
	conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(BatchNormalization()(conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(conv2)
	conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(conv3)
	conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(conv4)
	conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(BatchNormalization()(conv4))
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
	conv5 = Activation('relu')(conv5)
	conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
	conv5 = Activation('relu')(BatchNormalization()(conv5))

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), padding='same')(up6)
	conv6 = Activation('relu')(conv6)
	conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(BatchNormalization()(conv6))

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')(conv7)
	conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), padding='same')(up8)
	conv8 = Activation('relu')(conv8)
	conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), padding='same')(up9)
	conv9 = Activation('relu')(conv9)
	conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))

	conv10 = Conv2D(feature_num, (1, 1), activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet2d_fullbn(lossfunc, lr, input_dim, feature_num, metric='mse',multigpu=False, include_top = True):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')(BatchNormalization()(conv1))
	conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(BatchNormalization()(conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(BatchNormalization()(conv4))
	conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(BatchNormalization()(conv4))
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
	conv5 = Activation('relu')(BatchNormalization()(conv5))
	conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
	conv5 = Activation('relu')(BatchNormalization()(conv5))

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), padding='same')(up6)
	conv6 = Activation('relu')(BatchNormalization()(conv6))
	conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(BatchNormalization()(conv6))

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))
	conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), padding='same')(up8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))
	conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), padding='same')(up9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))
	conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))

	conv10 = Conv2D(feature_num, (1, 1), activation='sigmoid')(conv9)


	if include_top:
		model = Model(inputs=[inputs], outputs=[conv10])
	else:
		model = Model(inputs=[inputs], outputs=[conv9])

	if multigpu:
		model = multi_gpu_model(model,gpus=2)

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=[metric])

	return model

def unet2d_fullbn_vgg19_1024deep_linout(lossfunc, lr, input_dim, feature_num):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	base_model = VGG19(weights='imagenet', include_top=False)

	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')((conv1))
	conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
	conv1 = Activation('relu')((conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(BatchNormalization()(conv4))
	conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(BatchNormalization()(conv4))
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
	conv5 = Activation('relu')(BatchNormalization()(conv5))
	conv5 = Conv2D(1024, (3, 3), padding='same')(conv5)
	conv5 = Activation('relu')(BatchNormalization()(conv5))

	up7 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv7 = Conv2D(512, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))
	conv7 = Conv2D(512, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))

	up8 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7), conv3], axis=3)
	conv8 = Conv2D(256, (3, 3), padding='same')(up8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))
	conv8 = Conv2D(256, (3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))

	up9 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8), conv2], axis=3)
	conv9 = Conv2D(128, (3, 3), padding='same')(up9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))
	conv9 = Conv2D(128, (3, 3), padding='same')(conv9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))

	up10 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9), conv1], axis=3)
	conv10 = Conv2D(64, (3, 3), padding='same')(up10)
	conv10 = Activation('relu')(BatchNormalization()(conv10))
	conv10 = Conv2D(64, (3, 3), padding='same')(conv10)
	conv10 = Activation('relu')(BatchNormalization()(conv10))


	conv11 = Conv2D(feature_num, (1, 1), activation='linear')(conv10)

	model = Model(inputs=[inputs], outputs=[conv11])

	# Take the weights from first two layers of VGG and lock them
	model.layers[1].set_weights(base_model.layers[1].get_weights())
	model.layers[3].set_weights(base_model.layers[2].get_weights())

	model.layers[1].trainable = False
	model.layers[3].trainable = False

	del base_model
	gc.collect()

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet3d_big_slowramp_2gpu(lossfunc, lr, input_dim, feature_num, num_cams, batch_norm=False):

	if batch_norm:
		def fun(inputs):
			return BatchNormalization()(inputs)
	else:
		def fun(inputs):
			return inputs

	with tf.device('/gpu:0'):
		inputs = Input((None,None,None, input_dim*num_cams))
		conv1 = Conv3D(32, (3, 3, 3), padding='same')(inputs)
		conv1 = Activation('relu')(fun(conv1))
		conv1 = Conv3D(32, (3, 3, 3), padding='same')(conv1)
		conv1 = Activation('relu')(fun(conv1))
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

		conv2 = Conv3D(64, (3, 3, 3), padding='same')(pool1)
		conv2 = Activation('relu')(fun(conv2))
		conv2 = Conv3D(64, (3, 3, 3), padding='same')(conv2)
		conv2 = Activation('relu')(fun(conv2))
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

		conv3 = Conv3D(128, (3, 3, 3), padding='same')(pool2)
		conv3 = Activation('relu')(fun(conv3))
		conv3 = Conv3D(128, (3, 3, 3), padding='same')(conv3)
		conv3 = Activation('relu')(fun(conv3))
		pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

		conv4 = Conv3D(512, (3, 3, 3), padding='same')(pool3)
		conv4 = Activation('relu')(fun(conv4))


	with tf.device('/gpu:1'):
		conv4 = Conv3D(512, (3, 3, 3), padding='same')(conv4)
		conv4 = Activation('relu')(fun(conv4))
		up6 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
		conv6 = Conv3D(128, (3, 3, 3), padding='same')(up6)
		conv6 = Activation('relu')(fun(conv6))
		conv6 = Conv3D(128, (3, 3, 3), padding='same')(conv6)
		conv6 = Activation('relu')(fun(conv6))

		up7 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv2], axis=4)
		conv7 = Conv3D(64, (3, 3, 3), padding='same')(up7)
		conv7 = Activation('relu')(fun(conv7))
		conv7 = Conv3D(64, (3, 3, 3), padding='same')(conv7)
		conv7 = Activation('relu')(fun(conv7))

		up8 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv1], axis=4)
		conv8 = Conv3D(32, (3, 3, 3), padding='same')(up8)
		conv8 = Activation('relu')(fun(conv8))
		conv8 = Conv3D(32, (3, 3, 3), padding='same')(conv8)
		conv8 = Activation('relu')(fun(conv8))

		conv10 = Conv3D(feature_num, (1, 1, 1), activation='sigmoid')(conv8)

		model = Model(inputs=[inputs], outputs=[conv10])


	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet3d_big_1gpu_evalonly(lossfunc, lr, input_dim, feature_num, num_cams, batch_norm=False):

	if batch_norm:
		def fun(inputs):
			return BatchNormalization()(inputs)
	else:
		def fun(inputs):
			return inputs

	inputs = Input((None,None,None, input_dim*num_cams))
	conv1 = Conv3D(64, (3, 3, 3), padding='same')(inputs)
	conv1 = Activation('relu')(fun(conv1))
	conv1 = Conv3D(64, (3, 3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(fun(conv1))
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

	conv2 = Conv3D(128, (3, 3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(fun(conv2))
	conv2 = Conv3D(128, (3, 3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(fun(conv2))
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = Conv3D(256, (3, 3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(fun(conv3))
	conv3 = Conv3D(256, (3, 3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(fun(conv3))
	pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

	conv4 = Conv3D(512, (3, 3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(fun(conv4))
	conv4 = Conv3D(512, (3, 3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(fun(conv4))


	up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(up6)
	conv6 = Activation('relu')(fun(conv6))
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(fun(conv6))

	up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv2], axis=4)
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(up7)
	conv7 = Activation('relu')(fun(conv7))
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(fun(conv7))

	up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv1], axis=4)
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(up8)
	conv8 = Activation('relu')(fun(conv8))
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(fun(conv8))

	conv10 = Conv3D(feature_num, (1, 1, 1), activation='sigmoid')(conv8)

	model = Model(inputs=[inputs], outputs=[conv10])


	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet_2D_3D_nointer(lossfunc, lr, input_dim, feature_num, num_cams, 
	batch_norm=False, batch_size=3, imwidheight = (512,512),grid_size = 64*64*64, grid_dims=2, loss_weights=[0,1]):

	if batch_norm:
		def fun(inputs):
			return BatchNormalization()(inputs)
	else:
		def fun(inputs):
			return inputs

	with tf.device('/gpu:0'): #2D Unet
		inputs = Input((num_cams, imwidheight[0], imwidheight[1], input_dim))
		# input will enter net as shape (batch_size, num_cams, im_height, im_width, num_channels)
		# Need to reshape and collapse batch_size and num_cams so that we can use a single conv. decoder
		inputs_ = Lambda(lambda x: K.reshape(x,(batch_size*num_cams,
			K.int_shape(x)[2],K.int_shape(x)[3],K.int_shape(x)[4])))(inputs)

		conv1 = Conv2D(32, (3, 3), padding='same')(inputs_)
		conv1 = Activation('relu')(fun(conv1))
		conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
		conv1 = Activation('relu')(fun(conv1))
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
		conv2 = Activation('relu')(fun(conv2))
		conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
		conv2 = Activation('relu')(fun(conv2))
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
		conv3 = Activation('relu')(fun(conv3))
		conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
		conv3 = Activation('relu')(fun(conv3))

		up1 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
		conv4 = Conv2D(64, (3, 3), padding='same')(up1)
		conv4 = Activation('relu')(fun(conv4))
		conv4 = Conv2D(64, (3, 3), padding='same')(conv4)
		conv4 = Activation('relu')(fun(conv4))

		up2 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv4), conv1], axis=3)
		conv5 = Conv2D(32, (3, 3), padding='same')(up2)
		conv5 = Activation('relu')(fun(conv5))
		conv5 = Conv2D(32, (3, 3), padding='same')(conv5)
		conv5 = Activation('relu')(fun(conv5))

		conv6 = Conv2D(feature_num, (3, 3), activation='sigmoid',padding='same')(conv5)

	with tf.device('/gpu:1'):
		# Project up to 3D
		inputs_grid = Input((num_cams,grid_size,grid_dims))
		inputs_grid_= Lambda(lambda x: K.reshape(x,(batch_size*num_cams,
			K.int_shape(x)[2],K.int_shape(x)[3])))(inputs_grid)

		unprojected = Lambda(lambda x: ops.unproj(x[0],x[1], batch_size))([conv6, inputs_grid_])

		# Permute so that the cameras are last (they come out of unproject as second) and can be reshaped together with channels to act ast he last feature axis
		unprojected = Permute((2,3,4,5,1))(unprojected)
		unprojected = Reshape((K.int_shape(x)[1],K.int_shape(x)[2],K.int_shape(x)[3],-1))(unprojected)

		#Now lets do a small 3D UNet
		
		conv1_3D = Conv3D(32, (3, 3, 3), padding='same')(unprojected)
		conv1_3D = Activation('relu')(fun(conv1_3D))
		conv1_3D = Conv3D(32, (3, 3, 3), padding='same')(conv1_3D)
		conv1_3D = Activation('relu')(fun(conv1_3D))
		pool1_3D = MaxPooling3D(pool_size=(2, 2, 2))(conv1_3D)

		conv2_3D = Conv3D(64, (3, 3, 3), padding='same')(pool1_3D)
		conv2_3D = Activation('relu')(fun(conv2_3D))
		conv2_3D = Conv3D(64, (3, 3, 3), padding='same')(conv2_3D)
		conv2_3D = Activation('relu')(fun(conv2_3D))
		pool2_3D = MaxPooling3D(pool_size=(2, 2, 2))(conv2_3D)

		conv3_3D = Conv3D(128, (3, 3, 3), padding='same')(pool2_3D)
		conv3_3D = Activation('relu')(fun(conv3_3D))
		conv3_3D = Conv3D(128, (3, 3, 3), padding='same')(conv3_3D)
		conv3_3D = Activation('relu')(fun(conv3_3D))

		up1_3D = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv3_3D), conv2_3D], axis=4)
		conv4_3D = Conv3D(64, (3, 3, 3), padding='same')(up1_3D)
		conv4_3D = Activation('relu')(fun(conv4_3D))
		conv4_3D = Conv3D(64, (3, 3, 3), padding='same')(conv4_3D)
		conv4_3D = Activation('relu')(fun(conv4_3D))

		up2_3D = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4_3D), conv1_3D], axis=4)
		conv5_3D = Conv3D(32, (3, 3, 3), padding='same')(up2_3D)
		conv5_3D = Activation('relu')(fun(conv5_3D))
		conv5_3D = Conv3D(32, (3, 3, 3), padding='same')(conv5_3D)
		conv5_3D = Activation('relu')(fun(conv5_3D))

		conv6_3D = Conv3D(feature_num, (3, 3, 3), activation='sigmoid', padding='same')(conv5_3D)

		model = Model(inputs=[inputs, inputs_grid], outputs=[conv6_3D])


	model.compile(optimizer=Adam(lr=lr), loss=[lossfunc], loss_weights=loss_weights)

	return model

def unet_2D_3D(lossfunc, lr, input_dim, feature_num, num_cams, 
	batch_norm=False, batch_size=3, imwidheight = (512,512),grid_size = 64*64*64, grid_dims=2, loss_weights=[0,1]):

	if batch_norm:
		def fun(inputs):
			return BatchNormalization()(inputs)
	else:
		def fun(inputs):
			return inputs

	with tf.device('/gpu:0'): #2D Unet
		inputs = Input((num_cams, imwidheight[0], imwidheight[1], input_dim))
		# input will enter net as shape (batch_size, num_cams, im_height, im_width, num_channels)
		# Need to reshape and collapse batch_size and num_cams so that we can use a single conv. decoder
		inputs_ = Lambda(lambda x: K.reshape(x,(batch_size*num_cams,
			K.int_shape(x)[2],K.int_shape(x)[3],K.int_shape(x)[4])))(inputs)

		conv1 = Conv2D(32, (3, 3), padding='same')(inputs_)
		conv1 = Activation('relu')(fun(conv1))
		conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
		conv1 = Activation('relu')(fun(conv1))
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
		conv2 = Activation('relu')(fun(conv2))
		conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
		conv2 = Activation('relu')(fun(conv2))
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
		conv3 = Activation('relu')(fun(conv3))
		conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
		conv3 = Activation('relu')(fun(conv3))

		up1 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
		conv4 = Conv2D(64, (3, 3), padding='same')(up1)
		conv4 = Activation('relu')(fun(conv4))
		conv4 = Conv2D(64, (3, 3), padding='same')(conv4)
		conv4 = Activation('relu')(fun(conv4))

		up2 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv4), conv1], axis=3)
		conv5 = Conv2D(32, (3, 3), padding='same')(up2)
		conv5 = Activation('relu')(fun(conv5))
		conv5 = Conv2D(32, (3, 3), padding='same')(conv5)
		conv5 = Activation('relu')(fun(conv5))

		# Collect 2-D predictions and supervise with intermediate loss
		conv6 = Conv2D(feature_num, (3, 3), activation='sigmoid',padding='same')(conv5)
		conv6_out = Lambda(lambda x: K.reshape(x,[batch_size,num_cams,imwidheight[0], imwidheight[1], feature_num]))(conv6)

		# Before unprojection, concenate both readout and internal features
		concat_feats = concatenate([conv5, conv6],axis=-1)

	with tf.device('/gpu:1'):
		# Project up to 3D
		inputs_grid = Input((num_cams,grid_size,grid_dims))
		inputs_grid_= Lambda(lambda x: K.reshape(x,(batch_size*num_cams,
			K.int_shape(x)[2],K.int_shape(x)[3])))(inputs_grid)

		unprojected = Lambda(lambda x: ops.unproj(x[0],x[1], batch_size))([concat_feats, inputs_grid_])

		# Average over cameras
		unprojected = Lambda(lambda x: K.mean(x,axis=1))(unprojected)

		#Now lets do a small 3D UNet
		
		conv1_3D = Conv3D(32, (3, 3, 3), padding='same')(unprojected)
		conv1_3D = Activation('relu')(fun(conv1_3D))
		conv1_3D = Conv3D(32, (3, 3, 3), padding='same')(conv1_3D)
		conv1_3D = Activation('relu')(fun(conv1_3D))
		pool1_3D = MaxPooling3D(pool_size=(2, 2, 2))(conv1_3D)

		conv2_3D = Conv3D(64, (3, 3, 3), padding='same')(pool1_3D)
		conv2_3D = Activation('relu')(fun(conv2_3D))
		conv2_3D = Conv3D(64, (3, 3, 3), padding='same')(conv2_3D)
		conv2_3D = Activation('relu')(fun(conv2_3D))
		pool2_3D = MaxPooling3D(pool_size=(2, 2, 2))(conv2_3D)

		conv3_3D = Conv3D(128, (3, 3, 3), padding='same')(pool2_3D)
		conv3_3D = Activation('relu')(fun(conv3_3D))
		conv3_3D = Conv3D(128, (3, 3, 3), padding='same')(conv3_3D)
		conv3_3D = Activation('relu')(fun(conv3_3D))

		up1_3D = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv3_3D), conv2_3D], axis=4)
		conv4_3D = Conv3D(64, (3, 3, 3), padding='same')(up1_3D)
		conv4_3D = Activation('relu')(fun(conv4_3D))
		conv4_3D = Conv3D(64, (3, 3, 3), padding='same')(conv4_3D)
		conv4_3D = Activation('relu')(fun(conv4_3D))

		up2_3D = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4_3D), conv1_3D], axis=4)
		conv5_3D = Conv3D(32, (3, 3, 3), padding='same')(up2_3D)
		conv5_3D = Activation('relu')(fun(conv5_3D))
		conv5_3D = Conv3D(32, (3, 3, 3), padding='same')(conv5_3D)
		conv5_3D = Activation('relu')(fun(conv5_3D))

		conv6_3D = Conv3D(feature_num, (3, 3, 3), activation='sigmoid', padding='same')(conv5_3D)

		model = Model(inputs=[inputs, inputs_grid], outputs=[conv6_out, conv6_3D])


	model.compile(optimizer=Adam(lr=lr), loss=[lossfunc, lossfunc], loss_weights=loss_weights)

	return model

def unet3d_big_expectedvalue(lossfunc, lr, input_dim, feature_num, num_cams, gridsize=(64,64,64),
											batch_norm=False, instance_norm = False, include_top=True, regularize_var=False,
											loss_weights = None, metric = ['mse'],out_kernel = (1,1,1)):

	if batch_norm and not instance_norm:
		print('using batch normalization')
		def fun(inputs):
			return BatchNormalization()(inputs)
	elif instance_norm:
		print('using instance normalization')
		def fun(inputs):
			return ops.InstanceNormalization()(inputs)
	else:
		def fun(inputs):
			return inputs

	inputs = Input((*gridsize, input_dim*num_cams))
	conv1_layer = Conv3D(64, (3, 3, 3), padding='same')

	conv1 = conv1_layer(inputs)
	conv1 = Activation('relu')(fun(conv1))
	conv1 = Conv3D(64, (3, 3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(fun(conv1))
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

	conv2 = Conv3D(128, (3, 3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(fun(conv2))
	conv2 = Conv3D(128, (3, 3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(fun(conv2))
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = Conv3D(256, (3, 3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(fun(conv3))
	conv3 = Conv3D(256, (3, 3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(fun(conv3))
	pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

	conv4 = Conv3D(512, (3, 3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(fun(conv4))
	conv4 = Conv3D(512, (3, 3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(fun(conv4))


	up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(up6)
	conv6 = Activation('relu')(fun(conv6))
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(fun(conv6))

	up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv2], axis=4)
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(up7)
	conv7 = Activation('relu')(fun(conv7))
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(fun(conv7))

	up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv1], axis=4)
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(up8)
	conv8 = Activation('relu')(fun(conv8))
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(fun(conv8))

	conv10 = Conv3D(feature_num, out_kernel, activation='linear',padding='same')(conv8)

	grid_centers = Input((None,3))

	conv10 = Lambda(lambda x: ops.spatial_softmax(x))(conv10)

	output = Lambda(lambda x: ops.expected_value_3d(x[0],x[1]))([conv10, grid_centers])

	#Because I think it is easier, use a layer to calculate the variance and return it as a second output to be used for variance loss

	output_var = Lambda(lambda x: ops.var_3d(x[0],x[1],x[2]))([conv10, grid_centers,output])

	if include_top:
		if regularize_var:
			model = Model(inputs=[inputs, grid_centers], outputs=[output, output_var])
		else:
			model = Model(inputs=[inputs, grid_centers], outputs=[output])
	else:
		model = Model(inputs=[inputs], outputs=[conv8])


	# model.compile(optimizer=Adam(lr=lr), loss=[lossfunc[0], lossfunc[1]], metrics=['mse'])
	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=metric,loss_weights=loss_weights)

	return model

def slice_input(inp,k):
	print(K.int_shape(inp))
	return inp[:,:,:,:,k*input_dim:(k+1)*input_dim]

def unet3d_big_tiedfirstlayer_expectedvalue(lossfunc, lr, input_dim, feature_num, num_cams, gridsize=(64,64,64),
											batch_norm=False, instance_norm = False, include_top=True, regularize_var=False,
											loss_weights = None, metric = 'mse'):

	if batch_norm and not instance_norm:
		print('using batch normalization')
		def fun(inputs):
			return BatchNormalization()(inputs)
	elif instance_norm:
		print('using instance normalization')
		def fun(inputs):
			return ops.InstanceNormalization()(inputs)
	else:
		def fun(inputs):
			return inputs

	def slice_input(inp,k):
		print(K.int_shape(inp))
		return inp[:,:,:,:,k*input_dim:(k+1)*input_dim]

	inputs = Input((*gridsize, input_dim*num_cams))
	conv1_layer = Conv3D(64, (3, 3, 3), padding='same')

	conv1_in = []
	for i in range(num_cams):
		# conv1_in.append(conv1_layer(inputs[:,:,:,:,i*input_dim:(i+1)*input_dim]))
		conv1_in.append(conv1_layer(Lambda(lambda x: slice_input(x,i))(inputs)))

	conv1 = Add()(conv1_in)
	conv1 = Activation('relu')(fun(conv1))
	conv1 = Conv3D(64, (3, 3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(fun(conv1))
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

	conv2 = Conv3D(128, (3, 3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(fun(conv2))
	conv2 = Conv3D(128, (3, 3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(fun(conv2))
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = Conv3D(256, (3, 3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(fun(conv3))
	conv3 = Conv3D(256, (3, 3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(fun(conv3))
	pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

	conv4 = Conv3D(512, (3, 3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(fun(conv4))
	conv4 = Conv3D(512, (3, 3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(fun(conv4))


	up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(up6)
	conv6 = Activation('relu')(fun(conv6))
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(fun(conv6))

	up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv2], axis=4)
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(up7)
	conv7 = Activation('relu')(fun(conv7))
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(fun(conv7))

	up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv1], axis=4)
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(up8)
	conv8 = Activation('relu')(fun(conv8))
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(fun(conv8))

	conv10 = Conv3D(feature_num, (1, 1, 1), activation='linear')(conv8)

	grid_centers = Input((None,3))

	conv10 = Lambda(lambda x: ops.spatial_softmax(x))(conv10)

	output = Lambda(lambda x: ops.expected_value_3d(x[0],x[1]))([conv10, grid_centers])

	#Because I think it is easier, use a layer to calculate the variance and return it as a second output to be used for variance loss

	output_var = Lambda(lambda x: ops.var_3d(x[0],x[1],x[2]))([conv10, grid_centers,output])

	if include_top:
		if regularize_var:
			model = Model(inputs=[inputs, grid_centers], outputs=[output, output_var])
		else:
			model = Model(inputs=[inputs, grid_centers], outputs=[output])
	else:
		model = Model(inputs=[inputs], outputs=[conv8])


	# model.compile(optimizer=Adam(lr=lr), loss=[lossfunc[0], lossfunc[1]], metrics=['mse'])
	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=[metric],loss_weights=loss_weights)

	return model

def unet3d_big_1cam(lossfunc, lr, input_dim, feature_num, num_cams, batch_norm=False, instance_norm = False):

	if batch_norm and not instance_norm:
		print('using batch normalization')
		def fun(inputs):
			return BatchNormalization()(inputs)
	elif instance_norm:
		print('using instance normalization')
		def fun(inputs):
			return ops.InstanceNormalization()(inputs)
	else:
		def fun(inputs):
			return inputs

	inputs = Input((None,None,None, input_dim))
	conv1_layer = Conv3D(64, (3, 3, 3), padding='same')

	conv1 = conv1_layer(inputs)
	conv1 = Activation('relu')(fun(conv1))
	conv1 = Conv3D(64, (3, 3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(fun(conv1))
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

	conv2 = Conv3D(128, (3, 3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(fun(conv2))
	conv2 = Conv3D(128, (3, 3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(fun(conv2))
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = Conv3D(256, (3, 3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(fun(conv3))
	conv3 = Conv3D(256, (3, 3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(fun(conv3))
	pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

	conv4 = Conv3D(512, (3, 3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(fun(conv4))
	conv4 = Conv3D(512, (3, 3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(fun(conv4))


	up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(up6)
	conv6 = Activation('relu')(fun(conv6))
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(fun(conv6))

	up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv2], axis=4)
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(up7)
	conv7 = Activation('relu')(fun(conv7))
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(fun(conv7))

	up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv1], axis=4)
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(up8)
	conv8 = Activation('relu')(fun(conv8))
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(fun(conv8))

	conv10 = Conv3D(feature_num, (1, 1, 1), activation='sigmoid')(conv8)

	model = Model(inputs=[inputs], outputs=[conv10])


	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet3d_big_tiedfirstlayer_linout(lossfunc, lr, input_dim, feature_num, num_cams, batch_norm=False, instance_norm = False, bs=6):

	if batch_norm and not instance_norm:
		print('using batch normalization')
		def fun(inputs):
			return BatchNormalization()(inputs)
	elif instance_norm:
		print('using instance normalization')
		def fun(inputs):
			return ops.InstanceNormalization()(inputs)
	else:
		def fun(inputs):
			return inputs

	def slice_input(inp,k):
		print(K.int_shape(inp))
		return inp[:,:,:,:,k*input_dim:(k+1)*input_dim]

	inputs = Input((None,None,None, input_dim*num_cams))
	conv1_layer = Conv3D(64, (3, 3, 3), padding='same')

	conv1_in = []
	for i in range(num_cams):
		# conv1_in.append(conv1_layer(inputs[:,:,:,:,i*input_dim:(i+1)*input_dim]))
		conv1_in.append(conv1_layer(Lambda(lambda x: slice_input(x,i))(inputs)))

	conv1 = Add()(conv1_in)
	conv1 = Activation('relu')(fun(conv1))
	conv1 = Conv3D(64, (3, 3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(fun(conv1))
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

	conv2 = Conv3D(128, (3, 3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(fun(conv2))
	conv2 = Conv3D(128, (3, 3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(fun(conv2))
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = Conv3D(256, (3, 3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(fun(conv3))
	conv3 = Conv3D(256, (3, 3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(fun(conv3))
	pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

	conv4 = Conv3D(512, (3, 3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(fun(conv4))
	conv4 = Conv3D(512, (3, 3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(fun(conv4))


	up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(up6)
	conv6 = Activation('relu')(fun(conv6))
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(fun(conv6))

	up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv2], axis=4)
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(up7)
	conv7 = Activation('relu')(fun(conv7))
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(fun(conv7))

	up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv1], axis=4)
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(up8)
	conv8 = Activation('relu')(fun(conv8))
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(fun(conv8))

	conv10 = Conv3D(feature_num, (1, 1, 1), activation='linear')(conv8)

	model = Model(inputs=[inputs], outputs=[conv10])


	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet3d_big_tiedfirstlayer(lossfunc, lr, input_dim, feature_num, num_cams, batch_norm=False, instance_norm = False, bs=6):

	if batch_norm and not instance_norm:
		print('using batch normalization')
		def fun(inputs):
			return BatchNormalization()(inputs)
	elif instance_norm:
		print('using instance normalization')
		def fun(inputs):
			return ops.InstanceNormalization()(inputs)
	else:
		def fun(inputs):
			return inputs

	def slice_input(inp,k):
		print(K.int_shape(inp))
		return inp[:,:,:,:,k*input_dim:(k+1)*input_dim]

	inputs = Input((None,None,None, input_dim*num_cams))
	conv1_layer = Conv3D(64, (3, 3, 3), padding='same')

	conv1_in = []
	for i in range(num_cams):
		# conv1_in.append(conv1_layer(inputs[:,:,:,:,i*input_dim:(i+1)*input_dim]))
		conv1_in.append(conv1_layer(Lambda(lambda x: slice_input(x,i))(inputs)))

	conv1 = Add()(conv1_in)
	conv1 = Activation('relu')(fun(conv1))
	conv1 = Conv3D(64, (3, 3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(fun(conv1))
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

	conv2 = Conv3D(128, (3, 3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(fun(conv2))
	conv2 = Conv3D(128, (3, 3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(fun(conv2))
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = Conv3D(256, (3, 3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(fun(conv3))
	conv3 = Conv3D(256, (3, 3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(fun(conv3))
	pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

	conv4 = Conv3D(512, (3, 3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(fun(conv4))
	conv4 = Conv3D(512, (3, 3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(fun(conv4))


	up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(up6)
	conv6 = Activation('relu')(fun(conv6))
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(fun(conv6))

	up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv2], axis=4)
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(up7)
	conv7 = Activation('relu')(fun(conv7))
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(fun(conv7))

	up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv1], axis=4)
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(up8)
	conv8 = Activation('relu')(fun(conv8))
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(fun(conv8))

	conv10 = Conv3D(feature_num, (1, 1, 1), activation='sigmoid')(conv8)

	model = Model(inputs=[inputs], outputs=[conv10])


	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet3d_big_2gpu(lossfunc, lr, input_dim, feature_num, num_cams, batch_norm=False, instance_norm = False, include_top=True, last_kern_size=(1,1,1)):

	if batch_norm and not instance_norm:
		print('using batch normalization')
		def fun(inputs):
			return BatchNormalization()(inputs)
	elif instance_norm:
		print('using instance normalization')
		def fun(inputs):
			return ops.InstanceNormalization()(inputs)
	else:
		def fun(inputs):
			return inputs

	with tf.device('/gpu:0'):
		inputs = Input((None,None,None, input_dim*num_cams))
		conv1 = Conv3D(64, (3, 3, 3), padding='same')(inputs)
		conv1 = Activation('relu')(fun(conv1))
		conv1 = Conv3D(64, (3, 3, 3), padding='same')(conv1)
		conv1 = Activation('relu')(fun(conv1))
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

		conv2 = Conv3D(128, (3, 3, 3), padding='same')(pool1)
		conv2 = Activation('relu')(fun(conv2))
		conv2 = Conv3D(128, (3, 3, 3), padding='same')(conv2)
		conv2 = Activation('relu')(fun(conv2))
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

		conv3 = Conv3D(256, (3, 3, 3), padding='same')(pool2)
		conv3 = Activation('relu')(fun(conv3))
		conv3 = Conv3D(256, (3, 3, 3), padding='same')(conv3)
		conv3 = Activation('relu')(fun(conv3))
		pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

		conv4 = Conv3D(512, (3, 3, 3), padding='same')(pool3)
		conv4 = Activation('relu')(fun(conv4))
		conv4 = Conv3D(512, (3, 3, 3), padding='same')(conv4)
		conv4 = Activation('relu')(fun(conv4))

	with tf.device('/gpu:1'):

		up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
		conv6 = Conv3D(256, (3, 3, 3), padding='same')(up6)
		conv6 = Activation('relu')(fun(conv6))
		conv6 = Conv3D(256, (3, 3, 3), padding='same')(conv6)
		conv6 = Activation('relu')(fun(conv6))

		up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv2], axis=4)
		conv7 = Conv3D(128, (3, 3, 3), padding='same')(up7)
		conv7 = Activation('relu')(fun(conv7))
		conv7 = Conv3D(128, (3, 3, 3), padding='same')(conv7)
		conv7 = Activation('relu')(fun(conv7))

		up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv1], axis=4)
		conv8 = Conv3D(64, (3, 3, 3), padding='same')(up8)
		conv8 = Activation('relu')(fun(conv8))
		conv8 = Conv3D(64, (3, 3, 3), padding='same')(conv8)
		conv8 = Activation('relu')(fun(conv8))

		conv10 = Conv3D(feature_num, last_kern_size, activation='sigmoid')(conv8)

		if include_top:
			model = Model(inputs=[inputs], outputs=[conv10])
		else:
			model = Model(inputs=[inputs], outputs=[conv8])


	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def test_project(lossfunc, lr, input_dim, feature_num, num_cams, num_grids, batch_norm=False, 
	instance_norm = False, vmin=-120,vmax=120, nvox=64, outsize=512, batch_size=1):

	inputs = Input(batch_shape=(batch_size,num_grids,nvox,nvox,nvox,feature_num))

	conv10 = inputs#Lambda(lambda x: K.expand_dims(x,axis=1))(inputs)

	inputs_K = Input(batch_shape=(batch_size,num_cams,3,3))
	inputs_R = Input(batch_shape=(batch_size,num_cams,3,4))
	inputs_imrs = Input(batch_shape=(batch_size,num_cams,3,outsize**2))
	inputs_vmax = Input(batch_shape=(batch_size,2,3))

	output = Lambda(lambda x: ops.proj_slice(x[0][:,0,:],x[0][:,1,:],nvox,x[1],x[2],x[3],x[4], outsize))([inputs_vmax,inputs_imrs, conv10, inputs_K, inputs_R])

	#remove the singleton dim
	output = Lambda(lambda x: K.squeeze(x,axis=1))(output)

	model = Model(inputs=[inputs,inputs_K,inputs_R,inputs_imrs,inputs_vmax], outputs=[output])

	return model

def unet3d_project(lossfunc, lr, input_dim, feature_num, num_cams, batch_norm=False, 
	instance_norm = False, vmin=-120,vmax=120, nvox=64, outsize=512, batch_size=4):

	if batch_norm and not instance_norm:
		print('using batch normalization')
		def fun(inputs):
			return BatchNormalization()(inputs)
	elif instance_norm:
		print('using instance normalization')
		def fun(inputs):
			return ops.InstanceNormalization()(inputs)
	else:
		def fun(inputs):
			return inputs

	with tf.device('/gpu:0'):
		inputs = Input(batch_shape=(batch_size,nvox,nvox,nvox, input_dim*num_cams))
		conv1 = Conv3D(64, (3, 3, 3), padding='same')(inputs)
		conv1 = Activation('relu')(fun(conv1))
		conv1 = Conv3D(64, (3, 3, 3), padding='same')(conv1)
		conv1 = Activation('relu')(fun(conv1))
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

		conv2 = Conv3D(128, (3, 3, 3), padding='same')(pool1)
		conv2 = Activation('relu')(fun(conv2))
		conv2 = Conv3D(128, (3, 3, 3), padding='same')(conv2)
		conv2 = Activation('relu')(fun(conv2))
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

		conv3 = Conv3D(256, (3, 3, 3), padding='same')(pool2)
		conv3 = Activation('relu')(fun(conv3))
		conv3 = Conv3D(256, (3, 3, 3), padding='same')(conv3)
		conv3 = Activation('relu')(fun(conv3))
		pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

		conv4 = Conv3D(512, (3, 3, 3), padding='same')(pool3)
		conv4 = Activation('relu')(fun(conv4))
		conv4 = Conv3D(512, (3, 3, 3), padding='same')(conv4)
		conv4 = Activation('relu')(fun(conv4))

	with tf.device('/gpu:1'):

		up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
		conv6 = Conv3D(256, (3, 3, 3), padding='same')(up6)
		conv6 = Activation('relu')(fun(conv6))
		conv6 = Conv3D(256, (3, 3, 3), padding='same')(conv6)
		conv6 = Activation('relu')(fun(conv6))

		up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv2], axis=4)
		conv7 = Conv3D(128, (3, 3, 3), padding='same')(up7)
		conv7 = Activation('relu')(fun(conv7))
		conv7 = Conv3D(128, (3, 3, 3), padding='same')(conv7)
		conv7 = Activation('relu')(fun(conv7))

		up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv1], axis=4)
		conv8 = Conv3D(64, (3, 3, 3), padding='same')(up8)
		conv8 = Activation('relu')(fun(conv8))
		conv8 = Conv3D(64, (3, 3, 3), padding='same')(conv8)
		conv8 = Activation('relu')(fun(conv8))

		conv10 = Conv3D(feature_num, (1, 1, 1), activation='sigmoid')(conv8)

		# expand dims as expected by proj_slice
		conv10 = Lambda(lambda x: K.expand_dims(x,axis=1))(conv10)

		inputs_K = Input(batch_shape=(batch_size,num_cams,3,3))
		inputs_R = Input(batch_shape=(batch_size,num_cams,3,4))
		inputs_imrs = Input(batch_shape=(batch_size,num_cams,3,outsize**2))
		inputs_vmax = Input(batch_shape=(batch_size,2))

		output = Lambda(lambda x: ops.proj_slice(x[0][:,0],x[0][:,1],nvox,x[1],x[2],x[3],x[4], outsize))([inputs_vmax,inputs_imrs, conv10, inputs_K, inputs_R])

		#remove the singleton dim
		output = Lambda(lambda x: K.squeeze(x,axis=1))(output)

		model = Model(inputs=[inputs,inputs_K,inputs_R,inputs_imrs,inputs_vmax], outputs=[output])


	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model


def unet3d(lossfunc, lr, input_dim, feature_num, filt_size_3d, batch_size, grid_size, num_cams):
	inputs = Input((None,None,None, input_dim*num_cams))
	conv1 = Conv3D(32, (3, 3, 3), padding='same')(inputs)
	conv1 = Activation('relu')(BatchNormalization()(conv1))
	conv1 = Conv3D(32, (3, 3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(BatchNormalization()(conv1))
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

	conv2 = Conv3D(64, (3, 3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	conv2 = Conv3D(64, (3, 3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = Conv3D(128, (3, 3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	conv3 = Conv3D(128, (3, 3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(BatchNormalization()(conv3))

	up6 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv3), conv2], axis=4)
	conv6 = Conv3D(64, (3, 3, 3), padding='same')(up6)
	conv6 = Activation('relu')(BatchNormalization()(conv6))
	conv6 = Conv3D(64, (3, 3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(BatchNormalization()(conv6))

	up7 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv1], axis=4)
	conv7 = Conv3D(32, (3, 3, 3), padding='same')(up7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))
	conv7 = Conv3D(32, (3, 3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))

	conv10 = Conv3D(feature_num, (1, 1, 1), activation='sigmoid')(conv7)

	model = Model(inputs=[inputs], outputs=[conv10])

	# try multi_gpu
	model = multi_gpu_model(model,gpus=2)

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet3d_nobn(lossfunc, lr, input_dim, feature_num, filt_size_3d, batch_size, grid_size, num_cams):
	inputs = Input((None,None,None, input_dim*num_cams))
	conv1 = Conv3D(32, (3, 3, 3), padding='same')(inputs)
	conv1 = Activation('relu')((conv1))
	conv1 = Conv3D(32, (3, 3, 3), padding='same')(conv1)
	conv1 = Activation('relu')((conv1))
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

	conv2 = Conv3D(64, (3, 3, 3), padding='same')(pool1)
	conv2 = Activation('relu')((conv2))
	conv2 = Conv3D(64, (3, 3, 3), padding='same')(conv2)
	conv2 = Activation('relu')((conv2))
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = Conv3D(128, (3, 3, 3), padding='same')(pool2)
	conv3 = Activation('relu')((conv3))
	conv3 = Conv3D(128, (3, 3, 3), padding='same')(conv3)
	conv3 = Activation('relu')((conv3))

	up6 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv3), conv2], axis=4)
	conv6 = Conv3D(64, (3, 3, 3), padding='same')(up6)
	conv6 = Activation('relu')((conv6))
	conv6 = Conv3D(64, (3, 3, 3), padding='same')(conv6)
	conv6 = Activation('relu')((conv6))

	up7 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv1], axis=4)
	conv7 = Conv3D(32, (3, 3, 3), padding='same')(up7)
	conv7 = Activation('relu')((conv7))
	conv7 = Conv3D(32, (3, 3, 3), padding='same')(conv7)
	conv7 = Activation('relu')((conv7))

	conv10 = Conv3D(feature_num, (1, 1, 1), activation='sigmoid')(conv7)

	model = Model(inputs=[inputs], outputs=[conv10])

	# try multi_gpu
	#model = multi_gpu_model(model,gpus=2)

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet2d_fullbn_vgg19_1024deep_3d_2gpu(lossfunc, lr, input_dim, feature_num, filt_size_3d, batch_size, grid_size, num_cams):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	base_model = VGG19(weights='imagenet', include_top=False)

	with tf.device('/gpu:0'):
		inputs = Input((num_cams, 512, 512, input_dim))
		inputs_ = Lambda(lambda x: K.reshape(x,(num_cams*batch_size,512,512,input_dim)))(inputs)
		conv1 = Conv2D(64, (3, 3), padding='same')(inputs_)
		conv1 = Activation('relu')((conv1))
		conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
		conv1 = Activation('relu')((conv1))
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
		conv2 = Activation('relu')(BatchNormalization()(conv2))
		conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
		conv2 = Activation('relu')(BatchNormalization()(conv2))
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
		conv3 = Activation('relu')(BatchNormalization()(conv3))
		conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
		conv3 = Activation('relu')(BatchNormalization()(conv3))
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
		conv4 = Activation('relu')(BatchNormalization()(conv4))
		conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
		conv4 = Activation('relu')(BatchNormalization()(conv4))
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
		conv5 = Activation('relu')(BatchNormalization()(conv5))
		conv5 = Conv2D(1024, (3, 3), padding='same')(conv5)
		conv5 = Activation('relu')(BatchNormalization()(conv5))

		up7 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
		conv7 = Conv2D(512, (3, 3), padding='same')(up7)
		conv7 = Activation('relu')(BatchNormalization()(conv7))
		conv7 = Conv2D(512, (3, 3), padding='same')(conv7)
		conv7 = Activation('relu')(BatchNormalization()(conv7))

		up8 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7), conv3], axis=3)
		conv8 = Conv2D(256, (3, 3), padding='same')(up8)
		conv8 = Activation('relu')(BatchNormalization()(conv8))
		conv8 = Conv2D(256, (3, 3), padding='same')(conv8)
		conv8 = Activation('relu')(BatchNormalization()(conv8))

		up9 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8), conv2], axis=3)
		conv9 = Conv2D(128, (3, 3), padding='same')(up9)
		conv9 = Activation('relu')(BatchNormalization()(conv9))
		conv9 = Conv2D(128, (3, 3), padding='same')(conv9)
		conv9 = Activation('relu')(BatchNormalization()(conv9))

		up10 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9), conv1], axis=3)
		conv10 = Conv2D(64, (3, 3), padding='same')(up10)
		conv10 = Activation('relu')(BatchNormalization()(conv10))
		conv10 = Conv2D(64, (3, 3), padding='same')(conv10)
		conv10 = Activation('relu')(BatchNormalization()(conv10))


		#conv11 = Conv2D(feature_num, (1, 1), activation='sigmoid')(conv10)

		model = Model(inputs=[inputs], outputs=[conv10])

		# Take the weights from first two layers of VGG and lock them
		model.layers[2].set_weights(base_model.layers[1].get_weights())
		model.layers[4].set_weights(base_model.layers[2].get_weights())

		model.layers[2].trainable = False
		model.layers[4].trainable = False

		del base_model
		gc.collect()

	with tf.device('/gpu:1'):
		input2 = model(inputs)
		
		output1 = Conv2D(feature_num, (1, 1), activation='sigmoid')(input2)
		output1 = Lambda(lambda x: K.expand_dims(x,axis=0))(output1)

		#For the second part of the net, we need to reshape batch-wise
		input2 = Lambda(lambda x: K.reshape(x,(batch_size,num_cams,
			K.int_shape(x)[1],K.int_shape(x)[2],K.int_shape(x)[3])))(input2)
		#Permute so that the batch and channel dimensions are together and can be reshaped to concatenate

		input2 = Permute((2,3,4,1))(input2)

		input2 = Reshape((K.int_shape(input2)[1],K.int_shape(input2)[2],-1))(input2)

		bigconv = Conv2D(feature_num*grid_size,filt_size_3d,strides=filt_size_3d,padding='same',activation='sigmoid')(input2)

		bigconv = Reshape((grid_size,grid_size,grid_size,feature_num))(bigconv)

		model_bigconv = Model(inputs=[inputs],outputs=[output1, bigconv])

	model_bigconv.compile(optimizer=Adam(lr=lr), loss=lossfunc)

	return model_bigconv

def unet2d_fullbn_vgg19_1024deep_3d_2gpu_dilation(lossfunc, lr, input_dim, feature_num, filt_size_3d, batch_size, grid_size, num_cams):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	base_model = VGG19(weights='imagenet', include_top=False)

	with tf.device('/gpu:0'):
		inputs = Input((num_cams, 512, 512, input_dim))
		inputs_ = Lambda(lambda x: K.reshape(x,(num_cams*batch_size,512,512,input_dim)))(inputs)
		conv1 = Conv2D(64, (3, 3), padding='same')(inputs_)
		conv1 = Activation('relu')((conv1))
		conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
		conv1 = Activation('relu')((conv1))
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
		conv2 = Activation('relu')(BatchNormalization()(conv2))
		conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
		conv2 = Activation('relu')(BatchNormalization()(conv2))
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
		conv3 = Activation('relu')(BatchNormalization()(conv3))
		conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
		conv3 = Activation('relu')(BatchNormalization()(conv3))
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
		conv4 = Activation('relu')(BatchNormalization()(conv4))
		conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
		conv4 = Activation('relu')(BatchNormalization()(conv4))
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
		conv5 = Activation('relu')(BatchNormalization()(conv5))
		conv5 = Conv2D(1024, (3, 3), padding='same')(conv5)
		conv5 = Activation('relu')(BatchNormalization()(conv5))

		up7 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
		conv7 = Conv2D(512, (3, 3), padding='same')(up7)
		conv7 = Activation('relu')(BatchNormalization()(conv7))
		conv7 = Conv2D(512, (3, 3), padding='same')(conv7)
		conv7 = Activation('relu')(BatchNormalization()(conv7))

		up8 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7), conv3], axis=3)
		conv8 = Conv2D(256, (3, 3), padding='same')(up8)
		conv8 = Activation('relu')(BatchNormalization()(conv8))
		conv8 = Conv2D(256, (3, 3), padding='same')(conv8)
		conv8 = Activation('relu')(BatchNormalization()(conv8))

		up9 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8), conv2], axis=3)
		conv9 = Conv2D(128, (3, 3), padding='same')(up9)
		conv9 = Activation('relu')(BatchNormalization()(conv9))
		conv9 = Conv2D(128, (3, 3), padding='same')(conv9)
		conv9 = Activation('relu')(BatchNormalization()(conv9))

		up10 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9), conv1], axis=3)
		conv10 = Conv2D(64, (3, 3), padding='same')(up10)
		conv10 = Activation('relu')(BatchNormalization()(conv10))
		conv10 = Conv2D(64, (3, 3), padding='same')(conv10)
		conv10 = Activation('relu')(BatchNormalization()(conv10))


		#conv11 = Conv2D(feature_num, (1, 1), activation='sigmoid')(conv10)

		model = Model(inputs=[inputs], outputs=[conv10])

		# Take the weights from first two layers of VGG and lock them
		model.layers[2].set_weights(base_model.layers[1].get_weights())
		model.layers[4].set_weights(base_model.layers[2].get_weights())

		model.layers[2].trainable = False
		model.layers[4].trainable = False

		del base_model
		gc.collect()

	with tf.device('/gpu:1'):
		input2 = model(inputs)
		
		output1 = Conv2D(feature_num, (1, 1), activation='sigmoid')(input2)
		output1 = Lambda(lambda x: K.expand_dims(x,axis=0))(output1)

		#For the second part of the net, we need to reshape batch-wise
		input2 = Lambda(lambda x: K.reshape(x,(batch_size,num_cams,
			K.int_shape(x)[1],K.int_shape(x)[2],K.int_shape(x)[3])))(input2)
		#Permute so that the batch and channel dimensions are together and can be reshaped to concatenate

		input2 = Permute((2,3,4,1))(input2)

		input2 = Reshape((K.int_shape(input2)[1],K.int_shape(input2)[2],-1))(input2)

		bigconv = Conv2D(grid_size,filt_size_3d,strides=filt_size_3d,padding='same')(input2)
		bigconv = Activation('relu')(BatchNormalization()(bigconv))
		bigconv = Conv2D(2*grid_size,filt_size_3d,padding='same',dilation_rate=2)(bigconv)
		bigconv = Activation('relu')(BatchNormalization()(bigconv))
		bigconv = Conv2D(4*grid_size,filt_size_3d,padding='same',dilation_rate=4)(bigconv)
		bigconv = Activation('relu')(BatchNormalization()(bigconv))
		bigconv = Conv2D(8*grid_size,filt_size_3d,padding='same',dilation_rate=6)(bigconv)
		bigconv = Activation('relu')(BatchNormalization()(bigconv))
		bigconv = Conv2D(16*grid_size,filt_size_3d,padding='same',dilation_rate=8)(bigconv)
		bigconv = Activation('relu')(BatchNormalization()(bigconv))
		bigconv = Conv2D(feature_num*grid_size,(1,1),padding='same',activation='sigmoid')(bigconv)

		bigconv = Reshape((grid_size,grid_size,grid_size,feature_num))(bigconv)

		model_bigconv = Model(inputs=[inputs],outputs=[output1, bigconv])

	model_bigconv.compile(optimizer=Adam(lr=lr), loss=lossfunc)

	return model_bigconv

def unet2d_fullbn_vgg19_1024deep(lossfunc, lr, input_dim, feature_num):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convo lutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	base_model = VGG19(weights='imagenet', include_top=False)

	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')((conv1))
	conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
	conv1 = Activation('relu')((conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(BatchNormalization()(conv4))
	conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(BatchNormalization()(conv4))
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
	conv5 = Activation('relu')(BatchNormalization()(conv5))
	conv5 = Conv2D(1024, (3, 3), padding='same')(conv5)
	conv5 = Activation('relu')(BatchNormalization()(conv5))

	up7 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv7 = Conv2D(512, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))
	conv7 = Conv2D(512, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))

	up8 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7), conv3], axis=3)
	conv8 = Conv2D(256, (3, 3), padding='same')(up8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))
	conv8 = Conv2D(256, (3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))

	up9 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8), conv2], axis=3)
	conv9 = Conv2D(128, (3, 3), padding='same')(up9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))
	conv9 = Conv2D(128, (3, 3), padding='same')(conv9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))

	up10 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9), conv1], axis=3)
	conv10 = Conv2D(64, (3, 3), padding='same')(up10)
	conv10 = Activation('relu')(BatchNormalization()(conv10))
	conv10 = Conv2D(64, (3, 3), padding='same')(conv10)
	conv10 = Activation('relu')(BatchNormalization()(conv10))


	conv11 = Conv2D(feature_num, (1, 1), activation='sigmoid')(conv10)

	model = Model(inputs=[inputs], outputs=[conv11])

	# Take the weights from first two layers of VGG and lock them
	model.layers[1].set_weights(base_model.layers[1].get_weights())
	model.layers[3].set_weights(base_model.layers[2].get_weights())

	model.layers[1].trainable = False
	model.layers[3].trainable = False

	del base_model
	gc.collect()

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet2d_fullbn_vgg19(lossfunc, lr, input_dim, feature_num):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	base_model = VGG19(weights='imagenet', include_top=False)

	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')((conv1))
	conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
	conv1 = Activation('relu')((conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(BatchNormalization()(conv4))
	conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(BatchNormalization()(conv4))

	up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
	conv7 = Conv2D(256, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))
	conv7 = Conv2D(256, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))

	up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(128, (3, 3), padding='same')(up8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))
	conv8 = Conv2D(128, (3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))

	up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(64, (3, 3), padding='same')(up9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))
	conv9 = Conv2D(64, (3, 3), padding='same')(conv9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))

	conv10 = Conv2D(feature_num, (1, 1), activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	# Take the weights from first two layers of VGG and lock them
	model.layers[1].set_weights(base_model.layers[1].get_weights())
	model.layers[3].set_weights(base_model.layers[2].get_weights())

	model.layers[1].trainable = False
	model.layers[3].trainable = False

	del base_model
	gc.collect()

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet2d_nobn_vgg19(lossfunc, lr, input_dim, feature_num):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	base_model = VGG19(weights='imagenet', include_top=False)

	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')((conv1))
	conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
	conv1 = Activation('relu')((conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')((conv2))
	conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
	conv2 = Activation('relu')((conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')((conv3))
	conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
	conv3 = Activation('relu')((conv3))
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
	conv4 = Activation('relu')((conv4))
	conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
	conv4 = Activation('relu')((conv4))

	up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
	conv7 = Conv2D(256, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')((conv7))
	conv7 = Conv2D(256, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')((conv7))

	up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(128, (3, 3), padding='same')(up8)
	conv8 = Activation('relu')((conv8))
	conv8 = Conv2D(128, (3, 3), padding='same')(conv8)
	conv8 = Activation('relu')((conv8))

	up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(64, (3, 3), padding='same')(up9)
	conv9 = Activation('relu')((conv9))
	conv9 = Conv2D(64, (3, 3), padding='same')(conv9)
	conv9 = Activation('relu')((conv9))

	conv10 = Conv2D(feature_num, (1, 1), activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	# Take the weights from first two layers of VGG and lock them
	model.layers[1].set_weights(base_model.layers[1].get_weights())
	model.layers[3].set_weights(base_model.layers[2].get_weights())

	model.layers[1].trainable = False
	model.layers[3].trainable = False

	del base_model
	gc.collect()

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet2d_fullbn_linout(lossfunc, lr, input_dim, feature_num):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')(BatchNormalization()(conv1))
	conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(BatchNormalization()(conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(BatchNormalization()(conv4))
	conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(BatchNormalization()(conv4))
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
	conv5 = Activation('relu')(BatchNormalization()(conv5))
	conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
	conv5 = Activation('relu')(BatchNormalization()(conv5))

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), padding='same')(up6)
	conv6 = Activation('relu')(BatchNormalization()(conv6))
	conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(BatchNormalization()(conv6))

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))
	conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), padding='same')(up8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))
	conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), padding='same')(up9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))
	conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))

	conv10 = Conv2D(feature_num, (1, 1), activation='linear')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet2d_dilate_slower(lossfunc, lr, input_dim, feature_num):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')(conv1)
	conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(BatchNormalization()(conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(conv2)
	conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(conv3)
	conv3 = Conv2D(128, (3, 3), dilation_rate = 2, padding='same')(conv3)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(conv4)
	conv4 = Conv2D(256, (3, 3), dilation_rate = 4, padding='same')(conv4)
	conv4 = Activation('relu')(BatchNormalization()(conv4))
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
	conv5 = Activation('relu')(conv5)
	conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
	conv5 = Activation('relu')(BatchNormalization()(conv5))

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), padding='same')(up6)
	conv6 = Activation('relu')(conv6)
	conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(BatchNormalization()(conv6))

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')(conv7)
	conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), padding='same')(up8)
	conv8 = Activation('relu')(conv8)
	conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), padding='same')(up9)
	conv9 = Activation('relu')(conv9)
	conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))

	conv10 = Conv2D(feature_num, (1, 1), activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet2d_dilate(lossfunc, lr, input_dim, feature_num):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')(conv1)
	conv1 = Conv2D(32, (3, 3), dilation_rate = 2, padding='same')(conv1)
	conv1 = Activation('relu')(BatchNormalization()(conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(conv2)
	conv2 = Conv2D(64, (3, 3), dilation_rate = 4, padding='same')(conv2)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(conv3)
	conv3 = Conv2D(128, (3, 3), dilation_rate = 6, padding='same')(conv3)
	conv3 = Activation('relu')(BatchNormalization()(conv3))
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(conv4)
	conv4 = Conv2D(256, (3, 3), dilation_rate = 8, padding='same')(conv4)
	conv4 = Activation('relu')(BatchNormalization()(conv4))
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
	conv5 = Activation('relu')(conv5)
	conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
	conv5 = Activation('relu')(BatchNormalization()(conv5))

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), padding='same')(up6)
	conv6 = Activation('relu')(conv6)
	conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(BatchNormalization()(conv6))

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')(conv7)
	conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), padding='same')(up8)
	conv8 = Activation('relu')(conv8)
	conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(BatchNormalization()(conv8))

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), padding='same')(up9)
	conv9 = Activation('relu')(conv9)
	conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
	conv9 = Activation('relu')(BatchNormalization()(conv9))

	conv10 = Conv2D(feature_num, (1, 1), activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet2d_nobn(lossfunc, lr, input_dim, feature_num):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')(conv1)
	conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
	conv1 = Activation('relu')((conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(conv2)
	conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
	conv2 = Activation('relu')((conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(conv3)
	conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
	conv3 = Activation('relu')((conv3))
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(conv4)
	conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
	conv4 = Activation('relu')((conv4))
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
	conv5 = Activation('relu')(conv5)
	conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
	conv5 = Activation('relu')((conv5))

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), padding='same')(up6)
	conv6 = Activation('relu')(conv6)
	conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
	conv6 = Activation('relu')((conv6))

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')(conv7)
	conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')((conv7))

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), padding='same')(up8)
	conv8 = Activation('relu')(conv8)
	conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
	conv8 = Activation('relu')((conv8))

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), padding='same')(up9)
	conv9 = Activation('relu')(conv9)
	conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
	conv9 = Activation('relu')((conv9))

	conv10 = Conv2D(feature_num, (1, 1), activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet2d_nobn_linout(lossfunc, lr, input_dim, feature_num):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')(conv1)
	conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
	conv1 = Activation('relu')((conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(conv2)
	conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
	conv2 = Activation('relu')((conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(conv3)
	conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
	conv3 = Activation('relu')((conv3))
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
	conv4 = Activation('relu')(conv4)
	conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
	conv4 = Activation('relu')((conv4))
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
	conv5 = Activation('relu')(conv5)
	conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
	conv5 = Activation('relu')((conv5))

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), padding='same')(up6)
	conv6 = Activation('relu')(conv6)
	conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
	conv6 = Activation('relu')((conv6))

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')(conv7)
	conv7 = Conv2D(128, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')((conv7))

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), padding='same')(up8)
	conv8 = Activation('relu')(conv8)
	conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
	conv8 = Activation('relu')((conv8))

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), padding='same')(up9)
	conv9 = Activation('relu')(conv9)
	conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
	conv9 = Activation('relu')((conv9))

	conv10 = Conv2D(feature_num, (1, 1), activation='linear')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet2d_small(lossfunc, lr, input_dim, feature_num):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')(conv1)
	conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(BatchNormalization()(conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(conv2)
	conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(BatchNormalization()(conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(conv3)
	conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(BatchNormalization()(conv3))

	up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
	conv6 = Conv2D(64, (3, 3), padding='same')(up6)
	conv6 = Activation('relu')(conv6)
	conv6 = Conv2D(64, (3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(BatchNormalization()(conv6))

	up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv1], axis=3)
	conv7 = Conv2D(32, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')(conv7)
	conv7 = Conv2D(32, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(BatchNormalization()(conv7))

	conv10 = Conv2D(feature_num, (1, 1), activation='sigmoid')(conv7)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model

def unet2d_small_nobn(lossfunc, lr, input_dim, feature_num):
	"""
	Initialize 2D U-net

	Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
		and tested on variable size input (thus the x-y input dimensions are undefined)
	inputs--
		lossfunc: loss function
		lr: float; learning rate
		input_dim: int; number of feature channels in input
		feature_num: int; number of output features
	outputs--
		model: Keras model object
	"""
	inputs = Input((None, None, input_dim))
	conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
	conv1 = Activation('relu')(conv1)
	conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
	conv1 = Activation('relu')((conv1))
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
	conv2 = Activation('relu')(conv2)
	conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
	conv2 = Activation('relu')((conv2))
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
	conv3 = Activation('relu')(conv3)
	conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
	conv3 = Activation('relu')((conv3))

	up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
	conv6 = Conv2D(64, (3, 3), padding='same')(up6)
	conv6 = Activation('relu')(conv6)
	conv6 = Conv2D(64, (3, 3), padding='same')(conv6)
	conv6 = Activation('relu')((conv6))

	up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv1], axis=3)
	conv7 = Conv2D(32, (3, 3), padding='same')(up7)
	conv7 = Activation('relu')(conv7)
	conv7 = Conv2D(32, (3, 3), padding='same')(conv7)
	conv7 = Activation('relu')((conv7))

	conv10 = Conv2D(feature_num, (1, 1), activation='sigmoid')(conv7)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=['mse'])

	return model


def flynet3d_expectedvalue(lossfunc, lr, input_dim, feature_num, num_cams, gridsize=(64,64,64),
											batch_norm=False, instance_norm = False, include_top=True, regularize_var=False,
											loss_weights = None, metric = ['mse'],out_kernel = (1,1,1)):

	if batch_norm and not instance_norm:
		print('using batch normalization')
		def fun(inputs):
			return BatchNormalization()(inputs)
	elif instance_norm:
		print('using instance normalization')
		def fun(inputs):
			return ops.InstanceNormalization()(inputs)
	else:
		def fun(inputs):
			return inputs

	def slice_input(inp,k):
		print(K.int_shape(inp))
		return inp[:,:,:,:,k*input_dim:(k+1)*input_dim]

	inputs = Input((*gridsize, input_dim*num_cams))
	conv1_layer = Conv3D(64, (3, 3, 3), padding='same')

	conv1 = conv1_layer(inputs)
	conv1 = Activation('relu')(fun(conv1))
	conv1 = Conv3D(64, (3, 3, 3), padding='same')(conv1)
	conv1 = Activation('relu')(fun(conv1))
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

	conv1_layer_fly = Conv3D(32,(3,3,3),padding='same')
	fly1_in = []
	for i in range(num_cams):
		# conv1_in.append(conv1_layer(inputs[:,:,:,:,i*input_dim:(i+1)*input_dim]))
		fly1_in.append(conv1_layer_fly(Lambda(lambda x: slice_input(x,i))(inputs)))

	conv1_fly = concatenate(fly1_in,axis=-1)
	pool1_fly = MaxPooling3D(pool_size=(2, 2, 2))(conv1_fly)


	conv2 = Conv3D(128, (3, 3, 3), padding='same')(concatenate([pool1,pool1_fly],axis=-1))
	conv2 = Activation('relu')(fun(conv2))
	conv2 = Conv3D(128, (3, 3, 3), padding='same')(conv2)
	conv2 = Activation('relu')(fun(conv2))
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv2_layer_fly = Conv3D(64,(3,3,3),padding='same')
	fly2_in = []
	for i in range(num_cams):
		# conv1_in.append(conv1_layer(inputs[:,:,:,:,i*input_dim:(i+1)*input_dim]))
		fly2_in.append(conv2_layer_fly(Lambda(lambda x: slice_input(x,i))(pool1_fly)))

	conv2_fly = concatenate(fly2_in,axis=-1)
	pool2_fly = MaxPooling3D(pool_size=(2, 2, 2))(conv2_fly)

	conv3 = Conv3D(256, (3, 3, 3), padding='same')(concatenate([pool2,pool2_fly],axis=-1))
	conv3 = Activation('relu')(fun(conv3))
	conv3 = Conv3D(256, (3, 3, 3), padding='same')(conv3)
	conv3 = Activation('relu')(fun(conv3))
	pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

	conv3_layer_fly = Conv3D(128,(3,3,3),padding='same')
	fly3_in = []
	for i in range(num_cams):
		# conv1_in.append(conv1_layer(inputs[:,:,:,:,i*input_dim:(i+1)*input_dim]))
		fly3_in.append(conv3_layer_fly(Lambda(lambda x: slice_input(x,i))(pool2_fly)))

	conv3_fly = concatenate(fly3_in,axis=-1)
	pool3_fly = MaxPooling3D(pool_size=(2, 2, 2))(conv3_fly)

	conv4 = Conv3D(512, (3, 3, 3), padding='same')(concatenate([pool3,pool3_fly],axis=-1))
	conv4 = Activation('relu')(fun(conv4))
	conv4 = Conv3D(512, (3, 3, 3), padding='same')(conv4)
	conv4 = Activation('relu')(fun(conv4))


	up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(up6)
	conv6 = Activation('relu')(fun(conv6))
	conv6 = Conv3D(256, (3, 3, 3), padding='same')(conv6)
	conv6 = Activation('relu')(fun(conv6))

	up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv2], axis=4)
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(up7)
	conv7 = Activation('relu')(fun(conv7))
	conv7 = Conv3D(128, (3, 3, 3), padding='same')(conv7)
	conv7 = Activation('relu')(fun(conv7))

	up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv1], axis=4)
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(up8)
	conv8 = Activation('relu')(fun(conv8))
	conv8 = Conv3D(64, (3, 3, 3), padding='same')(conv8)
	conv8 = Activation('relu')(fun(conv8))

	conv10 = Conv3D(feature_num, out_kernel, activation='linear',padding='same')(conv8)

	grid_centers = Input((None,3))

	conv10 = Lambda(lambda x: ops.spatial_softmax(x))(conv10)

	output = Lambda(lambda x: ops.expected_value_3d(x[0],x[1]))([conv10, grid_centers])

	#Because I think it is easier, use a layer to calculate the variance and return it as a second output to be used for variance loss

	output_var = Lambda(lambda x: ops.var_3d(x[0],x[1],x[2]))([conv10, grid_centers,output])

	if include_top:
		if regularize_var:
			model = Model(inputs=[inputs, grid_centers], outputs=[output, output_var])
		else:
			model = Model(inputs=[inputs, grid_centers], outputs=[output])
	else:
		model = Model(inputs=[inputs], outputs=[conv8])


	# model.compile(optimizer=Adam(lr=lr), loss=[lossfunc[0], lossfunc[1]], metrics=['mse'])
	model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=metric,loss_weights=loss_weights)

	return model