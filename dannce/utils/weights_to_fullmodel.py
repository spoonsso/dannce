"""
Predicting with predict_DANNCE.py requires a full model, not just the weights. If one
needs to convert a weights file to a full model for this purpose, use this script.

Usage: python weights_to_fullmodel.py path_to_full_model path_to_weights gpuID
"""

#from keras.models import model_from_json
from keras.models import load_model
import dannce.engine.ops as ops
import dannce.engine.nets as nets
import dannce.engine.losses as losses
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] =  sys.argv[3]

mdl = sys.argv[1]
weights = sys.argv[2]
newmdl = weights.split('.hdf5')[0] + '_fullmodel.hdf5'

model = load_model(mdl, custom_objects={'ops': ops,
										'slice_input': nets.slice_input,
										'mask_nan_keep_loss': losses.mask_nan_keep_loss,
										'euclidean_distance_3D': losses.euclidean_distance_3D,
										'centered_euclidean_distance_3D': losses.centered_euclidean_distance_3D})

model.load_weights(weights)
model.save(newmdl)

print("Wrote new model to: " + newmdl)
