"""
If we have trained a multi-gpu TF 1.x model and want to use it for
training or prediction on a single GPU, we need to convert it.

Usage: python multigpu_to_singlegpu.py path_to_model target_GPU
"""

from tensorflow.keras.models import load_model
import sys
import os
import dannce.engine.ops as ops
import dannce.engine.nets as nets
import dannce.engine.losses as losses


if __name__ == "__main__":
    mdl = sys.argv[1]
    newmdl = mdl.split(".hdf5")[0] + "_singleGPU.hdf5"

    # Target an unused GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

    pm = load_model(
        mdl,
        {
            "slice_input": nets.slice_input,
            "euclidean_distance_3D": losses.euclidean_distance_3D,
            "centered_euclidean_distance_3D": losses.centered_euclidean_distance_3D,
            "ops": ops,
        },
    )

    pm.layers[10].save(newmdl)

    print("Converted and wrote new model to: " + newmdl)
