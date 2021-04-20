"""
After fine-tuning, all of our model weights, up until the last conv. layer,
are contained within a single layer model object, making it difficult for Keras
to load in these layer weights by name for any subsequent fine-tuning. Here,
the model is loaded, the interior model is extracted,
and its weights are saved.

Usage: python extracted_embedded_model path_to_full_model
"""

# from keras.models import model_from_json
from tensorflow.keras.models import load_model
import dannce.engine.ops as ops
import dannce.engine.nets as nets
import dannce.engine.losses as losses
import sys

if __name__ == "__main__":
    mdl = sys.argv[1]
    newmdl = mdl.split(".hdf5")[0] + "_coremodel.hdf5"

    model = load_model(
        mdl,
        custom_objects={
            "ops": ops,
            "slice_input": nets.slice_input,
            "mask_nan_keep_loss": losses.mask_nan_keep_loss,
            "euclidean_distance_3D": losses.euclidean_distance_3D,
            "centered_euclidean_distance_3D": losses.centered_euclidean_distance_3D,
        },
    )

    model.layers[1].save_weights(newmdl)

    print("Extracted and wrote new model to: " + newmdl)
