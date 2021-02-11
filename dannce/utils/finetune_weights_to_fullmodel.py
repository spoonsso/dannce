"""
Predicting with predict_DANNCE.py requires a full model, not just the weights. If one
needs to convert a weights file to a full model for this purpose, use this script.

Usage: python weights_to_fullmodel.py path_to_full_model path_to_weights gpu_id
"""

# from keras.models import model_from_json
from tensorflow.keras.models import load_model, Model
import dannce.engine.ops as ops
import dannce.engine.nets as nets
import dannce.engine.losses as losses
from dannce.cli import parse_clargs, build_clarg_params
from dannce import _param_defaults_dannce, _param_defaults_shared
import argparse
import os


def finetune_weights_to_fullmodel_cli():
    parser = argparse.ArgumentParser(
        description="Dannce train CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(**{**_param_defaults_shared, **_param_defaults_dannce})
    parser.add_argument(
        "weights_path",
        help="Path to weights to convert to model.",
    )
    args = parse_clargs(parser, model_type="dannce", prediction=False)
    params = build_clarg_params(args, dannce_net=True, prediction=False)
    finetune_weights_to_fullmodel(params)


def finetune_weights_to_fullmodel(params):
    weights = os.listdir(params["dannce_finetune_weights"])
    weights = [f for f in weights if ".hdf5" in f]
    weights = weights[0]

    params["dannce_finetune_weights"] = os.path.join(
        params["dannce_finetune_weights"], weights
    )

    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]
    params["depth"] = False
    weights = params["weights_path"]
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    params["loss"] = getattr(losses, params["loss"])
    params["net"] = getattr(nets, params["net"])
    newmdl = weights.split(".hdf5")[0] + "_fullmodel.hdf5"

    gridsize = tuple([params["nvox"]] * 3)
    model = params["net"](
        params["loss"],
        float(params["lr"]),
        params["chan_num"] + params["depth"],
        params["n_channels_out"],
        len(params["camnames"]),
        params["new_last_kernel_size"],
        params["new_n_channels_out"],
        params["dannce_finetune_weights"],
        params["n_layers_locked"],
        batch_norm=False,
        instance_norm=True,
        gridsize=gridsize,
    )

    model.load_weights(params["weights_path"])
    model.save(newmdl)
    print("Wrote new model to: " + newmdl)


if __name__ == "__main__":
    finetune_weights_to_fullmodel_cli()