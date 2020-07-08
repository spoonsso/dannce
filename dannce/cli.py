"""Entrypoints for dannce training and prediction."""
from dannce.interface import com_predict, com_train, dannce_predict, dannce_train, build_params
from dannce.engine.processing import (
    check_config,
    infer_params
)
from dannce import _param_defaults_dannce, _param_defaults_shared, _param_defaults_com
import sys
import ast
import argparse

def com_predict_cli():
    parser = argparse.ArgumentParser(description="Com predict CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(**{**_param_defaults_shared, **_param_defaults_com})
    args = parse_clargs(parser, model_type="com", prediction=True)
    params = build_clarg_params(args, dannce_net=False)
    com_predict(params)


def com_train_cli():
    parser = argparse.ArgumentParser(description="Com train CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(**{**_param_defaults_shared, **_param_defaults_com})
    args = parse_clargs(parser, model_type="com", prediction=False)
    params = build_clarg_params(args, dannce_net=False)
    com_train(params)


def dannce_predict_cli():
    parser = argparse.ArgumentParser(description="Dannce predict CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(**{**_param_defaults_shared, **_param_defaults_dannce})
    args = parse_clargs(parser, model_type="dannce", prediction=True)
    params = build_clarg_params(args, dannce_net=True)
    dannce_predict(params)


def dannce_train_cli():
    parser = argparse.ArgumentParser(description="Dannce train CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(**{**_param_defaults_shared, **_param_defaults_dannce})
    args = parse_clargs(parser, model_type="dannce", prediction=False)
    params = build_clarg_params(args, dannce_net=True)
    dannce_train(params)

def build_clarg_params(args, dannce_net):
    # Get the params specified in base config and io.yaml
    params = build_params(args.base_config, dannce_net)

    # Combine those params with the clargs
    params = combine(params, args, dannce_net)
    params = infer_params(params, dannce_net)
    check_config(params)
    return params


def add_shared_args(parser):
    # Parse shared args for all conditions
    parser.add_argument(
        "base_config", metavar="base_config", help="Path to base config."
    )
    parser.add_argument("--viddir", dest="viddir", help="Directory containing videos.")
    parser.add_argument(
        "--crop-height",
        dest="crop_height",
        type=ast.literal_eval,
        help="Image crop height.",
    )
    parser.add_argument(
        "--crop-width",
        dest="crop_width",
        type=ast.literal_eval,
        help="Image crop width.",
    )
    parser.add_argument(
        "--camnames",
        dest="camnames",
        type=ast.literal_eval,
        help="List of ordered camera names.",
    )
    parser.add_argument("--io-config", dest="io_config", help="Path to io.yaml file.")
    parser.add_argument(
        "--n-channels-out", dest="n_channels_out", help="Number of keypoints to output. For COM, this is typically 1, but can be equal to the number of points tracked to run in MULTI_MODE."
    )
    parser.add_argument(
        "--batch-size", dest="batch_size", help="Number of images per batch."
    )
    parser.add_argument(
        "--sigma", dest="sigma", help="Standard deviation of confidence maps."
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        help="verbose=0 prints nothing to std out. verbose=1 prints training summary to std out.",
    )
    parser.add_argument("--net", dest="net", help="Network architecture. See nets.py")
    parser.add_argument("--gpuID", dest="gpuID", help="String identifying GPU to use.")
    parser.add_argument("--immode", dest="immode", help="Data format for images.")
    return parser


def add_shared_train_args(parser):
    parser.add_argument(
        "--exp",
        dest="exp",
        type=ast.literal_eval,
        help="List of experiment dictionaries for network training. See examples in io.yaml.",
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        help="Loss function to use during training. See losses.py.",
    )
    parser.add_argument("--epochs", dest="epochs", help="Number of epochs to train.")
    parser.add_argument(
        "--num-validation-per-exp",
        dest="num_validation_per_exp",
        help="Number of validation images to use during training.",
    )
    parser.add_argument(
        "--metric",
        dest="metric",
        type=ast.literal_eval,
        help="List of additional metrics to report. See losses.py",
    )
    parser.add_argument("--lr", dest="lr", help="Learning rate.")
    return parser


def add_shared_predict_args(parser):
    parser.add_argument(
        "--max-num-samples",
        dest="max_num_samples",
        help="Maximum number of samples to predict during COM or DANNCE prediction.",
    )
    return parser


def add_dannce_shared_args(parser):
    parser.add_argument(
        "--com-fromlabels",
        dest="com_fromlabels",
        help="If True, uses the average 3D label position as the 3D COM. Inaccurate for frames with few labeled landmarks.",
    )
    parser.add_argument(
        "--medfilt-window",
        dest="medfilt_window",
        help="Sets the size of an optional median filter used to smooth the COM trace before DANNCE training or prediction.",
    )
    parser.add_argument(
        "--com-file",
        dest="com_file",
        help="Path to com file to use during dannce prediction.",
    )
    parser.add_argument(
        "--new-last-kernel-size",
        dest="new_last_kernel_size",
        type=ast.literal_eval,
        help="List denoting last 3d kernel size. Ex: --new-last-kernel-size=[3,3,3]",
    )
    parser.add_argument(
        "--new-n-channels_out",
        dest="new_n_channels_out",
        help="When finetuning, this refers to the new number of predicted keypoints.",
    )
    parser.add_argument(
        "--n-layers-locked",
        dest="n_layers_locked",
        help="Number of layers from model input to freeze during finetuning.",
    )
    parser.add_argument(
        "--vmin", dest="vmin", help="Minimum range of 3D grid. (Units of distance)"
    )
    parser.add_argument(
        "--vmax", dest="vmax", help="Maximum range of 3D grid. (Units of distance)"
    )
    parser.add_argument(
        "--nvox",
        dest="nvox",
        help="Number of voxels to span each dimension of 3D grid.",
    )
    parser.add_argument(
        "--interp",
        dest="interp",
        help="Voxel interpolation for 3D grid. Linear or nearest.",
    )
    parser.add_argument(
        "--depth",
        dest="depth",
        type=ast.literal_eval,
        help="If True, will append depth information when sampling images. Particularly useful when using just 1 cameras.",
    )
    parser.add_argument(
        "--comthresh",
        dest="comthresh",
        help="COM finder output confidence scores less than this threshold will be discarded.",
    )
    parser.add_argument(
        "--weighted",
        dest="weighted",
        type=ast.literal_eval,
        help="If True, will weight the COM estimate in each camera by its confidence score",
    )
    parser.add_argument(
        "--com-method",
        dest="com_method",
        help="Method of combining 3D COMs across camera pairs. Options: 'median', 'mean'",
    )
    parser.add_argument(
        "--cthresh",
        dest="cthresh",
        help="If the 3D COM has a coordinate beyond this value (in mm), discard it as an error.",
    )
    parser.add_argument(
        "--channel-combo",
        dest="channel_combo",
        help="Dictates whether or not to randomly shuffle the camera order when processing volumes. Options: 'None', 'random'",
    )
    parser.add_argument(
        "--predict-mode",
        dest="predict_mode",
        help="Method for unprojection. Options: numpy, torch, or tensorflow.",
    )
    return parser


def add_dannce_train_args(parser):
    parser.add_argument(
        "--dannce-train-dir",
        dest="dannce_train_dir",
        help="Training directory for dannce network.",
    )
    parser.add_argument(
        "--rotate",
        dest="rotate",
        type=ast.literal_eval,
        help="If True, use rotation augmentation for dannce training.",
    )
    parser.add_argument(
        "--dannce-finetune-weights",
        dest="dannce_finetune_weights",
        help="Path to weights of initial model for dannce fine tuning.",
    )
    parser.add_argument(
        "--train-mode",
        dest="train_mode",
        help="Training modes can be:\n"
        "new: initializes and trains a network from scratch\n"
        "finetune: loads in pre-trained weights and fine-tuned from there\n"
        "continued: initializes a full model, including optimizer state, and continuous training from the last full model checkpoint",
    )
    return parser


def add_dannce_predict_args(parser):
    parser.add_argument(
        "--dannce-predict-dir",
        dest="dannce_predict_dir",
        help="Prediction directory for dannce network.",
    )
    parser.add_argument(
        "--dannce-predict-model",
        dest="dannce_predict_model",
        help="Path to model to use for dannce prediction.",
    )
    parser.add_argument(
        "--start-batch",
        dest="start_batch",
        help="Starting batch number during dannce prediction.",
    )
    parser.add_argument(
        "--predict-model",
        dest="predict_model",
        help="Path to model to use for dannce prediction.",
    )
    parser.add_argument(
        "--expval",
        dest="expval",
        type=ast.literal_eval,
        help="If True, use expected value network. This is normally inferred from the network name. But because prediction can be decoupled from the net param, expval can be set independently if desired.",
    )
    return parser


def add_com_train_args(parser):
    parser.add_argument(
        "--com-train-dir",
        dest="com_train_dir",
        help="Training directory for COM network.",
    )
    parser.add_argument(
        "--com-finetune-weights",
        dest="com_finetune_weights",
        help="Initial weights to use for COM finetuning.",
    )
    return parser


def add_com_predict_args(parser):
    parser.add_argument(
        "--com-predict-dir",
        dest="com_predict_dir",
        help="Prediction directory for COM network.",
    )
    parser.add_argument(
        "--com-predict-weights",
        dest="com_predict_weights",
        help="Path to .hdf5 weights to use for COM prediction.",
    )
    return parser


def add_com_shared_args(parser):
    parser.add_argument(
        "--dsmode",
        dest="dsmode",
        help="Downsampling mode. Can be dsm (local average) or nn (nearest_neighbor).",
    )
    parser.add_argument(
        "--downfac", dest="downfac", help="Downfactoring rate of images."
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        type=ast.literal_eval,
        help="If True, perform debugging operations.",
    )
    return parser


def parse_clargs(parser, model_type, prediction):
    # Handle shared arguments between all models.
    parser = add_shared_args(parser)

    # Handle shared arguments between training and prediction for both models
    if prediction:
        parser = add_shared_predict_args(parser)
    else:
        parser = add_shared_train_args(parser)

    # Handle model specific arguments
    if model_type == "dannce":
        parser = add_dannce_shared_args(parser)
        if prediction:
            parser = add_dannce_predict_args(parser)
        else:
            parser = add_dannce_train_args(parser)
    else:
        parser = add_com_shared_args(parser)
        if prediction:
            parser = add_com_predict_args(parser)
        else:
            parser = add_com_train_args(parser)

    return parser.parse_args()


def combine(base_params, clargs, dannce_net):
    if dannce_net:
        alldefaults = {**_param_defaults_shared, **_param_defaults_dannce}
    else:
        alldefaults = {**_param_defaults_shared, **_param_defaults_com}

    # Logic ---
    # load defaults from parser if they are not already in config
    # use parser argument if different from the default
    for k, v in clargs.__dict__.items():
        if k in alldefaults:
            if v != alldefaults[k] or k not in base_params:
                base_params[k] = v
        elif v is not None:
            base_params[k] = v

    for k, v in base_params.items():
        print("{} set to: {}".format(k, v))
    return base_params
