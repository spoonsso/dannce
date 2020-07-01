"""Entrypoints for dannce training and prediction."""
from dannce.interface import com_predict, com_train, dannce_predict, dannce_train, build_params
from dannce.engine.processing import (
    read_config,
    make_paths_safe,
    inherit_config,
    check_config,
)
import sys
import ast
import argparse


def load_params(param_path):
    with open(param_path, "rb") as f:
        params = yaml.safe_load(f)
    return params


def com_predict_cli():
    parser = argparse.ArgumentParser(description="Com predict CLI")
    args = parse_clargs(parser, model_type="com", prediction=True)
    params = build_clarg_params(args)
    com_predict(params)


def com_train_cli():
    parser = argparse.ArgumentParser(description="Com train CLI")
    args = parse_clargs(parser, model_type="com", prediction=False)
    params = build_clarg_params(args)
    com_train(params)


def dannce_predict_cli():
    parser = argparse.ArgumentParser(description="Dannce predict CLI")
    args = parse_clargs(parser, model_type="dannce", prediction=True)
    params = build_clarg_params(args)
    dannce_predict(params)


def dannce_train_cli():
    parser = argparse.ArgumentParser(description="Dannce train CLI")
    args = parse_clargs(parser, model_type="dannce", prediction=False)
    params = build_clarg_params(args)
    dannce_train(params)


def build_clarg_params(args):
    # Get the params specified in base config and io.yaml
    params = build_params(args.base_config)

    # Combine those params with the clargs
    params = combine(params, args)
    check_config(params)
    return params


def add_shared_args(parser):
    # Parse shared args for all conditions
    parser.add_argument(
        "base_config", metavar="base_config", help="Path to base config."
    )
    parser.add_argument("--viddir", dest="viddir", help="Directory containing videos.")
    parser.add_argument(
        "--CROP-HEIGHT",
        dest="CROP_HEIGHT",
        type=ast.literal_eval,
        help="Image crop height.",
    )
    parser.add_argument(
        "--CROP-WIDTH",
        dest="CROP_WIDTH",
        type=ast.literal_eval,
        help="Image crop width.",
    )
    parser.add_argument(
        "--CAMNAMES",
        dest="CAMNAMES",
        type=ast.literal_eval,
        help="List of ordered camera names.",
    )
    parser.add_argument(
        "--vid-dir-flag",
        dest="vid_dir_flag",
        type=ast.literal_eval,
        help="Set to True if viddir contains nested directories to videos.",
    )
    parser.add_argument("--extension", dest="extension", help="Video extension.")
    parser.add_argument("--chunks", dest="chunks", help="Number of frames per video.")
    parser.add_argument("--io-config", dest="io_config", help="Path to io.yaml file.")
    parser.add_argument(
        "--N-CHANNELS-IN",
        dest="N_CHANNELS_IN",
        help="Number of channels in input image. (RGBD = 4, RGB = 3, grayscale = 1)",
    )
    parser.add_argument(
        "--N-CHANNELS-OUT", dest="N_CHANNELS_OUT", help="Number of keypoints to output."
    )
    parser.add_argument(
        "--BATCH-SIZE", dest="BATCH_SIZE", help="Number of images per batch."
    )
    parser.add_argument(
        "--SIGMA", dest="SIGMA", help="Standard deviation of confidence maps."
    )
    parser.add_argument(
        "--VERBOSE",
        dest="VERBOSE",
        help="VERBOSE=0 prints nothing to std out. VERBOSE=1 prints training summary to std out.",
    )
    parser.add_argument(
        "--DOWNFAC", dest="DOWNFAC", help="Downfactoring rate of images."
    )
    parser.add_argument("--net", dest="net", help="Network architecture. See nets.py")
    parser.add_argument(
        "--debug",
        dest="debug",
        type=ast.literal_eval,
        help="If True, perform debugging operations.",
    )
    parser.add_argument("--gpuID", dest="gpuID", help="String identifying GPU to use.")
    parser.add_argument("--IMMODE", dest="IMMODE", help="Data format for images.")
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
    parser.add_argument("--EPOCHS", dest="EPOCHS", help="Number of epochs to train.")
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
    parser.add_argument(
        "--train-mode",
        dest="train_mode",
        help="Training modes can be:\n"
        "new: initializes and trains a network from scratch\n"
        "finetune: loads in pre-trained weights and fine-tuned from there\n"
        "continued: initializes a full model, including optimizer state, and continuous training from the last full model checkpoint",
    )
    parser.add_argument("--lr", dest="lr", help="Learning rate.")
    return parser


def add_shared_predict_args(parser):
    return parser


def add_dannce_shared_args(parser):
    parser.add_argument(
        "--com-file",
        dest="com_file",
        help="Path to com file to use during dannce training and prediction.",
    )
    parser.add_argument(
        "--NEW-LAST-KERNEL-SIZE",
        dest="NEW_LAST_KERNEL_SIZE",
        type=ast.literal_eval,
        help="List denoting last 3d kernel size. Ex: --NEW-LAST-KERNEL-SIZE=[3,3,3]",
    )
    parser.add_argument(
        "--NEW-N-CHANNELS-OUT",
        dest="NEW_N_CHANNELS_OUT",
        help="When finetuning, this refers to the new number of predicted keypoints.",
    )
    parser.add_argument(
        "--MAX-QUEUE-SIZE",
        dest="MAX_QUEUE_SIZE",
        help="Number of images to keep in queue for the generator.",
    )
    parser.add_argument(
        "--batch-norm",
        dest="batch_norm",
        type=ast.literal_eval,
        help="If True, use batch normalization.",
    )
    parser.add_argument(
        "--instance-norm",
        dest="instance_norm",
        type=ast.literal_eval,
        help="If True, use instance normalization.",
    )
    parser.add_argument(
        "--N-LAYERS-LOCKED",
        dest="N_LAYERS_LOCKED",
        help="Number of layers from model input to freeze during finetuning.",
    )
    parser.add_argument(
        "--VMIN", dest="VMIN", help="Minimum range of 3D grid. (Units of distance)"
    )
    parser.add_argument(
        "--VMAX", dest="VMAX", help="Maximum range of 3D grid. (Units of distance)"
    )
    parser.add_argument(
        "--NVOX",
        dest="NVOX",
        help="Number of voxels to span each dimension of 3D grid.",
    )
    parser.add_argument(
        "--INTERP",
        dest="INTERP",
        help="Voxel interpolation for 3D grid. Linear or nearest.",
    )
    parser.add_argument(
        "--DEPTH",
        dest="DEPTH",
        type=ast.literal_eval,
        help="If True, will append depth information when sampling images. Particularly useful when using just 1 cameras.",
    )
    parser.add_argument(
        "--DISTORT",
        dest="DISTORT",
        type=ast.literal_eval,
        help="If True, apply lens distortion during sampling.",
    )
    parser.add_argument(
        "--EXPVAL",
        dest="EXPVAL",
        type=ast.literal_eval,
        help="If True, use expected value network.",
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
        "--CHANNEL-COMBO",
        dest="CHANNEL_COMBO",
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
        "--ROTATE",
        dest="ROTATE",
        type=ast.literal_eval,
        help="If True, use rotation augmentation for dannce training.",
    )
    parser.add_argument(
        "--dannce-finetune-weights",
        dest="dannce_finetune_weights",
        help="Path to weights of initial model for dannce fine tuning.",
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
        "--maxbatch",
        dest="maxbatch",
        help="Ending batch number during dannce prediction. Set to 'max' to predict from start_batch to the last batch.",
    )
    parser.add_argument(
        "--predict-model",
        dest="predict_model",
        help="Path to model to use for dannce prediction.",
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
    parser.add_argument(
        "--max-num-samples",
        dest="max_num_samples",
        help="Maximum number of samples to predict during COM prediction.",
    )
    return parser


def add_com_shared_args(parser):
    parser.add_argument(
        "--dsmode",
        dest="dsmode",
        help="Downsampling mode. Can be dsm (local average) or nn (nearest_neighbor).",
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


def combine(base_params, clargs):
    for k, v in clargs.__dict__.items():
        if v is not None:
            base_params[k] = v
    return base_params
