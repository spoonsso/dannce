"""Entrypoints for dannce training and prediction."""
from dannce.interface import (
    com_predict,
    com_train,
    dannce_predict,
    dannce_train,
    build_params,
)
from dannce.engine.processing import check_config, infer_params
from dannce import (
    _param_defaults_dannce,
    _param_defaults_shared,
    _param_defaults_com,
)
import sys
import ast
import argparse
from typing import Dict, Text


def com_predict_cli():
    """Entrypoint for com prediction."""
    parser = argparse.ArgumentParser(
        description="Com predict CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(**{**_param_defaults_shared, **_param_defaults_com})
    args = parse_clargs(parser, model_type="com", prediction=True)
    params = build_clarg_params(args, dannce_net=False, prediction=True)
    com_predict(params)


def com_train_cli():
    """Entrypoint for com training."""
    parser = argparse.ArgumentParser(
        description="Com train CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(**{**_param_defaults_shared, **_param_defaults_com})
    args = parse_clargs(parser, model_type="com", prediction=False)
    params = build_clarg_params(args, dannce_net=False, prediction=False)
    com_train(params)


def dannce_predict_cli():
    """Entrypoint for dannce prediction."""
    parser = argparse.ArgumentParser(
        description="Dannce predict CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(**{**_param_defaults_shared, **_param_defaults_dannce})
    args = parse_clargs(parser, model_type="dannce", prediction=True)
    params = build_clarg_params(args, dannce_net=True, prediction=True)
    dannce_predict(params)


def dannce_train_cli():
    """Entrypoint for dannce training."""
    parser = argparse.ArgumentParser(
        description="Dannce train CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(**{**_param_defaults_shared, **_param_defaults_dannce})
    args = parse_clargs(parser, model_type="dannce", prediction=False)
    params = build_clarg_params(args, dannce_net=True, prediction=False)
    dannce_train(params)


def build_clarg_params(
    args: argparse.Namespace, dannce_net: bool, prediction: bool
) -> Dict:
    """Build command line argument parameters

    Args:
        args (argparse.Namespace): Command line arguments parsed by argparse.
        dannce_net (bool): If true, use dannce net defaults.
        prediction (bool): If true, use prediction defaults.

    Returns:
        Dict: Parameters dictionary.
    """
    # Get the params specified in base config and io.yaml
    params = build_params(args.base_config, dannce_net)

    # Combine those params with the clargs
    params = combine(params, args, dannce_net)
    params = infer_params(params, dannce_net, prediction)
    check_config(params, dannce_net, prediction)
    return params


def add_shared_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments shared by all modes.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    # Parse shared args for all conditions
    parser.add_argument(
        "base_config", metavar="base_config", help="Path to base config."
    )
    parser.add_argument(
        "--viddir", dest="viddir", help="Directory containing videos."
    )
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

    parser.add_argument(
        "--io-config", dest="io_config", help="Path to io.yaml file."
    )

    parser.add_argument(
        "--n-channels-out",
        dest="n_channels_out",
        type=int,
        help="Number of keypoints to output. For COM, this is typically 1, but can be equal to the number of points tracked to run in MULTI_MODE.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="Number of images per batch.",
    )
    parser.add_argument(
        "--sigma",
        dest="sigma",
        type=int,
        help="Standard deviation of confidence maps.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        help="verbose=0 prints nothing to std out. verbose=1 prints training summary to std out.",
    )
    parser.add_argument(
        "--net", dest="net", help="Network architecture. See nets.py"
    )
    parser.add_argument(
        "--gpu-id", dest="gpu_id", help="String identifying GPU to use."
    )
    parser.add_argument(
        "--immode", dest="immode", help="Data format for images."
    )

    parser.add_argument(
        "--mono",
        dest="mono",
        type=ast.literal_eval,
        help="If true, converts 3-channel video frames into mono grayscale using standard RGB->gray conversion formula (ref. scikit-image).",
    )

    parser.add_argument(
        "--mirror",
        dest="mirror",
        type=ast.literal_eval,
        help="If true, uses a single video file for multiple views.",
    )

    return parser


def add_shared_train_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments shared by all train modes.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
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
    parser.add_argument(
        "--epochs", dest="epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--num-validation-per-exp",
        dest="num_validation_per_exp",
        type=int,
        help="Number of validation images to use per recording during training.",
    )
    parser.add_argument(
        "--num-train-per-exp",
        dest="num_train_per_exp",
        type=int,
        help="Number of training images to use per recording during training.",
    )
    parser.add_argument(
        "--metric",
        dest="metric",
        type=ast.literal_eval,
        help="List of additional metrics to report. See losses.py",
    )

    parser.add_argument("--lr", dest="lr", help="Learning rate.")

    parser.add_argument(
        "--augment-hue",
        dest="augment_hue",
        type=ast.literal_eval,
        help="If True, randomly augment hue of each image in training set during training.",
    )
    parser.add_argument(
        "--augment-brightness",
        dest="augment_brightness",
        type=ast.literal_eval,
        help="If True, randomly augment brightness of each image in training set during training.",
    )

    parser.add_argument(
        "--augment-hue-val",
        dest="augment_hue_val",
        type=float,
        help="If hue augmentation is True, chooses random hue delta in [-augment_hue_val, augment_hue_val]. Range = [0,1].",
    )
    parser.add_argument(
        "--augment-brightness-val",
        dest="augment_bright_val",
        type=float,
        help="If brightness augmentation is True, chooses random brightness delta in [-augment_hue_val, augment_hue_val]. Range = [0,1].",
    )
    parser.add_argument(
        "--augment-rotation-val",
        dest="augment_rotation_val",
        type=int,
        help="If continuous rotation augmentation is True, chooses random rotation angle in degrees in [-augment_rotation_val, augment_rotation_val]",
    )
    parser.add_argument(
        "--data-split-seed",
        dest="data_split_seed",
        type=int,
        help="Integer seed for the random numebr generator controlling train/test data splits",
    )
    parser.add_argument(
        "--valid-exp",
        dest="valid_exp",
        type=ast.literal_eval,
        help="Pass a list of the expfile indices (0-indexed, starting from the top of your expdict) to be set aside for validation",
    )
    return parser


def add_shared_predict_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments shared by all predict modes.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    parser.add_argument(
        "--max-num-samples",
        dest="max_num_samples",
        type=int,
        help="Maximum number of samples to predict during COM or DANNCE prediction.",
    )
    parser.add_argument(
        "--start-batch",
        dest="start_batch",
        type=int,
        help="Starting batch number during dannce prediction.",
    )
    parser.add_argument(
        "--start-sample",
        dest="start_sample",
        type=int,
        help="Starting sample number during dannce prediction.",
    )
    return parser


def add_dannce_shared_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments shared by all dannce modes.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    parser.add_argument(
        "--net-type",
        dest="net_type",
        help="Net types can be:\n"
        "AVG: more precise spatial average DANNCE, can be harder to train\n"
        "MAX: DANNCE where joint locations are at the maximum of the 3D output distribution\n",
    )
    parser.add_argument(
        "--com-fromlabels",
        dest="com_fromlabels",
        help="If True, uses the average 3D label position as the 3D COM. Inaccurate for frames with few labeled landmarks.",
    )
    parser.add_argument(
        "--medfilt-window",
        dest="medfilt_window",
        type=int,
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
        type=int,
        help="When finetuning, this refers to the new number of predicted keypoints.",
    )
    parser.add_argument(
        "--n-layers-locked",
        dest="n_layers_locked",
        type=int,
        help="Number of layers from model input to freeze during finetuning.",
    )
    parser.add_argument(
        "--vmin",
        dest="vmin",
        type=int,
        help="Minimum range of 3D grid. (Units of distance)",
    )
    parser.add_argument(
        "--vmax",
        dest="vmax",
        type=int,
        help="Maximum range of 3D grid. (Units of distance)",
    )
    parser.add_argument(
        "--nvox",
        dest="nvox",
        type=int,
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
        type=int,
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
    parser.add_argument(
        "--n-views",
        dest="n_views",
        type=int,
        help="Sets the absolute number of views (when using fewer than 6 views only)")
    parser.add_argument(
        "--train-mode",
        dest="train_mode",
        help="Training modes can be:\n"
        "new: initializes and trains a network from scratch\n"
        "finetune: loads in pre-trained weights and fine-tuned from there\n"
        "continued: initializes a full model, including optimizer state, and continuous training from the last full model checkpoint",
    )
    parser.add_argument(
        "--dannce-finetune-weights",
        dest="dannce_finetune_weights",
        help="Path to weights of initial model for dannce fine tuning.",
    )
    return parser


def add_dannce_train_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments specific to dannce training.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
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
        "--augment-continuous-rotation",
        dest="augment_continuous_rotation",
        type=ast.literal_eval,
        help="If True, rotate all images in each sample of the training set by a random value between [-5 and 5] degrees during training.",
    )
    parser.add_argument(
        "--drop-landmark",
        dest="drop_landmark",
        type=ast.literal_eval,
        help="Pass a list of landmark indices to exclude these landmarks from training",
    )
    parser.add_argument(
        "--use-npy",
        dest="use_npy",
        type=ast.literal_eval,
        help="If True, loads training data from npy files"
    )
    parser.add_argument(
        "--rand-view-replace",
        dest="rand_view_replace",
        type=ast.literal_eval,
        help="If True, samples n_rand_views with replacement"
    )
    parser.add_argument(
        "--n-rand-views",
        dest="n_rand_views",
        type=int,
        help="Number of views to sample from the full viewset during training"
    )
    parser.add_argument(
        "--multi-gpu-train",
        dest="multi_gpu_train",
        type=ast.literal_eval,
        help="If True, distribute training data across multiple GPUs for each batch",
    )
    return parser


def add_dannce_predict_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments specific to dannce prediction.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
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
    parser.add_argument(
        "--from-weights",
        dest="from_weights",
        type=ast.literal_eval,
        help="If True, attempt to load in a prediction model without requiring a full model file (i.e. just using weights). May fail for some model types.",
    )
    parser.add_argument(
    "--write-npy",
    dest="write_npy",
    help="If not None, uses this base path to write large dataset to npy files"
    )
 
    return parser


def add_com_train_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments specific to COM training.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
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
    parser.add_argument(
        "--augment-shift",
        dest="augment_shift",
        type=ast.literal_eval,
        help="If True, shift all images in each sample of the training set by a random value during training.",
    )
    parser.add_argument(
        "--augment-zoom",
        dest="augment_zoom",
        type=ast.literal_eval,
        help="If True, zoom all images in each sample of the training set by a random value during training.",
    )
    parser.add_argument(
        "--augment-shear",
        dest="augment_shear",
        type=ast.literal_eval,
        help="If True, shear all images in each sample of the training set by a random value during training.",
    )
    parser.add_argument(
        "--augment-rotation",
        dest="augment_rotation",
        type=ast.literal_eval,
        help="If True, rotate all images in each sample of the training set by a random value during training.",
    )
    parser.add_argument(
        "--augment-shear-val",
        dest="augment_shear_val",
        type=int,
        help="If shear augmentation is True, chooses random shear angle in degrees in [-augment_shear_val, augment_shear_val]",
    )
    parser.add_argument(
        "--augment-zoom-val",
        dest="augment_zoom_val",
        type=float,
        help="If zoom augmentation is True, chooses random zoom factor in [1-augment_zoom_val, 1+augment_zoom_val]",
    )
    parser.add_argument(
        "--augment-shift-val",
        dest="augment_shift_val",
        type=float,
        help="If shift augmentation is True, chooses random offset for rows and columns in [im_size*augment_shift_val, im_size*augment_shift_val]. So augment_shift_val is a fraction of the image size (must be in range [0,1])",
    )
    return parser


def add_com_predict_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments specific to COM prediction.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
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


def add_com_shared_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments shared by all COM modes.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    parser.add_argument(
        "--dsmode",
        dest="dsmode",
        help="Downsampling mode. Can be dsm (local average) or nn (nearest_neighbor).",
    )
    parser.add_argument(
        "--downfac", dest="downfac", type=int, help="Downfactoring rate of images."
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        type=ast.literal_eval,
        help="If True, perform debugging operations.",
    )
    return parser


def parse_clargs(
    parser: argparse.ArgumentParser, model_type: Text, prediction: bool
) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser
        model_type (Text): Type of model. E.g. "dannce"
        prediction (bool): If true, use prediction arg parsers.

    Returns:
        argparse.Namespace: Namespace object with parsed clargs and defaults.
    """
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


def combine(
    base_params: Dict, clargs: argparse.Namespace, dannce_net: bool
) -> Dict:
    """Combine command line, io, and base configurations.

    Args:
        base_params (Dict): Parameters dictionary.
        clargs (argparse.Namespace): Command line argument namespace.
        dannce_net (bool): Description

    Returns:
        Dict: Parameters dictionary.
    """
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
