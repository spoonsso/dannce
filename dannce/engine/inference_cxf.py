"""Handle inference procedures for dannce and com networks.
"""
import numpy as np
import os
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import dannce.engine.processing_cxf as processing
from dannce.engine import ops
from typing import List, Dict, Text, Tuple, Union
import torch
import matplotlib
from dannce.engine.processing_cxf import savedata_tomat, savedata_expval
import tqdm
import keras as K
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def print_checkpoint(
    n_frame: int, start_ind: int, end_time: float, sample_save: int = 100
) -> float:
    """Print checkpoint messages indicating frame and fps for inference.
    
    Args:
        n_frame (int): Frame number
        start_ind (int): Start index
        end_time (float): Timing reference
        sample_save (int, optional): Number of samples to use in fps estimation.
    
    No Longer Returned:
        float: New timing reference.
    """
    print("Predicting on sample %d" % (n_frame), flush=True)
    if (n_frame - start_ind) % sample_save == 0 and n_frame != start_ind:
        print(n_frame)
        print("{} samples took {} seconds".format(sample_save, time.time() - end_time))
        end_time = time.time()
    return end_time


def predict_batch(
    model: Model, generator: keras.utils.Sequence, n_frame: int, params: Dict
) -> np.ndarray:
    """Predict for a single batch and reformat output.
    
    Args:
        model (Model): interence model
        generator (keras.utils.Sequence): Data generator
        n_frame (int): Frame number
        params (Dict): Parameters dictionary.
    
    No Longer Returned:
        np.ndarray: n_batch x n_cam x h x w x c predictions
    """
    pred = model.predict(generator.__getitem__(n_frame)[0])
    if params["mirror"]:
        n_cams = 1
    else:
        n_cams = len(params["camnames"])
    shape = [-1, n_cams, pred.shape[1], pred.shape[2], pred.shape[3]]
    pred = np.reshape(pred, shape)
    return pred




def infer_dannce(
    start_ind: int,
    end_ind: int,
    generator: keras.utils.Sequence,
    params: Dict,
    model: Model,
    partition: Dict,
    save_data: Dict,
    device: Text,
    n_chn: int,
):
    """Perform dannce detection over a set of frames.
    
    Args:
        start_ind (int): Starting frame index
        end_ind (int): Ending frame index
        generator (keras.utils.Sequence): Keras data generator
        params (Dict): Parameters dictionary.
        model (Model): Inference model.
        partition (Dict): Partition dictionary
        save_data (Dict): Saved data dictionary
        device (Text): Gpu device name
        n_chn (int): Number of output channels
    """
    for idx, i in enumerate(tqdm.tqdm(range(start_ind, end_ind))):
        if (i - start_ind) % 1000 == 0 and i != start_ind:
            print("Saving checkpoint at {}th batch".format(i))
            if params["expval"]:
                p_n = savedata_expval(
                    params["dannce_predict_dir"] + "save_data_AVG.mat",
                    params,
                    write=True,
                    data=save_data,
                    tcoord=False,
                    num_markers=n_chn,
                    pmax=True,
                )
            else:
                p_n = savedata_tomat(
                    params["dannce_predict_dir"] + "save_data_MAX.mat",
                    params,
                    params["vmin"],
                    params["vmax"],
                    params["nvox"],
                    write=True,
                    data=save_data,
                    num_markers=n_chn,
                    tcoord=False,
                )

        ims = generator.__getitem__(i)
        pred = model.predict(ims[0])

        if params["expval"]:
            probmap = pred[1]
            pred = pred[0]
            for j in range(pred.shape[0]):
                pred_max = probmap[j]
                sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]
                save_data[idx * pred.shape[0] + j] = {
                    "pred_max": pred_max,
                    "pred_coord": pred[j],
                    "sampleID": sampleID,
                }
        else:
            predict_mode = (
                params["predict_mode"]
                if params["predict_mode"] is not None
                else "numpy"
            )
            if predict_mode == "torch":
                for j in range(pred.shape[0]):
                    preds = torch.as_tensor(pred[j], dtype=torch.float32, device=device)
                    pred_max = preds.max(0).values.max(0).values.max(0).values
                    pred_total = preds.sum((0, 1, 2))
                    (
                        xcoord,
                        ycoord,
                        zcoord,
                    ) = processing.plot_markers_3d_torch(preds)
                    coord = torch.stack([xcoord, ycoord, zcoord])
                    pred_log = pred_max.log() - pred_total.log()
                    sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]

                    save_data[idx * pred.shape[0] + j] = {
                        "pred_max": pred_max.cpu().numpy(),
                        "pred_coord": coord.cpu().numpy(),
                        "true_coord_nogrid": ims[1][j],
                        "logmax": pred_log.cpu().numpy(),
                        "sampleID": sampleID,
                    }

            elif predict_mode == "tf":
                # get coords for each map
                with tf.device(device):
                    for j in range(pred.shape[0]):
                        preds = tf.constant(pred[j], dtype="float32")
                        pred_max = tf.math.reduce_max(
                            tf.math.reduce_max(tf.math.reduce_max(preds))
                        )
                        pred_total = tf.math.reduce_sum(
                            tf.math.reduce_sum(tf.math.reduce_sum(preds))
                        )
                        (
                            xcoord,
                            ycoord,
                            zcoord,
                        ) = processing.plot_markers_3d_tf(preds)
                        coord = tf.stack([xcoord, ycoord, zcoord], axis=0)
                        pred_log = tf.math.log(pred_max) - tf.math.log(pred_total)
                        sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]

                        save_data[idx * pred.shape[0] + j] = {
                            "pred_max": pred_max.numpy(),
                            "pred_coord": coord.numpy(),
                            "true_coord_nogrid": ims[1][j],
                            "logmax": pred_log.numpy(),
                            "sampleID": sampleID,
                        }

            else:
                # get coords for each map
                for j in range(pred.shape[0]):
                    pred_max = np.max(pred[j], axis=(0, 1, 2))
                    pred_total = np.sum(pred[j], axis=(0, 1, 2))
                    xcoord, ycoord, zcoord = processing.plot_markers_3d(
                        pred[j, :, :, :, :]
                    )
                    coord = np.stack([xcoord, ycoord, zcoord])
                    pred_log = np.log(pred_max) - np.log(pred_total)
                    sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]

                    save_data[idx * pred.shape[0] + j] = {
                        "pred_max": pred_max,
                        "pred_coord": coord,
                        "true_coord_nogrid": ims[1][j],
                        "logmax": pred_log,
                        "sampleID": sampleID,
                    }
    return save_data


def infer_dannce_max(
    start_ind: int,
    end_ind: int,
    generator: keras.utils.Sequence,
    params: Dict,
    model: Model,
    partition: Dict,
    save_data: Dict,
    device: Text,
    n_chn: int,
):
    """Perform dannce detection over a set of frames.
    
    Args:
        start_ind (int): Starting frame index
        end_ind (int): Ending frame index
        generator (keras.utils.Sequence): Keras data generator
        params (Dict): Parameters dictionary.
        model (Model): Inference model.
        partition (Dict): Partition dictionary
        save_data (Dict): Saved data dictionary
        device (Text): Gpu device name
        n_chn (int): Number of output channels
    """
    iskeras = type(model) in (K.Model, K.engine.functional.Functional)
    with torch.cuda.device(device):
        if not iskeras:
            # model warmup
            
            from torch2trt.torch2trt import torch_dtype_from_trt, torch_device_from_trt
            idx = model.engine.get_binding_index(model.input_names[0])
            dtype = torch_dtype_from_trt(model.engine.get_binding_dtype(idx))
            shape = tuple(model.context.get_binding_shape(idx))
            input = torch.empty(size=shape, dtype=dtype).cuda()
            output = model(input)
            dtype = output.dtype

        for idx, i in enumerate(tqdm.tqdm(range(start_ind, end_ind))):
            assert not params["expval"]
            [X, X_grid], y = generator.__getitem__(i)

            if iskeras:
                pred = model.predict(X)
            else:
                # X_torch = torch.from_numpy(X).cuda().type(dtype)
                # pred = [model(X_torch[[i]]) for i in range(len(X_torch))]
                pred = [model(torch.from_numpy(X[i][None,...]).cuda().type(dtype))
                        for i in range(len(X))]
                pred = torch.cat(pred)

            assert params["predict_mode"] == "torch"
            assert not params["expval"]
            for j in range(pred.shape[0]):
                preds = torch.as_tensor(pred[j], dtype=torch.float32, device=device)
                pred_max = preds.max(0).values.max(0).values.max(0).values
                pred_total = preds.sum((0, 1, 2))
                (xcoord, ycoord, zcoord) = processing.plot_markers_3d_torch(preds)
                coord = X_grid[j][xcoord.cpu().numpy(), ycoord.cpu().numpy(), zcoord.cpu().numpy(), :].T
                com_3d = X_grid[j][[0,-1], [0,-1], [0,-1]].mean(axis=0)
                # coord = torch.stack([xcoord, ycoord, zcoord]).cpu().numpy()
                pred_log = pred_max.log() - pred_total.log()
                sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]
                save_data[idx * pred.shape[0] + j] = {
                    "pred_max": pred_max.cpu().numpy(),
                    "pred_coord": coord.astype(np.float32),
                    "logmax": pred_log.cpu().numpy(),
                    "com_3d": com_3d,
                    "sampleID": sampleID,
                }
    return save_data


def infer_dannce_max_trt(
    start_ind: int,
    end_ind: int,
    generator: keras.utils.Sequence,
    params: Dict,
    model: Model,
    partition: Dict,
    save_data: Dict,
    device: Text,
    n_chn: int,
):
    """Perform dannce detection over a set of frames.
    
    Args:
        start_ind (int): Starting frame index
        end_ind (int): Ending frame index
        generator (keras.utils.Sequence): Keras data generator
        params (Dict): Parameters dictionary.
        model (Model): Inference model.
        partition (Dict): Partition dictionary
        save_data (Dict): Saved data dictionary
        device (Text): Gpu device name
        n_chn (int): Number of output channels
    """
    iskeras = type(model) in (K.Model, K.engine.functional.Functional)
    
    with torch.cuda.device(device):
        assert not iskeras

        # model warmup
        from torch2trt.torch2trt import torch_dtype_from_trt
        idx = model.engine.get_binding_index(model.input_names[0])
        dtype = torch_dtype_from_trt(model.engine.get_binding_dtype(idx))
        shape = tuple(model.context.get_binding_shape(idx))
        input = torch.empty(size=shape, dtype=dtype).cuda()
        output = model(input)
        dtype = output.dtype

        assert params["predict_mode"] == "torch"
        assert not params["expval"]

        X, X_grid = input.cpu().numpy(), np.zeros((*shape[:-1], 2), dtype='float16')

        for idx, i in enumerate(tqdm.tqdm(range(start_ind, end_ind))):
            assert not params["expval"]
            pred_wait = mid_gpu(X, dtype, model)
            X_next, X_grid_next = pre_cpu(generator, i)
            torch.cuda.current_stream().synchronize()
            pred = pred_wait
            post_cpu(pred,X_grid,idx-1,i-1,partition, save_data)
            X, X_grid = X_next, X_grid_next

        pred_wait = mid_gpu(X, dtype, model)
        torch.cuda.current_stream().synchronize()
        pred = pred_wait
        post_cpu(pred,X_grid,idx,i,partition, save_data)
        
    return save_data

def pre_cpu(generator, i):
    [X, X_grid], y = generator[i]
    return X, X_grid

def mid_gpu(X, dtype, model):
    X_torch = torch.from_numpy(X).cuda().type(dtype)
    pred = model(X_torch)
    return pred


def post_cpu(pred,X_grid,idx,i,partition, save_data):
    if idx<0: return
    for j in range(pred.shape[0]):
        preds = pred[j].float()
        pred_max = preds.max(0).values.max(0).values.max(0).values
        pred_total = preds.sum((0, 1, 2))
        (xcoord, ycoord, zcoord) = processing.plot_markers_3d_torch(preds)
        coord = X_grid[j][xcoord.cpu().numpy(), ycoord.cpu().numpy(), zcoord.cpu().numpy(), :].T
        com_3d = X_grid[j][[0,-1], [0,-1], [0,-1]].mean(axis=0)
        pred_log = pred_max.log() - pred_total.log()
        sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]
        save_data[idx * pred.shape[0] + j] = {
            "pred_max": pred_max.cpu().numpy(),
            "pred_coord": coord.astype(np.float32),
            "logmax": pred_log.cpu().numpy(),
            "com_3d": com_3d,
            "sampleID": sampleID,
        }


def infer_dannce_max_trt_copy(
    start_ind: int,
    end_ind: int,
    generator: keras.utils.Sequence,
    params: Dict,
    model: Model,
    partition: Dict,
    save_data: Dict,
    device: Text,
    n_chn: int,
):
    """Perform dannce detection over a set of frames.
    
    Args:
        start_ind (int): Starting frame index
        end_ind (int): Ending frame index
        generator (keras.utils.Sequence): Keras data generator
        params (Dict): Parameters dictionary.
        model (Model): Inference model.
        partition (Dict): Partition dictionary
        save_data (Dict): Saved data dictionary
        device (Text): Gpu device name
        n_chn (int): Number of output channels
    """
    iskeras = type(model) in (K.Model, K.engine.functional.Functional)
    
    with torch.cuda.device(device):
        assert not iskeras

        # model warmup
        from torch2trt.torch2trt import torch_dtype_from_trt, torch_device_from_trt
        idx = model.engine.get_binding_index(model.input_names[0])
        dtype = torch_dtype_from_trt(model.engine.get_binding_dtype(idx))
        shape = tuple(model.context.get_binding_shape(idx))
        input = torch.empty(size=shape, dtype=dtype).cuda()
        output = model(input)
        dtype = output.dtype

        assert params["predict_mode"] == "torch"
        assert not params["expval"]

        for idx, i in enumerate(tqdm.tqdm(range(start_ind, end_ind))):
            assert not params["expval"]
            [X, X_grid], y = generator[i]
            X_torch = torch.from_numpy(X).cuda().type(dtype)
            pred = model(X_torch)

            for j in range(pred.shape[0]):
                preds = pred[j].float()
                pred_max = preds.max(0).values.max(0).values.max(0).values
                pred_total = preds.sum((0, 1, 2))
                (xcoord, ycoord, zcoord) = processing.plot_markers_3d_torch(preds)
                coord = X_grid[j][xcoord.cpu().numpy(), ycoord.cpu().numpy(), zcoord.cpu().numpy(), :].T
                com_3d = X_grid[j][[0,-1], [0,-1], [0,-1]].mean(axis=0)
                pred_log = pred_max.log() - pred_total.log()
                sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]
                save_data[idx * pred.shape[0] + j] = {
                    "pred_max": pred_max.cpu().numpy(),
                    "pred_coord": coord.astype(np.float32),
                    "logmax": pred_log.cpu().numpy(),
                    "com_3d": com_3d,
                    "sampleID": sampleID,
                }
    return save_data
