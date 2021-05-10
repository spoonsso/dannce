"""Handle inference procedures for dannce and com networks.
"""
import numpy as np
import os
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
import dannce.engine.processing as processing
from dannce.engine import ops
from typing import List, Dict, Text, Tuple, Union
import torch
import matplotlib
from dannce.engine.processing import savedata_tomat, savedata_expval

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


def debug_com(
    params: Dict,
    pred: np.ndarray,
    pred_batch: np.ndarray,
    generator: keras.utils.Sequence,
    ind: np.ndarray,
    n_frame: int,
    n_batch: int,
    n_cam: int,
):
    """Print useful figures for COM debugging.
    
    Args:
        params (Dict): Parameters dictionary.
        pred (np.ndarray): Reformatted batch predictions.
        pred_batch (np.ndarray): Batch prediction.
        generator (keras.utils.Sequence): DataGenerator
        ind (np.ndarray): Prediction in image indices
        n_frame (int): Frame number
        n_batch (int): Batch number
        n_cam (int): Camera number
    """
    com_predict_dir = params["com_predict_dir"]
    cmapdir = os.path.join(com_predict_dir, "cmap")
    overlaydir = os.path.join(com_predict_dir, "overlay")
    if not os.path.exists(cmapdir):
        os.makedirs(cmapdir)
    if not os.path.exists(overlaydir):
        os.makedirs(overlaydir)
    print("Writing " + params["com_debug"] + " confidence maps to " + cmapdir)
    print("Writing " + params["com_debug"] + "COM-image overlays to " + overlaydir)

    # Write preds
    plt.figure(0)
    plt.cla()
    plt.imshow(np.squeeze(pred[n_cam]))
    plt.savefig(
        os.path.join(
            cmapdir,
            params["com_debug"] + str(n_frame + n_batch) + ".png",
        )
    )

    plt.figure(1)
    plt.cla()
    im = generator.__getitem__(n_frame * n_batches + n_batch)
    plt.imshow(processing.norm_im(im[0][n_cam]))
    plt.plot(
        (ind[0] - params["crop_width"][0]) / params["downfac"],
        (ind[1] - params["crop_height"][0]) / params["downfac"],
        "or",
    )
    plt.savefig(
        os.path.join(
            overlaydir,
            params["com_debug"] + str(n_frame + n_batch) + ".png",
        )
    )


def extract_multi_instance_single_channel(
    pred: np.ndarray,
    pred_batch: np.ndarray,
    n_cam: int,
    sample_id: Text,
    n_frame: int,
    n_batch: int,
    params: Dict,
    save_data: Dict,
    cameras: Dict,
    generator: keras.utils.Sequence,
) -> Dict:
    """Extract prediction indices for multi-instance single-channel tracking.
    
    Args:
        pred (np.ndarray): Reformatted batch predictions.
        pred_batch (np.ndarray): Batch prediction.
        n_cam (int): Camera number
        sample_id (Text): Sample identifier
        n_frame (int): Frame number
        n_batch (int): Batch number
        params (Dict): Parameters dictionary.
        save_data (Dict): Saved data dictionary.
        cameras (Dict): Camera dictionary
        generator (keras.utils.Sequence): DataGenerator
    
    No Longer Returned:
        (Dict): Updated saved data dictionary.
    """
    pred_max = np.max(np.squeeze(pred[n_cam]))
    ind = (
        np.array(
            processing.get_peak_inds_multi_instance(
                np.squeeze(pred[n_cam]),
                params["n_instances"],
                window_size=3,
            )
        )
        * params["downfac"]
    )
    for instance in range(params["n_instances"]):
        ind[instance, 0] += params["crop_height"][0]
        ind[instance, 1] += params["crop_width"][0]
        ind[instance, :] = ind[instance, ::-1]

    # now, the center of mass is (x,y) instead of (i,j)
    # now, we need to use camera calibration to triangulate
    # from 2D to 3D
    if params["com_debug"] is not None:
        cnum = params["camnames"].index(params["com_debug"])
        if n_cam == cnum:
            debug_com(
                params,
                pred,
                pred_batch,
                generator,
                ind,
                n_frame,
                n_batch,
                n_cam,
            )

    save_data[sample_id][params["camnames"][n_cam]] = {
        "pred_max": pred_max,
        "COM": ind,
    }

    # Undistort this COM here.
    for instance in range(params["n_instances"]):
        pts1 = np.squeeze(
            save_data[sample_id][params["camnames"][n_cam]]["COM"][instance, :]
        )
        pts1 = pts1[np.newaxis, :]
        pts1 = ops.unDistortPoints(
            pts1,
            cameras[params["camnames"][n_cam]]["K"],
            cameras[params["camnames"][n_cam]]["RDistort"],
            cameras[params["camnames"][n_cam]]["TDistort"],
            cameras[params["camnames"][n_cam]]["R"],
            cameras[params["camnames"][n_cam]]["t"],
        )
        save_data[sample_id][params["camnames"][n_cam]]["COM"][
            instance, :
        ] = np.squeeze(pts1)

    return save_data


def extract_multi_instance_multi_channel(
    pred: np.ndarray,
    pred_batch: np.ndarray,
    n_cam: int,
    sample_id: Text,
    n_frame: int,
    n_batch: int,
    params: Dict,
    save_data: Dict,
    cameras: Dict,
    generator: keras.utils.Sequence,
) -> Dict:
    """Extract prediction indices for multi-instance multi-channel tracking.
    
    Args:
        pred (np.ndarray): Reformatted batch predictions.
        pred_batch (np.ndarray): Batch prediction.
        n_cam (int): Camera number
        sample_id (Text): Sample identifier
        n_frame (int): Frame number
        n_batch (int): Batch number
        params (Dict): Parameters dictionary.
        save_data (Dict): Saved data dictionary.
        cameras (Dict): Camera dictionary
        generator (keras.utils.Sequence): DataGenerator
    
    No Longer Returned:
        (Dict): Updated saved data dictionary.
    """
    save_data[sample_id][params["camnames"][n_cam]] = {
        "COM": np.zeros((params["n_instances"], 2)),
    }
    for instance in range(params["n_instances"]):
        pred_max = np.max(np.squeeze(pred[n_cam, :, :, instance]))
        ind = (
            np.array(processing.get_peak_inds(np.squeeze(pred[n_cam, :, :, instance])))
            * params["downfac"]
        )
        ind[0] += params["crop_height"][0]
        ind[1] += params["crop_width"][0]
        ind = ind[::-1]
        # now, the center of mass is (x,y) instead of (i,j)
        # now, we need to use camera calibration to triangulate
        # from 2D to 3D
        if params["com_debug"] is not None:
            cnum = params["camnames"].index(params["com_debug"])
            if n_cam == cnum:
                debug_com(
                    params,
                    pred,
                    pred_batch,
                    generator,
                    ind,
                    n_frame,
                    n_batch,
                    n_cam,
                )

        # Undistort this COM here.
        pts = np.squeeze(ind)
        pts = pts[np.newaxis, :]
        pts = ops.unDistortPoints(
            pts,
            cameras[params["camnames"][n_cam]]["K"],
            cameras[params["camnames"][n_cam]]["RDistort"],
            cameras[params["camnames"][n_cam]]["TDistort"],
            cameras[params["camnames"][n_cam]]["R"],
            cameras[params["camnames"][n_cam]]["t"],
        )
        save_data[sample_id][params["camnames"][n_cam]]["COM"][
            instance, :
        ] = np.squeeze(pts)

        # TODO(pred_max): Currently only saves for one instance.
        save_data[sample_id][params["camnames"][n_cam]]["pred_max"] = pred_max
    return save_data


def extract_single_instance(
    pred: np.ndarray,
    pred_batch: np.ndarray,
    n_cam: int,
    sample_id: Text,
    n_frame: int,
    n_batch: int,
    params: Dict,
    save_data: Dict,
    cameras: Dict,
    generator: keras.utils.Sequence,
):
    """Extract prediction indices for single-instance tracking.
    
    Args:
        pred (np.ndarray): Reformatted batch predictions.
        pred_batch (np.ndarray): Batch prediction.
        n_cam (int): Camera number
        sample_id (Text): Sample identifier
        n_frame (int): Frame number
        n_batch (int): Batch number
        params (Dict): Parameters dictionary.
        save_data (Dict): Saved data dictionary.
        cameras (Dict): Camera dictionary
        generator (keras.utils.Sequence): DataGenerator
    
    No Longer Returned:
        (Dict): Updated saved data dictionary.
    """
    pred_max = np.max(np.squeeze(pred[n_cam]))
    ind = (
        np.array(processing.get_peak_inds(np.squeeze(pred[n_cam]))) * params["downfac"]
    )
    ind[0] += params["crop_height"][0]
    ind[1] += params["crop_width"][0]
    ind = ind[::-1]

    # mirror flip each coord if indicated
    if params["mirror"] and cameras[params["camnames"][n_cam]]["m"] == 1:
        ind[1] = params["raw_im_h"] - ind[1] - 1
        
    # now, the center of mass is (x,y) instead of (i,j)
    # now, we need to use camera calibration to triangulate
    # from 2D to 3D
    if params["com_debug"] is not None:
        cnum = params["camnames"].index(params["com_debug"])
        if n_cam == cnum:
            debug_com(
                params,
                pred,
                pred_batch,
                generator,
                ind,
                n_frame,
                n_batch,
                n_cam,
            )

    save_data[sample_id][params["camnames"][n_cam]] = {
        "pred_max": pred_max,
        "COM": ind,
    }

    # Undistort this COM here.
    pts1 = save_data[sample_id][params["camnames"][n_cam]]["COM"]
    pts1 = pts1[np.newaxis, :]
    pts1 = ops.unDistortPoints(
        pts1,
        cameras[params["camnames"][n_cam]]["K"],
        cameras[params["camnames"][n_cam]]["RDistort"],
        cameras[params["camnames"][n_cam]]["TDistort"],
        cameras[params["camnames"][n_cam]]["R"],
        cameras[params["camnames"][n_cam]]["t"],
    )
    save_data[sample_id][params["camnames"][n_cam]]["COM"] = np.squeeze(pts1)
    return save_data


def triangulate_single_instance(
    n_cams: int, sample_id: Text, params: Dict, camera_mats: Dict, save_data: Dict
) -> Dict:
    """Triangulate for a single instance.
    
    Args:
        n_cams (int): Numver of cameras
        sample_id (Text): Sample identifier.
        params (Dict): Parameters dictionary.
        camera_mats (Dict): Camera matrices dictioanry.
        save_data (Dict): Saved data dictionary.
    
    No Longer Returned:
        Dict: Updated saved data dictionary.
    """
    # Triangulate for all unique pairs
    for n_cam1 in range(n_cams):
        for n_cam2 in range(n_cam1 + 1, n_cams):
            pts1 = save_data[sample_id][params["camnames"][n_cam1]]["COM"]
            pts2 = save_data[sample_id][params["camnames"][n_cam2]]["COM"]
            pts1 = pts1[np.newaxis, :]
            pts2 = pts2[np.newaxis, :]

            test3d = ops.triangulate(
                pts1,
                pts2,
                camera_mats[params["camnames"][n_cam1]],
                camera_mats[params["camnames"][n_cam2]],
            ).squeeze()

            save_data[sample_id]["triangulation"][
                "{}_{}".format(params["camnames"][n_cam1], params["camnames"][n_cam2])
            ] = test3d
    return save_data


def triangulate_multi_instance_multi_channel(
    n_cams: int, sample_id: Text, params: Dict, camera_mats: Dict, save_data: Dict
) -> Dict:
    """Triangulate for multi-instance multi-channel.
    
    Args:
        n_cams (int): Numver of cameras
        sample_id (Text): Sample identifier.
        params (Dict): Parameters dictionary.
        camera_mats (Dict): Camera matrices dictioanry.
        save_data (Dict): Saved data dictionary.
    
    No Longer Returned:
        Dict: Updated saved data dictionary.
    """
    # Triangulate for all unique pairs
    save_data[sample_id]["triangulation"]["instances"] = []
    for instance in range(params["n_instances"]):
        for n_cam1 in range(n_cams):
            for n_cam2 in range(n_cam1 + 1, n_cams):
                pts1 = save_data[sample_id][params["camnames"][n_cam1]]["COM"][
                    instance, :
                ]
                pts2 = save_data[sample_id][params["camnames"][n_cam2]]["COM"][
                    instance, :
                ]
                pts1 = pts1[np.newaxis, :]
                pts2 = pts2[np.newaxis, :]

                test3d = ops.triangulate(
                    pts1,
                    pts2,
                    camera_mats[params["camnames"][n_cam1]],
                    camera_mats[params["camnames"][n_cam2]],
                ).squeeze()

                save_data[sample_id]["triangulation"][
                    "{}_{}".format(
                        params["camnames"][n_cam1], params["camnames"][n_cam2]
                    )
                ] = test3d

        pairs = [
            v for v in save_data[sample_id]["triangulation"].values() if len(v) == 3
        ]
        # import pdb
        # pdb.set_trace()
        pairs = np.stack(pairs, axis=1)
        final = np.nanmedian(pairs, axis=1).squeeze()
        save_data[sample_id]["triangulation"]["instances"].append(final[:, np.newaxis])
    return save_data


def triangulate_multi_instance_single_channel(
    n_cams: int,
    sample_id: Text,
    params: Dict,
    camera_mats: Dict,
    cameras: Dict,
    save_data: Dict,
) -> Dict:
    """Triangulate for multi-instance single-channel.
    
    Args:
        n_cams (int): Numver of cameras
        sample_id (Text): Sample identifier.
        params (Dict): Parameters dictionary.
        camera_mats (Dict): Camera matrices dictioanry.
        cameras (Dict): Camera dictionary.
        save_data (Dict): Saved data dictionary.
    
    No Longer Returned:
        Dict: Updated saved data dictionary.
    """
    # Go through the instances, adding the most parsimonious
    # points of the n_instances available points at each camera.
    cams = [camera_mats[params["camnames"][n_cam]] for n_cam in range(n_cams)]
    best_pts = []
    best_pts_inds = []
    for instance in range(params["n_instances"]):
        pts = []
        pts_inds = []

        # Build the initial list of points
        for n_cam in range(n_cams):
            pt = save_data[sample_id][params["camnames"][n_cam]]["COM"][instance, :]
            pt = pt[np.newaxis, :]
            pts.append(pt)
            pts_inds.append(instance)

        # Go through each camera (other than the first) and test
        # each instance
        for n_cam in range(1, n_cams):
            candidate_errors = []
            for n_point in range(params["n_instances"]):
                if len(best_pts_inds) >= 1:
                    if any(n_point == p[n_cam] for p in best_pts_inds):
                        candidate_errors.append(np.Inf)
                        continue

                pt = save_data[sample_id][params["camnames"][n_cam]]["COM"][n_point, :]
                pt = pt[np.newaxis, :]
                pts[n_cam] = pt
                pts_inds[n_cam] = n_point
                pts3d = ops.triangulate_multi_instance(pts, cams)

                # Loop through each camera, reproject the point
                # into image coordinates, and save the error.
                error = 0
                for n_proj in range(n_cams):
                    K = cameras[params["camnames"][n_proj]]["K"]
                    R = cameras[params["camnames"][n_proj]]["R"]
                    t = cameras[params["camnames"][n_proj]]["t"]
                    proj = ops.project_to2d(pts3d.T, K, R, t)
                    proj = proj[:, :2]
                    ref = save_data[sample_id][params["camnames"][n_proj]]["COM"][
                        pts_inds[n_proj], :
                    ]
                    error += np.sqrt(np.sum((proj - ref) ** 2))
                candidate_errors.append(error)

            # Keep the best instance combinations across cameras
            best_candidate = np.argmin(candidate_errors)
            pt = save_data[sample_id][params["camnames"][n_cam]]["COM"][
                best_candidate, :
            ]
            pt = pt[np.newaxis, :]
            pts[n_cam] = pt
            pts_inds[n_cam] = best_candidate

        best_pts.append(pts)
        best_pts_inds.append(pts_inds)

    # Do one final triangulation
    final3d = [
        ops.triangulate_multi_instance(best_pts[k], cams)
        for k in range(params["n_instances"])
    ]
    save_data[sample_id]["triangulation"]["instances"] = final3d
    return save_data


def infer_com(
    start_ind: int,
    end_ind: int,
    generator: keras.utils.Sequence,
    params: Dict,
    model: Model,
    partition: Dict,
    save_data: Dict,
    camera_mats: Dict,
    cameras: Dict,
    sample_save: int = 100,
):
    """Perform COM detection over a set of frames.
    
    Args:
        start_ind (int): Starting frame index
        end_ind (int): Ending frame index
        generator (keras.utils.Sequence): Keras data generator
        params (Dict): Parameters dictionary.
        model (Model): Inference model.
        partition (Dict): Partition dictionary
        save_data (Dict): Saved data dictionary
        camera_mats (Dict): Camera matrix dictionary
        cameras (Dict): Camera dictionary.
        sample_save (int, optional): Number of samples to use in fps estimation.
    """
    end_time = time.time()
    for n_frame in range(start_ind, end_ind):
        end_time = print_checkpoint(
            n_frame, start_ind, end_time, sample_save=sample_save
        )
        pred_batch = predict_batch(model, generator, n_frame, params)
        n_batches = pred_batch.shape[0]

        for n_batch in range(n_batches):
            # By selecting -1 for the last axis, we get the COM index for a
            # normal COM network, and also the COM index for a multi_mode COM network,
            # as in multimode the COM label is put at the end
            if params["mirror"] and params["n_instances"] == 1:
                # For mirror we need to reshape pred so that the cameras are in front, so
                # it works with the downstream code
                pred = pred_batch[n_batch, 0]
                pred = np.transpose(pred, (2, 0, 1))
            elif params["mirror"]:
                raise Exception("mirror mode with multiple animal instances not currently supported.")
            elif params["n_instances"] > 1 and params["n_channels_out"] > 1:
                pred = pred_batch[n_batch, ...]
            else:
                pred = pred_batch[n_batch, :, :, :, -1]
            sample_id = partition["valid_sampleIDs"][n_frame * n_batches + n_batch]
            save_data[sample_id] = {}
            save_data[sample_id]["triangulation"] = {}
            n_cams = pred.shape[0]

            for n_cam in range(n_cams):
                args = [
                    pred,
                    pred_batch,
                    n_cam,
                    sample_id,
                    n_frame,
                    n_batch,
                    params,
                    save_data,
                    cameras,
                    generator,
                ]
                if params["n_instances"] == 1:
                    save_data = extract_single_instance(*args)
                elif params["n_channels_out"] == 1:
                    save_data = extract_multi_instance_single_channel(*args)
                elif params["n_channels_out"] > 1:
                    save_data = extract_multi_instance_multi_channel(*args)

            # Handle triangulation for single or multi instance
            if params["n_instances"] == 1:
                save_data = triangulate_single_instance(
                    n_cams, sample_id, params, camera_mats, save_data
                )
            elif params["n_channels_out"] == 1:
                save_data = triangulate_multi_instance_single_channel(
                    n_cams, sample_id, params, camera_mats, cameras, save_data
                )
            elif params["n_channels_out"] > 1:
                save_data = triangulate_multi_instance_multi_channel(
                    n_cams, sample_id, params, camera_mats, save_data
                )
    return save_data


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

    end_time = time.time()
    for idx, i in enumerate(range(start_ind, end_ind)):
        print("Predicting on batch {}".format(i), flush=True)
        if (i - start_ind) % 10 == 0 and i != start_ind:
            print(i)
            print("10 batches took {} seconds".format(time.time() - end_time))
            end_time = time.time()

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