"""Operations for dannce."""
import tensorflow as tf
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import tensorflow.keras.backend as K
import tensorflow.keras.initializers as initializers
import tensorflow.keras.constraints as constraints
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.utils import get_custom_objects
import cv2
import time
from typing import Text, List, Dict, Tuple, Union
import torch


def camera_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Derive the camera matrix.

    Derive the camera matrix from the camera intrinsic matrix (K),
    and the extrinsic rotation matric (R), and extrinsic
    translation vector (t).

    Note that this uses the matlab convention, such that
    M = [R;t] * K
    """
    return np.concatenate((R, t), axis=0) @ K


def project_to2d(
    pts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Project 3d points to 2d.

    Projects a set of 3-D points, pts, into 2-D using the camera intrinsic
    matrix (K), and the extrinsic rotation matric (R), and extrinsic
    translation vector (t). Note that this uses the matlab
    convention, such that
    M = [R;t] * K, and pts2d = pts3d * M
    """

    M = np.concatenate((R, t), axis=0) @ K
    projPts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1) @ M
    projPts[:, :2] = projPts[:, :2] / projPts[:, 2:]

    return projPts


def project_to2d_torch(pts, M: np.ndarray, device: Text) -> torch.Tensor:
    """Project 3d points to 2d.

    Projects a set of 3-D points, pts, into 2-D using the camera intrinsic
    matrix (K), and the extrinsic rotation matric (R), and extrinsic
    translation vector (t). Note that this uses the matlab
    convention, such that
    M = [R;t] * K, and pts2d = pts3d * M
    """
    import torch

    # pts = torch.Tensor(pts.copy()).to(device)
    M = M.to(device=device)
    pts1 = torch.ones(pts.shape[0], 1, dtype=torch.float32, device=device)

    projPts = torch.matmul(torch.cat((pts, pts1), 1), M)
    projPts[:, :2] = projPts[:, :2] / projPts[:, 2:]

    return projPts


def project_to2d_tf(projPts, M):
    """Project 3d points to 2d.

    Projects a set of 3-D points, pts, into 2-D using the camera intrinsic
    matrix (K), and the extrinsic rotation matric (R), and extrinsic
    translation vector (t). Note that this uses the matlab
    convention, such that
    M = [R;t] * K, and pts2d = pts3d * M
    """

    projPts = tf.matmul(projPts, M)
    projPts = projPts[:, :2] / projPts[:, 2:]

    return projPts


def sample_grid(im: np.ndarray, projPts: np.ndarray, method: Text = "linear"):
    """Transfer 3d features to 2d by projecting down to 2d grid.

    Use 2d interpolation to transfer features to 3d points that have
    projected down onto a 2d grid
    Note that function expects proj_grid to be flattened, so results should be
    reshaped after being returned
    """
    if method == "linear":
        f_r = RegularGridInterpolator(
            (np.arange(im.shape[0]), np.arange(im.shape[1])),
            im[:, :, 0],
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        f_g = RegularGridInterpolator(
            (np.arange(im.shape[0]), np.arange(im.shape[1])),
            im[:, :, 1],
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        f_b = RegularGridInterpolator(
            (np.arange(im.shape[0]), np.arange(im.shape[1])),
            im[:, :, 2],
            method="linear",
            bounds_error=False,
            fill_value=0,
        )

        proj_r = f_r(projPts[:, ::-1])
        proj_g = f_g(projPts[:, ::-1])
        proj_b = f_b(projPts[:, ::-1])

    # Nearest neighbor rounding technique
    # Remember that projPts[:,0] is the "x" coordinate, i.e. the
    # column dimension, and projPts[:,1] is "y", indexing in the row
    # dimension, matrix-wise (i.e. from the top of the image)
    elif method == "nearest":
        # Now I could index an array with the values
        projPts = np.round(projPts[:, ::-1]).astype("int")

        # But some of them could be rounded outside of the image
        projPts[projPts[:, 0] < 0, 0] = 0
        projPts[projPts[:, 0] >= im.shape[0], 0] = im.shape[0] - 1
        projPts[projPts[:, 1] < 0, 1] = 0
        projPts[projPts[:, 1] >= im.shape[1], 1] = im.shape[1] - 1

        projPts = (projPts[:, 0], projPts[:, 1])

        proj_r = im[:, :, 0]
        proj_r = proj_r[projPts]
        proj_g = im[:, :, 1]
        proj_g = proj_g[projPts]
        proj_b = im[:, :, 2]
        proj_b = proj_b[projPts]

    # Do nearest, but because the channel dimension can be arbitrarily large,
    # we put the final part of this in a loop
    elif method == "out2d":
        # Now I could index an array with the values
        projPts = np.round(projPts[:, ::-1]).astype("int")

        # But some of them could be outside of the image
        projPts[projPts[:, 0] < 0, 0] = 0
        projPts[projPts[:, 0] >= im.shape[0], 0] = im.shape[0] - 1
        projPts[projPts[:, 1] < 0, 1] = 0
        projPts[projPts[:, 1] >= im.shape[1], 1] = im.shape[1] - 1

        imout = np.zeros((projPts.shape[0], im.shape[-1]))
        projPts = (projPts[:, 0], projPts[:, 1])

        for ii in range(im.shape[-1]):
            tmp = im[:, :, ii]
            imout[:, ii] = tmp[projPts]
        return imout
    else:
        raise Exception("not a valid interpolation method")
    return proj_r, proj_g, proj_b


def sample_grid_torch_nearest(
    im: np.ndarray, projPts: np.ndarray, device: Text
) -> torch.Tensor:
    """Unproject features."""
    # im_x, im_y are the x and y coordinates of each projected 3D position.
    # These are concatenated here for every image in each batch,
    import torch

    feats = torch.as_tensor(im.copy(), device=device)
    grid = projPts
    c = int(round(projPts.shape[0] ** (1 / 3.0)))

    fh, fw, fdim = list(feats.shape)

    # # make sure all projected indices fit onto the feature map
    im_x = torch.clamp(grid[:, 0], 0, fw - 1)
    im_y = torch.clamp(grid[:, 1], 0, fh - 1)

    im_xr = im_x.round().type(torch.long)
    im_yr = im_y.round().type(torch.long)
    im_xr[im_xr < 0] = 0
    im_yr[im_yr < 0] = 0
    Ir = feats[im_yr, im_xr]

    return Ir.reshape((c, c, c, -1)).permute(3, 0, 1, 2).unsqueeze(0)


def sample_grid_torch_linear(
    im: np.ndarray, projPts: np.ndarray, device: Text
) -> torch.Tensor:
    """Unproject features."""
    # im_x, im_y are the x and y coordinates of each projected 3D position.
    # These are concatenated here for every image in each batch,
    import torch

    feats = torch.as_tensor(im.copy(), device=device)
    grid = projPts
    c = int(round(projPts.shape[0] ** (1 / 3.0)))

    fh, fw, fdim = list(feats.shape)

    # # make sure all projected indices fit onto the feature map
    im_x = torch.clamp(grid[:, 0], 0, fw - 1)
    im_y = torch.clamp(grid[:, 1], 0, fh - 1)

    # round all indices
    im_x0 = torch.floor(im_x).type(torch.long)
    # new array with rounded projected indices + 1
    im_x1 = im_x0 + 1
    im_y0 = torch.floor(im_y).type(torch.long)
    im_y1 = im_y0 + 1

    # Convert from int to float -- but these are still round
    # numbers because of rounding step above
    im_x0_f, im_x1_f = im_x0.type(torch.float), im_x1.type(torch.float)
    im_y0_f, im_y1_f = im_y0.type(torch.float), im_y1.type(torch.float)

    # Gather  values
    # Samples all featuremaps at the projected indices,
    # and their +1 counterparts. Stop at Ia for nearest neighbor interpolation.

    # need to clip the corner indices because they might be out of bounds...
    # This could lead to different behavior compared to TF/numpy, which return 0
    # when an index is out of bounds
    im_x1_safe = torch.clamp(im_x1, 0, fw - 1)
    im_y1_safe = torch.clamp(im_y1, 0, fh - 1)

    im_x1[im_x1 < 0] = 0
    im_y1[im_y1 < 0] = 0
    im_x0[im_x0 < 0] = 0
    im_y0[im_y0 < 0] = 0
    im_x1_safe[im_x1_safe < 0] = 0
    im_y1_safe[im_y1_safe < 0] = 0

    Ia = feats[im_y0, im_x0]
    Ib = feats[im_y0, im_x1_safe]
    Ic = feats[im_y1_safe, im_x0]
    Id = feats[im_y1_safe, im_x1_safe]

    # To recaptiulate behavior  in numpy/TF, zero out values that fall outside bounds
    Ib[im_x1 > fw - 1] = 0
    Ic[im_y1 > fh - 1] = 0
    Id[(im_x1 > fw - 1) | (im_y1 > fh - 1)] = 0
    # Calculate bilinear weights
    # We've now sampled the feature maps at corners around the projected values
    # Here, the corners are weighted by distance from the projected value
    wa = (im_x1_f - im_x) * (im_y1_f - im_y)
    wb = (im_x1_f - im_x) * (im_y - im_y0_f)
    wc = (im_x - im_x0_f) * (im_y1_f - im_y)
    wd = (im_x - im_x0_f) * (im_y - im_y0_f)

    Ibilin = (
        wa.unsqueeze(1) * Ia
        + wb.unsqueeze(1) * Ib
        + wc.unsqueeze(1) * Ic
        + wd.unsqueeze(1) * Id
    )

    return Ibilin.reshape((c, c, c, -1)).permute(3, 0, 1, 2).unsqueeze(0)


def sample_grid_torch(im: np.ndarray, projPts: np.ndarray, device: Text, method: Text = "linear"):
    """Transfer 3d features to 2d by projecting down to 2d grid, using torch.

    Use 2d interpolation to transfer features to 3d points that have
    projected down onto a 2d grid
    Note that function expects proj_grid to be flattened, so results should be
    reshaped after being returned
    """
    if method == "nearest" or method == "out2d":
        proj_rgb = sample_grid_torch_nearest(im, projPts, device)
    elif method == "linear" or method == "bilinear":
        proj_rgb = sample_grid_torch_linear(im, projPts, device)
    else:
        raise Exception("{} not a valid interpolation method".format(method))

    return proj_rgb


def sample_grid_tf(im, projPts, device, method="linear"):
    """Transfer 3d features to 2d by projecting down to 2d grid.

    Use 2d interpolation to transfer features to 3d points that have
    projected down onto a 2d grid
    Note that function expects proj_grid to be flattened, so results should be
    reshaped after being returned
    """
    with tf.device(device):
        im = tf.constant(im)
        im = tf.expand_dims(im, 0)
        if method == "nearest":
            projPts = tf.expand_dims(projPts, 0)
            proj_rgb = unproj_tf_nearest(im, projPts, 1)
        elif method == "linear":
            im = tf.cast(im, "float32")
            projPts = tf.expand_dims(projPts, 0)
            projPts = tf.reverse(projPts, [1])
            proj_rgb = tf.cast(unproj_tf_linear(im, projPts, 1), "uint8")
            proj_rgb = tf.reshape(proj_rgb, (tf.shape(projPts)[1], 3))
            proj_rgb = tf.reverse(proj_rgb, [0])
        else:
            raise Exception("not a valid interpolation method")

        return proj_rgb


def sample_grid_tf(im, projPts, device, method="linear"):
    """Transfer 3d features to 2d by projecting down to 2d grid.

    Use 2d interpolation to transfer features to 3d points that have
    projected down onto a 2d grid
    Note that function expects proj_grid to be flattened, so results should be
    reshaped after being returned
    """
    with tf.device(device):
        im = tf.constant(im)
        im = tf.expand_dims(im, 0)
        if method == "nearest":
            projPts = tf.expand_dims(projPts, 0)
            proj_rgb = unproj_tf_nearest(im, projPts, 1)
        elif method == "linear":
            im = tf.cast(im, "float32")
            projPts = tf.expand_dims(projPts, 0)
            projPts = tf.reverse(projPts, [1])
            proj_rgb = tf.cast(unproj_tf_linear(im, projPts, 1), "uint8")
            proj_rgb = tf.reshape(proj_rgb, (tf.shape(projPts)[1], 3))
            proj_rgb = tf.reverse(proj_rgb, [0])
        else:
            raise Exception("not a valid interpolation method")

        return proj_rgb


@tf.function
def unproj_tf_nearest(feats, grid, batch_size):
    """Unproject features.
    Modified from https://github.com/akar43/lsm
    """
    # im_x, im_y are the x and y coordinates of each projected 3D position.
    # These are concatenated here for every image in each batch,

    nR, fh, fw, fdim = K.int_shape(feats)
    nR2, nV, nD = K.int_shape(grid)

    # # make sure all projected indices fit onto the feature map
    im_x = tf.clip_by_value(grid[:, :, 0], 0, fw - 1)
    im_y = tf.clip_by_value(grid[:, :, 1], 0, fh - 1)

    # nR should be batch_size*num_cams
    # eg. [0,1,2,3,4,5] for 3 cams, batch_size=2
    ind_grid = tf.range(0, nR)
    ind_grid = tf.expand_dims(ind_grid, 1)

    # nV is the number of voxels, so this tiling operation
    # produces e.g. [0,0,0,0,0,0; 1,1,1,1,1,1]
    im_ind = tf.tile(ind_grid, [1, nV])

    @tf.function
    def _get_gather_inds(x, y):
        return tf.reshape(tf.stack([im_ind, y, x], axis=2), [-1, 3])

    im_xr = tf.cast(tf.round(im_x), "int32")
    im_yr = tf.cast(tf.round(im_y), "int32")
    Ir = tf.gather_nd(feats, _get_gather_inds(im_xr, im_yr))
    return Ir


@tf.function
def unproj_tf_linear(feats, grid, batch_size):
    """Unproject features.
        Modified from https://github.com/akar43/lsm
    """
    # im_x, im_y are the x and y coordinates of each projected 3D position.
    # These are concatenated here for every image in each batch,

    nR, fh, fw, fdim = K.int_shape(feats)
    nR2, nV, nD = K.int_shape(grid)

    # make sure all projected indices fit onto the feature map
    im_x = tf.clip_by_value(grid[:, :, 0], 0, fw - 1)
    im_y = tf.clip_by_value(grid[:, :, 1], 0, fh - 1)

    # round all indices
    im_x0 = tf.cast(tf.floor(im_x), "int32")
    # new array with rounded projected indices + 1
    im_x1 = im_x0 + 1
    im_y0 = tf.cast(tf.floor(im_y), "int32")
    im_y1 = im_y0 + 1

    # Convert from int to float -- but these are still round
    # numbers because of rounding step above
    im_x0_f, im_x1_f = tf.cast(im_x0, "float32"), tf.cast(im_x1, "float32")
    im_y0_f, im_y1_f = tf.cast(im_y0, "float32"), tf.cast(im_y1, "float32")

    # nR should be batch_size*num_cams
    # eg. [0,1,2,3,4,5] for 3 cams, batch_size=2
    ind_grid = tf.range(0, nR)
    ind_grid = tf.expand_dims(ind_grid, 1)

    # nV is the number of voxels, so this tiling operation
    # produces e.g. [0,0,0,0,0,0; 1,1,1,1,1,1]
    im_ind = tf.tile(ind_grid, [1, nV])

    @tf.function
    def _get_gather_inds(x, y):
        return tf.reshape(tf.stack([im_ind, y, x], axis=2), [-1, 3])

    # Gather  values
    # Samples all featuremaps per batch/camera at the projected indices,
    # and their +1 counterparts. Stop at Ia for nearest neighbor interpolation.
    # I* should be a tensor of shape:
    # (num_cams*batch_size*len(im_x0)*len(im_y0), fdim)

    Ia = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y0))
    Ib = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y1))
    Ic = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y0))
    Id = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y1))

    # Calculate bilinear weights
    # We've now sampled the feature maps at corners around the projected values
    # Here, the corners are weights by distance from the projected value
    wa = (im_x1_f - im_x) * (im_y1_f - im_y)
    wb = (im_x1_f - im_x) * (im_y - im_y0_f)
    wc = (im_x - im_x0_f) * (im_y1_f - im_y)
    wd = (im_x - im_x0_f) * (im_y - im_y0_f)

    # TODO(reshape): Why is this reshape necessary?
    wa, wb = tf.reshape(wa, [-1, 1]), tf.reshape(wb, [-1, 1])
    wc, wd = tf.reshape(wc, [-1, 1]), tf.reshape(wd, [-1, 1])
    Ibilin = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    Ibilin = tf.reshape(
        Ibilin,
        [
            batch_size,
            nR // batch_size,
            int((nV + 1) ** (1 / 3)),
            int((nV + 1) ** (1 / 3)),
            int((nV + 1) ** (1 / 3)),
            fdim,
        ],
    )
    Ibilin = tf.transpose(Ibilin, [0, 1, 3, 2, 4, 5])
    return Ibilin


def unproj(feats, grid, batch_size):
    """Unproject features.
    
    Modified from https://github.com/akar43/lsm
    """
    # im_x, im_y are the x and y coordinates of each projected 3D position.
    # These are concatenated here for every image in each batch,
    nR, fh, fw, fdim = K.int_shape(feats)
    nR2, nV, nD = K.int_shape(grid)

    # make sure all projected indices fit onto the feature map
    im_x = tf.clip_by_value(grid[:, :, 0], 0, fw - 1)
    im_y = tf.clip_by_value(grid[:, :, 1], 0, fh - 1)

    # round all indices down?
    im_x0 = tf.cast(tf.floor(im_x), "int32")
    # new array with rounded projected indices + 1
    im_x1 = im_x0 + 1
    im_y0 = tf.cast(tf.floor(im_y), "int32")
    im_y1 = im_y0 + 1

    # Convert from int to float -- but these are still round
    # numbers because of rounding step above
    im_x0_f, im_x1_f = tf.to_float(im_x0), tf.to_float(im_x1)
    im_y0_f, im_y1_f = tf.to_float(im_y0), tf.to_float(im_y1)

    # nR should be batch_size*num_cams
    # eg. [0,1,2,3,4,5] for 3 cams, batch_size=2
    ind_grid = tf.range(0, nR)
    ind_grid = tf.expand_dims(ind_grid, 1)
    # nV is the number of voxels, so this tiling operation
    # produces e.g. [0,0,0,0,0,0; 1,1,1,1,1,1]
    im_ind = tf.tile(ind_grid, [1, nV])

    def _get_gather_inds(x, y):
        return tf.reshape(tf.stack([im_ind, y, x], axis=2), [-1, 3])

    # Gather  values
    # Samples all featuremaps per batch/camera at the projected indices,
    # and their +1 counterparts. Stop at Ia for nearest neighbor interpolation.
    # I* should be a tensor of shape:
    # (num_cams*batch_size*len(im_x0)*len(im_y0), fdim)
    Ia = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y0))
    Ib = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y1))
    Ic = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y0))
    Id = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y1))

    # Calculate bilinear weights
    # We've now sampled the feature maps at corners around the projected values
    # Here, the corners are weights by distance from the projected value
    wa = (im_x1_f - im_x) * (im_y1_f - im_y)
    wb = (im_x1_f - im_x) * (im_y - im_y0_f)
    wc = (im_x - im_x0_f) * (im_y1_f - im_y)
    wd = (im_x - im_x0_f) * (im_y - im_y0_f)

    # TODO(reshape): Why is this reshape necessary?
    wa, wb = tf.reshape(wa, [-1, 1]), tf.reshape(wb, [-1, 1])
    wc, wd = tf.reshape(wc, [-1, 1]), tf.reshape(wd, [-1, 1])
    Ibilin = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    Ibilin = tf.reshape(
        Ibilin,
        [
            batch_size,
            nR // batch_size,
            int((nV + 1) ** (1 / 3)),
            int((nV + 1) ** (1 / 3)),
            int((nV + 1) ** (1 / 3)),
            fdim,
        ],
    )
    Ibilin = tf.transpose(Ibilin, [0, 1, 3, 2, 4, 5])
    return Ibilin


def unDistortPoints(
    pts,
    intrinsicMatrix,
    radialDistortion,
    tangentDistortion,
    rotationMatrix,
    translationVector,
):
    """Remove lens distortion from the input points.

    Input is size (M,2), where M is the number of points
    """
    dcoef = radialDistortion.ravel()[:2].tolist() + tangentDistortion.ravel().tolist()

    if len(radialDistortion.ravel()) == 3:
        dcoef = dcoef + [radialDistortion.ravel()[-1]]
    else:
        dcoef = dcoef + [0]

    ts = time.time()
    pts_u = cv2.undistortPoints(
        np.reshape(pts, (-1, 1, 2)).astype("float32"),
        intrinsicMatrix.T,
        np.array(dcoef),
        P=intrinsicMatrix.T,
    )

    pts_u = np.reshape(pts_u, (-1, 2))

    return pts_u


def triangulate(pts1, pts2, cam1, cam2):
    """Return triangulated 3- coordinates.

    Following Matlab convetion, given lists of matching points, and their
    respective camera matrices, returns the triangulated 3- coordinates.
    pts1 and pts2 must be Mx2, where M is the number of points with
    (x,y) positions. M 3-D points will be returned after triangulation
    """
    pts1 = pts1.T
    pts2 = pts2.T

    cam1 = cam1.T
    cam2 = cam2.T

    out_3d = np.zeros((3, pts1.shape[1]))

    for i in range(out_3d.shape[1]):
        if ~np.isnan(pts1[0, i]):
            pt1 = pts1[:, i : i + 1]
            pt2 = pts2[:, i : i + 1]

            A = np.zeros((4, 4))
            A[0:2, :] = pt1 @ cam1[2:3, :] - cam1[0:2, :]
            A[2:, :] = pt2 @ cam2[2:3, :] - cam2[0:2, :]

            u, s, vh = np.linalg.svd(A)
            v = vh.T

            X = v[:, -1]
            X = X / X[-1]

            out_3d[:, i] = X[0:3].T
        else:
            out_3d[:, i] = np.nan

    return out_3d


def triangulate_multi_instance(pts, cams):
    """Return triangulated 3- coordinates.

    Following Matlab convetion, given lists of matching points, and their
    respective camera matrices, returns the triangulated 3- coordinates.
    pts1 and pts2 must be Mx2, where M is the number of points with
    (x,y) positions. M 3-D points will be returned after triangulation
    """
    pts = [pt.T for pt in pts]
    cams = [c.T for c in cams]
    out_3d = np.zeros((3, pts[0].shape[1]))
    # traces = np.zeros((out_3d.shape[1],))

    for i in range(out_3d.shape[1]):
        if ~np.isnan(pts[0][0, i]):
            p = [p[:, i : i + 1] for p in pts]

            A = np.zeros((2 * len(cams), 4))
            for j in range(len(cams)):
                A[(j) * 2 : (j + 1) * 2] = p[j] @ cams[j][2:3, :] - cams[j][0:2, :]

            u, s, vh = np.linalg.svd(A)
            v = vh.T

            X = v[:, -1]
            X = X / X[-1]

            out_3d[:, i] = X[0:3].T
            # traces[i] = np.sum(s[0:3])

        else:
            out_3d[:, i] = np.nan

    return out_3d


def ravel_multi_index(I, J, shape):
    """Create an array of flat indices from coordinate arrays.

    shape is (rows, cols)
    """
    r, c = shape
    return I * c + J


def collapse_dims(T):
    """Collapse dimensions."""
    shape = list(K.int_shape(T))
    return tf.reshape(T, [-1] + shape[2:])


def repeat_tensor(T, nrep, rep_dim=1):
    """Repeat tensor."""
    repT = tf.expand_dims(T, rep_dim)
    tile_dim = [1] * len(K.int_shape(repT))
    tile_dim[rep_dim] = nrep
    repT = tf.tile(repT, tile_dim)
    return repT


def nearest3(grid, idx, clip=False):
    """TODO(Describe): I'm having a hard time reading this one."""
    with tf.variable_scope("NearestInterp"):
        _, h, w, d, f = grid.get_shape().as_list()
        x, y, z = idx[:, 1], idx[:, 2], idx[:, 3]
        g_val = tf.gather_nd(grid, tf.cast(tf.round(idx), "int32"))
        if clip:
            x_inv = tf.logical_or(x < 0, x > h - 1)
            y_inv = tf.logical_or(y < 0, y > w - 1)
            z_inv = tf.logical_or(z < 0, x > d - 1)
            valid_idx = 1 - tf.to_float(
                tf.logical_or(tf.logical_or(x_inv, y_inv), z_inv)
            )
            g_val = g_val * valid_idx[..., tf.newaxis]
        return g_val


# Todo(simplify): This function had many comments that could be condensed
def proj_slice(
    vmin,
    vmax,
    nvox,
    rs_grid,
    grid,
    K_,
    R,
    proj_size=512,
    samples=64,
    min_z=1000.0,
    max_z=2100.0,
):
    """Project slice.

    grid = nv grids, R = nv x nr rotation matrices.
    R = (bs, im, 3, 4), K = (bs, im, 3, 3), grid = (bs, im, h, w, d, ch)

    Modified from https://github.com/akar43/lsm
    """
    # Scale the camera intrinsic matrix accordingly if the final output is
    # a different shape than the input
    # Maybe best to start with the native image size so we don't have to deal
    # with this headache
    rsz_factor = 1
    K_ = K_ * rsz_factor
    K_shape = K.int_shape(K_)

    bs, im_bs, h, w, d, ch = K.int_shape(grid)
    npix = proj_size ** 2

    # Compute Xc - points in camera frame
    Xc = tf.matrix_triangular_solve(K_, rs_grid, lower=False, name="KinvX")

    print(K.int_shape(Xc))

    # Define z values of samples along ray
    z_samples = tf.linspace(min_z, max_z, samples)

    # Transform Xc to Xw using transpose of rotation matrix
    Xc = repeat_tensor(Xc, samples, rep_dim=2)
    Xc = Xc * z_samples[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis]
    Xc = tf.concat([Xc, tf.ones([K_shape[0], K_shape[1], samples, 1, npix])], axis=-2)

    # Construct [R^{T}|-R^{T}t]
    Rt = tf.matrix_transpose(R[:, :, :, :3])
    tr = tf.expand_dims(R[:, :, :, 3], axis=-1)
    R_c2w = tf.concat([Rt, -tf.matmul(Rt, tr)], axis=3)
    R_c2w = repeat_tensor(R_c2w, samples, rep_dim=2)
    Xw = tf.matmul(R_c2w, Xc)

    # But remember, some rays/world points will not contact the grid --
    # Took me a day to figure out, but the trick is that the x-coordinate was
    # indexing the rows rather than the columns, so the grid needs to be fed
    # in with the first two grid dimensions permuted...
    vmin = vmin[:, tf.newaxis, tf.newaxis, :, tf.newaxis]
    vmax = vmax[:, tf.newaxis, tf.newaxis, :, tf.newaxis]
    Xw = ((Xw - vmin) / (vmax - vmin)) * nvox
    # size now (bs, num_cams, samples, npix, 3)
    Xw = tf.transpose(Xw, [0, 1, 2, 4, 3])
    # size now (bs, num_grids, num_cams, samples, npix, 3)
    Xw = repeat_tensor(Xw, im_bs, rep_dim=1)

    # Todo(describe): Describe these operations in concepts rather than linalg
    sample_grid = collapse_dims(grid)
    sample_locs = collapse_dims(Xw)
    lshape = K.int_shape(sample_locs)
    vox_idx = tf.range(lshape[0])
    vox_idx = repeat_tensor(vox_idx, lshape[1], rep_dim=1)
    vox_idx = tf.reshape(vox_idx, [-1, 1])
    vox_idx = repeat_tensor(vox_idx, samples * npix, rep_dim=1)
    vox_idx = tf.reshape(vox_idx, [-1, 1])
    sample_idx = tf.concat(
        [tf.to_float(vox_idx), tf.reshape(sample_locs, [-1, 3])], axis=1
    )

    # The first column indicates which "grid" should be sampled for each
    # x,y,z position. In my case, there should only be as many grids as there
    # are samples in the mini-batch,
    # but for some reason this code allows multiple 3D grids per sample.
    # the order in rows (for the last 3 cols) should be rougly like this:
    # [batch1_grid1_allcam1samples_locs, batch1_grid1_allcam2sample_locs,
    # batch1_grid1_allcam3sample_locs, batch1_grid2_allcam1samples_locs, ...]
    g_val = nearest3(sample_grid, sample_idx, clip=True)
    g_val = tf.reshape(
        g_val, [bs, im_bs, K_shape[1], samples, proj_size, proj_size, -1]
    )
    ray_slices = tf.transpose(g_val, [0, 1, 2, 4, 5, 6, 3])
    return K.max(ray_slices, axis=-1, keepdims=False)


class InstanceNormalization(Layer):
    """Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).

    Modified from from keras_contrib/layer/normalization

    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each instance of
            the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid
            errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization]
            (https://arxiv.org/abs/1607.08022)
    """

    def __init__(
        self,
        axis=None,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs
    ):
        """Initialize instance normalization."""
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        """Build instance normalization."""
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError("Axis cannot be zero")

        if (self.axis is not None) and (ndim == 2):
            raise ValueError("Cannot specify axis for rank 1 tensor")

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        """Call instance normalization."""
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        """Reuturn configuration."""
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# TODO(comment): Reading requires knowledge of the get_custom_objects function
get_custom_objects().update({"InstanceNormalization": InstanceNormalization})


def distortPoints(points, intrinsicMatrix, radialDistortion, tangentialDistortion):
    """Distort points according to camera parameters.

    Ported from Matlab 2018a
    """
    # unpack the intrinisc matrix
    cx = intrinsicMatrix[2, 0]
    cy = intrinsicMatrix[2, 1]
    fx = intrinsicMatrix[0, 0]
    fy = intrinsicMatrix[1, 1]
    skew = intrinsicMatrix[1, 0]

    # center the points
    center = np.array([cx, cy])
    centeredPoints = points - center[np.newaxis, :]

    # normalize the points
    yNorm = centeredPoints[:, 1] / fy
    xNorm = (centeredPoints[:, 0] - skew * yNorm) / fx

    # compute radial distortion
    r2 = xNorm ** 2 + yNorm ** 2
    r4 = r2 * r2
    r6 = r2 * r4

    k = np.zeros((3,))
    k[:2] = radialDistortion[:2]
    if len(radialDistortion) < 3:
        k[2] = 0
    else:
        k[2] = radialDistortion[2]
    alpha = k[0] * r2 + k[1] * r4 + k[2] * r6

    # compute tangential distortion
    p = tangentialDistortion
    xyProduct = xNorm * yNorm
    dxTangential = 2 * p[0] * xyProduct + p[1] * (r2 + 2 * xNorm ** 2)
    dyTangential = p[0] * (r2 + 2 * yNorm ** 2) + 2 * p[1] * xyProduct

    # apply the distortion to the points
    normalizedPoints = np.stack((xNorm, yNorm)).T
    distortedNormalizedPoints = (
        normalizedPoints
        + normalizedPoints * np.array([alpha, alpha]).T
        + np.stack((dxTangential, dyTangential)).T
    )

    # # convert back to pixels
    distortedPointsX = (
        (distortedNormalizedPoints[:, 0] * fx)
        + cx
        + (skew * distortedNormalizedPoints[:, 1])
    )
    distortedPointsY = distortedNormalizedPoints[:, 1] * fy + cy
    distortedPoints = np.stack((distortedPointsX, distortedPointsY))

    return distortedPoints


def distortPoints_torch(
    points, intrinsicMatrix, radialDistortion, tangentialDistortion, device
):
    """Distort points according to camera parameters.
    Ported from Matlab 2018a
    """
    import torch

    # unpack the intrinsic matrix
    cx = intrinsicMatrix[2, 0]
    cy = intrinsicMatrix[2, 1]
    fx = intrinsicMatrix[0, 0]
    fy = intrinsicMatrix[1, 1]
    skew = intrinsicMatrix[1, 0]

    # center the points
    center = torch.as_tensor((cx, cy), dtype=torch.float32, device=device)
    centeredPoints = points - center

    # normalize the pcenteredPoints[:, 1] / fyoints
    yNorm = centeredPoints[:, 1] / fy
    xNorm = (centeredPoints[:, 0] - skew * yNorm) / fx

    # compute radial distortion
    r2 = xNorm ** 2 + yNorm ** 2
    r4 = r2 * r2
    r6 = r2 * r4

    k = np.zeros((3,))
    k[:2] = radialDistortion[:2]
    if list(radialDistortion.shape)[0] < 3:
        k[2] = 0
    else:
        k[2] = radialDistortion[2]
    alpha = k[0] * r2 + k[1] * r4 + k[2] * r6

    # compute tangential distortion
    p = tangentialDistortion
    xyProduct = xNorm * yNorm
    dxTangential = 2 * p[0] * xyProduct + p[1] * (r2 + 2 * xNorm ** 2)
    dyTangential = p[0] * (r2 + 2 * yNorm ** 2) + 2 * p[1] * xyProduct

    # apply the distortion to the points
    normalizedPoints = torch.transpose(torch.stack((xNorm, yNorm)), 0, 1)

    distortedNormalizedPoints = (
        normalizedPoints
        + normalizedPoints * torch.transpose(torch.stack((alpha, alpha)), 0, 1)
        + torch.transpose(torch.stack((dxTangential, dyTangential)), 0, 1)
    )

    distortedPointsX = (
        distortedNormalizedPoints[:, 0] * fx
        + cx
        + skew * distortedNormalizedPoints[:, 1]
    )

    distortedPointsY = distortedNormalizedPoints[:, 1] * fy + cy

    distortedPoints = torch.stack((distortedPointsX, distortedPointsY))

    return distortedPoints


@tf.function
def distortPoints_tf(points, intrinsicMatrix, radialDistortion, tangentialDistortion):
    """Distort points according to camera parameters.

    Ported from Matlab 2018a
    """
    # unpack the intrinsic matrix
    cx = intrinsicMatrix[2, 0]
    cy = intrinsicMatrix[2, 1]
    fx = intrinsicMatrix[0, 0]
    fy = intrinsicMatrix[1, 1]
    skew = intrinsicMatrix[1, 0]

    # center the points
    center = tf.stack((cx, cy))
    p = tangentialDistortion
    centeredPoints = points - center

    # normalize the pcenteredPoints[:, 1] / fyoints
    yNorm = centeredPoints[:, 1] / fy
    xNorm = (centeredPoints[:, 0] - skew * yNorm) / fx

    # compute radial distortion
    r2 = xNorm ** 2 + yNorm ** 2
    r4 = r2 * r2
    r6 = r2 * r4

    k = radialDistortion
    if list(radialDistortion.shape)[0] < 3:
        k[2] = 0
    alpha = k[0] * r2 + k[1] * r4 + k[2] * r6

    # compute tangential distortion
    xyProduct = xNorm * yNorm
    dxTangential = 2 * p[0] * xyProduct + p[1] * (r2 + 2 * xNorm ** 2)
    dyTangential = p[0] * (r2 + 2 * yNorm ** 2) + 2 * p[1] * xyProduct

    # apply the distortion to the points
    normalizedPoints = tf.transpose(tf.stack((xNorm, yNorm)))

    distortedNormalizedPoints = (
        normalizedPoints
        + normalizedPoints * tf.transpose(tf.stack((alpha, alpha)))
        + tf.transpose(tf.stack((dxTangential, dyTangential)))
    )

    distortedPointsX = (
        distortedNormalizedPoints[:, 0] * fx
        + cx
        + skew * distortedNormalizedPoints[:, 1]
    )

    distortedPointsY = distortedNormalizedPoints[:, 1] * fy + cy

    distortedPoints = tf.stack((distortedPointsX, distortedPointsY))

    return distortedPoints


def expected_value_3d(prob_map, grid_centers):
    """Calculate expected value of spatial distribution over output 3D grid.

    prob_map should be (batch_size,h,w,d,channels)
    grid_centers should be (batch_size,h*w*d,3)
    # For this to work, the values in a single prob map channel must sum to one
    """
    bs, h, w, d, channels = K.int_shape(prob_map)

    prob_map = tf.reshape(prob_map, [-1, channels])

    grid_centers = tf.reshape(grid_centers, [-1, 3])

    weighted_centers = prob_map[:, tf.newaxis, :] * grid_centers[:, :, tf.newaxis]

    # weighted centers now (bs*h*w*d,3,channels).
    # So we now sum over the grid to get 3D coordinates
    # first reshape to put batch_size back on its own axis
    weighted_centers = tf.reshape(weighted_centers, [-1, h * w * d, 3, channels])
    weighted_centers = tf.reduce_sum(weighted_centers, axis=1)
    return weighted_centers


def spatial_softmax(feats):
    """Normalize acros channels.

    Channel/marker-wise softmax normalization so that each 3d probability map
    represents a normalized probability distribution feats enters as size
    (bs, h, w, d, channels) but needs to reshapes to
    (bs, h*w*d, channels) for the softmax, then
    reshaped back again
    """
    bs, h, w, d, channels = K.int_shape(feats)
    feats = tf.reshape(feats, [-1, h * w * d, channels])
    feats = tf.nn.softmax(feats, axis=1)
    feats = tf.reshape(feats, [-1, h, w, d, channels])
    return feats


def var_3d(prob_map, grid_centers, markerlocs):
    """Return the average variance across all marker probability maps.

    Used a loss to promote "peakiness" in the probability map output
    prob_map should be (batch_size,h,w,d,channels)
    grid_centers should be (batch_size,h*w*d,3)
    markerlocs is (batch_size,3,channels)
    """
    bs, h, w, d, channels = K.int_shape(prob_map)

    prob_map = tf.reshape(prob_map, [-1, channels])

    # we need the squared distance between all grid centers and
    # the mean for each channel grid dist now (bs, h*w*d,17)
    grid_dist = tf.reduce_sum(
        (grid_centers[:, :, :, tf.newaxis] - markerlocs[:, tf.newaxis, :, :]) ** 2,
        axis=2,
    )
    grid_dist = tf.reshape(grid_dist, [-1, channels])
    weighted_var = prob_map * grid_dist
    weighted_var = tf.reshape(weighted_var, [-1, h * w * d, channels])
    weighted_var = tf.reduce_sum(weighted_var, axis=1)
    return tf.reduce_mean(weighted_var, axis=-1)[:, tf.newaxis]
