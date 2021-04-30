"""Losses for tf models."""
import tensorflow as tf
from tensorflow.keras import backend as K

# TODO(nan_true): nan_true is where y_true is not nan. This is confusing
def mask_nan_keep_loss(y_true, y_pred):
    """Mask out nan values in the calulation of MSE."""
    nan_true = K.cast(~tf.math.is_nan(y_true), "float32")
    num_notnan = K.sum(K.flatten(nan_true))
    y_pred = tf.math.multiply(y_pred, nan_true)

    # We need to use tf.where to do this substitution, because when trying to
    # multiply with just the nan_true masks,
    # NaN*0 = NaN, so NaNs are not removed
    y_true = K.cast(
        tf.where(~tf.math.is_nan(y_true), y_true, tf.zeros_like(y_true)), "float32"
    )
    loss = K.sum((K.flatten(y_pred) - K.flatten(y_true)) ** 2) / num_notnan
    return tf.where(~tf.math.is_nan(loss), loss, 0)


def multiview_consistency(y_true, y_pred):
    """
    In a semi-supervised strategy, we have a normal mask_nan mse loss for where there are labels,
    but also a loss that checks whether the output using different combinations of views is the same
    """
    alpha = 1  # 0.0001 # The weight on the multiview loss, which we hard-code for now..

    msk_loss = mask_nan_keep_loss(y_true[-1], y_pred[-1])

    # The output should be (batch_size,nvox,nvox,nvox,n_markers)
    # For a 3-cam system, there are only 3 different possible pairs, so we discard the last, which is the complete set
    y_pred_ = y_pred[:-1]
    y_pred_diff = y_pred_[1:] - y_pred_[:-1]
    multiview_loss = K.mean(K.flatten(y_pred_diff) ** 2)

    return msk_loss + alpha * multiview_loss


def metric_dist_max(y_true, y_pred):
    """Get distance between the (row, col) indices of each maximum.

    y_true and y_pred are image-sized confidence maps.
    Let's get the (row, col) indicies of each maximum and calculate the
    distance between the two
    """
    x = K.reshape(y_true, [K.int_shape(y_true)[0], -1])
    indices = K.argmax(x, axis=1)

    col_indices_true = indices / K.int_shape(y_true)[1]
    row_indices_true = indices % K.int_shape(y_true)[1]

    x = K.reshape(y_pred, [K.int_shape(y_pred)[0], -1])
    indices = K.argmax(x, axis=1)

    col_indices_pred = indices / K.int_shape(y_pred)[1]
    row_indices_pred = indices % K.int_shape(y_pred)[1]

    dist = K.sqrt(
        K.pow(col_indices_pred - col_indices_true, 2)
        + K.pow(row_indices_pred - row_indices_true, 2)
    )
    return K.mean(dist)


def mse_with_var_regularization(y_true, y_pred):
    """Return the mean of y_true."""
    return K.mean(y_true)


def identity_pred(y_true, y_pred):
    """Works with variance regularizer in spatial expected value networks."""
    return K.mean(K.flatten(y_pred))


def K_nanmean_infmean(tensor):
    """
    Returns the nanmean of the input tensor. If tensor is all NaN, returns 0

    Also removes inf
    """
    notnan = K.cast((~tf.math.is_nan(tensor)) & (~tf.math.is_inf(tensor)), "float32")
    num_notnan = K.sum(K.flatten(notnan))

    nonan = K.cast(
        tf.where((~tf.math.is_nan(tensor)) & (~tf.math.is_inf(tensor)),
                 tensor,
                 tf.zeros_like(tensor)), "float32"
    )

    loss = K.sum(nonan) / num_notnan

    return loss#tf.where(~tf.math.is_inf(loss), loss, 0)


def euclidean_distance_3D(y_true, y_pred):
    """Get 3d Euclidean distance.

    Assumes predictions of shape (batch_size,3,num_markers)

    Ignores NaN when necessary. But because K.sqrt(NaN) == inf, whenthere
        are NaNs in the labels, the distance function returns inf
    """
    ed3D = K.flatten(K.sqrt(K.sum(K.pow(y_true - y_pred, 2), axis=1)))
    return K_nanmean_infmean(ed3D)


def centered_euclidean_distance_3D(y_true, y_pred):
    """Get centered 3d Euclidean distance.

    Assumes predictions of shape (batch_size,3,num_markers)

    Ignores NaN when necessary
    """
    y_true = y_true - K.mean(y_true, axis=-1, keepdims=True)
    y_pred = y_pred - K.mean(y_pred, axis=-1, keepdims=True)

    ced3D = K.flatten(K.sqrt(K.sum(K.pow(y_true - y_pred, 2), axis=1)))
    return K_nanmean_infmean(ced3D)
