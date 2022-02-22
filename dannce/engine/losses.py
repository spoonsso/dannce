"""Losses for tf models."""
import tensorflow as tf
from tensorflow.keras import backend as K

def mask_nan(y_true, y_pred):
    """Mask nans and return tensors for use by loss functions
    """
    notnan_true = K.cast(~tf.math.is_nan(y_true), "float32")
    num_notnan = K.sum(K.flatten(notnan_true))
    y_pred = tf.math.multiply(y_pred, notnan_true)

    # We need to use tf.where to do this substitution, because when trying to
    # multiply with just the notnan_true masks,
    # NaN*0 = NaN, so NaNs are not removed
    y_true = K.cast(
        tf.where(~tf.math.is_nan(y_true), y_true, tf.zeros_like(y_true)), "float32"
    )
    return y_pred, y_true, num_notnan


def mask_nan_keep_loss(y_true, y_pred):
    """Mask out nan values in the calulation of MSE."""
    y_pred, y_true, num_notnan = mask_nan(y_true, y_pred)
    loss = K.sum((K.flatten(y_pred) - K.flatten(y_true)) ** 2) / num_notnan
    return tf.where(~tf.math.is_nan(loss), loss, 0)


def mask_nan_l1_loss(y_true, y_pred):
    y_pred, y_true, num_notnan = mask_nan(y_true, y_pred)
    loss = K.sum(K.abs(K.flatten(y_pred) - K.flatten(y_true))) / num_notnan
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

    return loss #tf.where(~tf.math.is_inf(loss), loss, 0)


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

def heatmap_max_regularizer(y_true, y_pred):
    """Regularizes 3d confidence maps by maximizing density in the voxel
    containing the GT coordinate.

    y_true contains the beta coefficient, just repeated for each example, with
    output: -beta*log(V_output(gt_voxel)) 

    """

    return -1*K.mean(K.flatten(y_true)*K.log(K.flatten(y_pred)))

# Huber and Cosh losses copied from implementation by robb
def huber_loss(delta):
    def huber_model(y_true,y_pred):
         y_pred, y_true, num_notnan = mask_nan(y_true, y_pred)

         model = tf.keras.losses.Huber(delta=delta,reduction=tf.keras.losses.Reduction.SUM)
         h = model((y_true), (y_pred))/num_notnan

         loss = h
         return tf.where(~tf.math.is_nan(loss), loss, 0)
         
    return huber_model

def log_cosh_loss(y_true, y_pred):
    y_pred, y_true, num_notnan = mask_nan(y_true, y_pred)
    
    lc_ = tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.SUM)
    lc = lc_(y_true, y_pred)/num_notnan

    loss = lc
    
    return tf.where(~tf.math.is_nan(loss), loss, 0)

def gaussian_cross_entropy_loss(y_true, y_pred):
    """Get cross entropy loss of output distribution and Gaussian centered around target

    Assumes predictions of shape (batch_size,3,num_markers)
    """
    y_pred, y_true, num_notnan = mask_nan(y_true, y_pred)
    loss = K.sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=K.flatten(y_true), logits=K.flatten(y_pred))) / num_notnan
    return tf.where(~tf.math.is_nan(loss), loss, 0)


###
def mask_nan_pair(y1, y2):
    # need a different mask nan function for unsupervised losses
    # since there's no "ground truth"
    notnan = ~tf.math.logical_or(tf.math.is_nan(y1), tf.math.is_nan(y2))
    num_notnan = K.sum(K.flatten(K.cast(notnan, "float32") ))

    y1 = K.cast(tf.where(notnan, y1, tf.zeros_like(y1)), "float32")
    y2 = K.cast(tf.where(notnan, y2, tf.zeros_like(y2)), "float32")
    return y1, y2, num_notnan

def temporal_consistency(y_pred_t1, y_pred_t2):
    """Unsupervised pairwise loss with respect to the temporal dimension.
    """
    assert y_pred_t1.shape == y_pred_t2.shape, "Imput shapes are inconsistent when computing the temporal consistency loss."

    y_pred_t1, y_pred_t2, num_notnan = mask_nan_pair(y_pred_t1, y_pred_t2)

    loss = K.sum((K.flatten(y_pred_t1) - K.flatten(y_pred_t2)) ** 2) / num_notnan
    loss = tf.where(~tf.math.is_nan(loss), loss, 0)

    return loss

def temporal_loss(chunk_size):
    def loss(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, chunk_size, *y_pred.shape[1:])) # [batch size, chunk size, N, 3]
        temp_losses = tf.zeros(())
        for chunk in y_pred:
            for i in range(chunk.shape[0]-1):
                temp_losses += temporal_consistency(chunk[i], chunk[i+1])
        return temp_losses

    return loss

def silhouette_loss(y_true, y_pred, dim=3):
    # y_true and y_pred will both have shape
    # (n_batch, width, height, n_keypts)
    assert dim == 2 or dim == 3, "The silhouette loss only supports 2D or 3D"
    reduce_axes = [1, 2] if dim == 2 else [1, 2, 3]
    sil = K.sum(y_pred * y_true, axis=reduce_axes)
    sil = K.mean(-K.log(sil + 1e-12), axis=-1)
    
    return K.mean(sil, axis=0)

def pair_repulsion_loss(y_pred_s1, y_pred_s2):
    """Unsupervised pairwise loss with respect to two subjects. 
    The predictions should be as far as possible, i.e. repelling each other.
    Input:
        y_pred_s1, y_pred_s2: (B, N, 3)"""

    return 1 / K.sum((y_pred_s1 - y_pred_s2)**2)


def separation_loss(delta=10):
    def _separation_loss(y_true, y_pred):
        """
        Loss which penalizes 3D keypoint predictions being too close.
        """
        num_kpts = y_pred.shape[1]

        t1 = K.tile(y_pred, [1, num_kpts, 1])
        t2 = K.reshape(K.tile(y_pred, [1, 1, num_kpts]), K.shape(t1))

        lensqr = K.sum((t1 - t2) ** 2, axis=2)
        sep = K.sum(K.maximum(delta-lensqr, 0.0), axis=1) / K.cast(num_kpts*num_kpts, "float32")

        return K.mean(sep)
    return _separation_loss
