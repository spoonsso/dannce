import keras
from keras import backend as K
import tensorflow as tf

def mask_nan_keep_loss(y_true,y_pred):
    nan_true = K.cast(~tf.is_nan(y_true),'float32')
    num_notnan = K.sum(K.flatten(nan_true))
    y_pred = tf.multiply(y_pred,nan_true)
    # y_true = tf.multiply(y_true,nan_true)

    # We need to use tf.where to do this substitution, because when trying to multiply with just the nan_true masks, NaN*0 = NaN, so NaNs are not removed
    y_true = K.cast(tf.where(~tf.is_nan(y_true),y_true,tf.zeros_like(y_true)),'float32')

    #y_pred = K.cast(tf.where(~tf.is_nan(y_pred),y_pred,tf.zeros_like(y_pred)),'float32')
#     y_pred = tf.multiply(y_pred,tf.cast(K.not_equal(y_true,0),'float32'))
    loss = K.sum((K.flatten(y_pred)-K.flatten(y_true))**2)/num_notnan
    return loss#tf.where(~tf.is_nan(loss),loss,0)

def mask_nan_keep_loss_safe(y_true,y_pred):
    nan_true = K.cast(~tf.is_nan(y_true),'float32')
    num_notnan = K.sum(K.flatten(nan_true))
    y_pred = tf.multiply(y_pred,nan_true)
    # y_true = tf.multiply(y_true,nan_true)

    # We need to use tf.where to do this substitution, because when trying to multiply with just the nan_true masks, NaN*0 = NaN, so NaNs are not removed
    y_true = K.cast(tf.where(~tf.is_nan(y_true),y_true,tf.zeros_like(y_true)),'float32')

    #y_pred = K.cast(tf.where(~tf.is_nan(y_pred),y_pred,tf.zeros_like(y_pred)),'float32')
#     y_pred = tf.multiply(y_pred,tf.cast(K.not_equal(y_true,0),'float32'))
    # if num_notnan ==0:
    # 	return 0
    loss = K.sum((K.flatten(y_pred)-K.flatten(y_true))**2)/num_notnan


    # if num_notnan == 0:
    # 	print("WARNING: no valid values in loss")

    # loss = tf.where(~tf.is_nan(loss),loss,0)
    # if loss ==0:
    # 	print("WARNING: this batch loss is all zero")
    # print("positive controls")	
    return tf.where(~tf.is_nan(loss),loss,0)

def metric_dist_max(y_true,y_pred):
    #y_true and y_pred are image-sized confidence maps. Let's get the (row, col) indicies of each maximum and calculate the
    # distance between the two
    x = K.reshape(y_true, [K.int_shape(y_true)[0], -1])
    indices = K.argmax(x, axis=1)

    col_indices_true = indices / K.int_shape(y_true)[1]
    row_indices_true = indices % K.int_shape(y_true)[1]

    x = K.reshape(y_pred, [K.int_shape(y_pred)[0], -1])
    indices = K.argmax(x, axis=1)

    col_indices_pred = indices / K.int_shape(y_pred)[1]
    row_indices_pred = indices % K.int_shape(y_pred)[1]

    dist = K.sqrt(K.pow(col_indicies_pred-col_indices_true,2) + K.pow(row_indicies_pred-row_indices_true,2))


    return K.mean(dist)

def mse_with_var_regularization(y_true,y_pred):

    return K.mean(y_true)

def identity_pred(y_true,y_pred):
    """
    I created this loss to work with the variance regularizer accompanying my spatial expected value networks
    """

    return K.mean(K.flatten(y_pred))
    
def euclidean_distance_3D(y_true,y_pred):
    """
    Assumes predictions of shape (batch_size,3*num_markers)
    """
    return K.mean(K.flatten(K.sqrt(K.sum(K.pow(y_true-y_pred,2),axis=1))))

def centered_euclidean_distance_3D(y_true,y_pred):
    """
    Assumes predictions of shape (batch_size,3,num_markers)
    """
    y_true = y_true - K.mean(y_true,axis=-1,keepdims=True)
    y_pred = y_pred - K.mean(y_pred,axis=-1,keepdims=True)
    return K.mean(K.flatten(K.sqrt(K.sum(K.pow(y_true-y_pred,2),axis=1))))