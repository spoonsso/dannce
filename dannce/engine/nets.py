"""Define networks for dannce."""
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, Conv3D, Lambda
from tensorflow.keras.layers import MaxPooling3D, Conv3DTranspose
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalMaxPooling3D
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from dannce.engine import ops as ops
from dannce.engine import losses as losses
import numpy as np
import h5py
import tensorflow as tf

def get_metrics(params):
    """
    """
    metrics = []
    for m in params["metric"]:
        try:
            m_obj = getattr(losses, m)
        except AttributeError:
            m_obj = getattr(tf.keras.losses, m)
        metrics.append(m_obj)

    return metrics

# TODO (JOSH). Move the if/else normalization block to its own function. And in this function
# add the correct InstanceNormalization() call, using the appropriate axis setting.
def norm_fun(
    norm_method=None,
):
    """
    method: Normalization method can be "batch", "instance", or "layer"
    """
    method_parse = norm_method.lower()
    if method_parse.startswith("batch"):
        print("using batch normalization")

        def fun(inputs):
            print("calling batch norm fun")
            return BatchNormalization()(inputs)
    elif method_parse.startswith("layer"):
        print("using layer normalization")

        def fun(inputs):
            print("calling layer norm fun")
            return ops.InstanceNormalization(axis=None)(inputs)
    elif method_parse.startswith("instance"):
        print("using instance normalization")

        def fun(inputs):
            print("calling instance norm fun")
            return ops.InstanceNormalization(axis=-1)(inputs)
    else:
        def fun(inputs):
            return inputs
    
    return fun

def unet2d_fullbn(
    lossfunc, lr, input_dim, feature_num, metric="mse", include_top=True
):
    """Initialize 2D U-net with batch normalization

    Uses the Keras functional API to construct a U-Net. The net is fully
        convolutional, so it can be trained and tested on variable size input
        (thus the x-y input dimensions are undefined)
    inputs--
        lossfunc: loss function
        lr: float; learning rate
        input_dim: int; number of feature channels in input
        feature_num: int; number of output features
    outputs--
        model: Keras model object
    """
    return unet2d_full(lossfunc, lr, input_dim, feature_num, metric=metric, 
                       include_top=include_top, norm_method="batch")

def unet2d_fullIN(
    lossfunc, lr, input_dim, feature_num, metric="mse", include_top=True
):
    """
    Initialize 2D U-net with instance normalization

    Uses the Keras functional API to construct a U-Net. The net is fully convolutional, so it can be trained
        and tested on variable size input (thus the x-y input dimensions are undefined)
    inputs--
        lossfunc: loss function
        lr: float; learning rate
        input_dim: int; number of feature channels in input
        feature_num: int; number of output features
    outputs--
        model: Keras model object
    """

    return unet2d_full(lossfunc, lr, input_dim, feature_num, metric=metric, 
                       include_top=include_top, norm_method="instance")

def unet2d_full(
    lossfunc, lr, input_dim, feature_num, metric="mse", include_top=True,
    norm_method="layer"
):
    """Initialize 2D U-net.

    Uses the Keras functional API to construct a U-Net. The net is fully
        convolutional, so it can be trained and tested on variable size input
        (thus the x-y input dimensions are undefined)
    inputs--
        lossfunc: loss function
        lr: float; learning rate
        input_dim: int; number of feature channels in input
        feature_num: int; number of output features
        norm_method: str; normalization method ("instance","batch","layer",None)
    outputs--
        model: Keras model object
    """
    fun = norm_fun(norm_method)

    inputs = Input((None, None, input_dim))
    conv1 = Conv2D(32, (3, 3), padding="same")(inputs)
    conv1 = Activation("relu")(fun(conv1))
    conv1 = Conv2D(32, (3, 3), padding="same")(conv1)
    conv1 = Activation("relu")(fun(conv1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding="same")(pool1)
    conv2 = Activation("relu")(fun(conv2))
    conv2 = Conv2D(64, (3, 3), padding="same")(conv2)
    conv2 = Activation("relu")(fun(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same")(pool2)
    conv3 = Activation("relu")(fun(conv3))
    conv3 = Conv2D(128, (3, 3), padding="same")(conv3)
    conv3 = Activation("relu")(fun(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same")(pool3)
    conv4 = Activation("relu")(fun(conv4))
    conv4 = Conv2D(256, (3, 3), padding="same")(conv4)
    conv4 = Activation("relu")(fun(conv4))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same")(pool4)
    conv5 = Activation("relu")(fun(conv5))
    conv5 = Conv2D(512, (3, 3), padding="same")(conv5)
    conv5 = Activation("relu")(fun(conv5))

    up6 = concatenate(
        [Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv5), conv4],
        axis=3,
    )
    conv6 = Conv2D(256, (3, 3), padding="same")(up6)
    conv6 = Activation("relu")(fun(conv6))
    conv6 = Conv2D(256, (3, 3), padding="same")(conv6)
    conv6 = Activation("relu")(fun(conv6))

    up7 = concatenate(
        [Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv6), conv3],
        axis=3,
    )
    conv7 = Conv2D(128, (3, 3), padding="same")(up7)
    conv7 = Activation("relu")(fun(conv7))
    conv7 = Conv2D(128, (3, 3), padding="same")(conv7)
    conv7 = Activation("relu")(fun(conv7))

    up8 = concatenate(
        [Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv7), conv2],
        axis=3,
    )
    conv8 = Conv2D(64, (3, 3), padding="same")(up8)
    conv8 = Activation("relu")(fun(conv8))
    conv8 = Conv2D(64, (3, 3), padding="same")(conv8)
    conv8 = Activation("relu")(fun(conv8))

    up9 = concatenate(
        [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(conv8), conv1],
        axis=3,
    )
    conv9 = Conv2D(32, (3, 3), padding="same")(up9)
    conv9 = Activation("relu")(fun(conv9))
    conv9 = Conv2D(32, (3, 3), padding="same")(conv9)
    conv9 = Activation("relu")(fun(conv9))

    conv10 = Conv2D(feature_num, (1, 1), activation="sigmoid")(conv9)

    if include_top:
        model = Model(inputs=[inputs], outputs=[conv10])
    else:
        model = Model(inputs=[inputs], outputs=[conv9])

    model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=[metric])

    return model

def unet3d_big_expectedvalue(
    lossfunc,
    lr,
    input_dim,
    feature_num,
    num_cams,
    gridsize=(64, 64, 64),
    norm_method="layer",
    include_top=True,
    regularize_var=False,
    loss_weights=None,
    metric=["mse"],
    out_kernel=(1, 1, 1),
):

    fun = norm_fun(norm_method)

    inputs = Input((*gridsize, input_dim * num_cams), name="image_input")
    conv1_layer = Conv3D(64, (3, 3, 3), padding="same")

    conv1 = conv1_layer(inputs)
    conv1 = Activation("relu")(fun(conv1))
    conv1 = Conv3D(64, (3, 3, 3), padding="same")(conv1)
    conv1 = Activation("relu")(fun(conv1))
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), padding="same")(pool1)
    conv2 = Activation("relu")(fun(conv2))
    conv2 = Conv3D(128, (3, 3, 3), padding="same")(conv2)
    conv2 = Activation("relu")(fun(conv2))
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(256, (3, 3, 3), padding="same")(pool2)
    conv3 = Activation("relu")(fun(conv3))
    conv3 = Conv3D(256, (3, 3, 3), padding="same")(conv3)
    conv3 = Activation("relu")(fun(conv3))
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(512, (3, 3, 3), padding="same")(pool3)
    conv4 = Activation("relu")(fun(conv4))
    conv4 = Conv3D(512, (3, 3, 3), padding="same")(conv4)
    conv4 = Activation("relu")(fun(conv4))

    up6 = concatenate(
        [
            Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv4),
            conv3,
        ],
        axis=4,
    )
    conv6 = Conv3D(256, (3, 3, 3), padding="same")(up6)
    conv6 = Activation("relu")(fun(conv6))
    conv6 = Conv3D(256, (3, 3, 3), padding="same")(conv6)
    conv6 = Activation("relu")(fun(conv6))

    up7 = concatenate(
        [
            Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv6),
            conv2,
        ],
        axis=4,
    )
    conv7 = Conv3D(128, (3, 3, 3), padding="same")(up7)
    conv7 = Activation("relu")(fun(conv7))
    conv7 = Conv3D(128, (3, 3, 3), padding="same")(conv7)
    conv7 = Activation("relu")(fun(conv7))

    up8 = concatenate(
        [
            Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv7),
            conv1,
        ],
        axis=4,
    )
    conv8 = Conv3D(64, (3, 3, 3), padding="same")(up8)
    conv8 = Activation("relu")(fun(conv8))
    conv8 = Conv3D(64, (3, 3, 3), padding="same")(conv8)
    conv8 = Activation("relu")(fun(conv8))

    conv10 = Conv3D(feature_num, out_kernel, activation="linear", padding="same")(conv8)

    grid_centers = Input((None, 3), name="grid_input")

    conv10 = Lambda(lambda x: ops.spatial_softmax(x), name="normed_map")(conv10)

    output = Lambda(lambda x: ops.expected_value_3d(x[0], x[1]), name="final_output")([conv10, grid_centers])

    # Because I think it is easier, use a layer to calculate the variance and return it as a second output to be used for variance loss

    output_var = Lambda(lambda x: ops.var_3d(x[0], x[1], x[2]))(
        [conv10, grid_centers, output]
    )

    if include_top:
        if regularize_var:
            model = Model(inputs=[inputs, grid_centers], outputs=[output, output_var])
        else:
            model = Model(inputs=[inputs, grid_centers], outputs=[output])
    else:
        model = Model(inputs=[inputs], outputs=[conv8])

    # model.compile(optimizer=Adam(lr=lr), loss=[lossfunc[0], lossfunc[1]], metrics=['mse'])
    model.compile(
        optimizer=Adam(lr=lr), loss=lossfunc, metrics=metric, loss_weights=loss_weights
    )

    return model


def slice_input(inp, k):
    print(K.int_shape(inp))
    return inp[:, :, :, :, k * 3 : (k + 1) * 3]

def unet3d_big_1cam(
    lossfunc,
    lr,
    input_dim,
    feature_num,
    num_cams,
    norm_method="layer",
):

    fun = norm_fun(norm_method)

    inputs = Input((None, None, None, input_dim))
    conv1_layer = Conv3D(64, (3, 3, 3), padding="same")

    conv1 = conv1_layer(inputs)
    conv1 = Activation("relu")(fun(conv1))
    conv1 = Conv3D(64, (3, 3, 3), padding="same")(conv1)
    conv1 = Activation("relu")(fun(conv1))
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), padding="same")(pool1)
    conv2 = Activation("relu")(fun(conv2))
    conv2 = Conv3D(128, (3, 3, 3), padding="same")(conv2)
    conv2 = Activation("relu")(fun(conv2))
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(256, (3, 3, 3), padding="same")(pool2)
    conv3 = Activation("relu")(fun(conv3))
    conv3 = Conv3D(256, (3, 3, 3), padding="same")(conv3)
    conv3 = Activation("relu")(fun(conv3))
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(512, (3, 3, 3), padding="same")(pool3)
    conv4 = Activation("relu")(fun(conv4))
    conv4 = Conv3D(512, (3, 3, 3), padding="same")(conv4)
    conv4 = Activation("relu")(fun(conv4))

    up6 = concatenate(
        [
            Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv4),
            conv3,
        ],
        axis=4,
    )
    conv6 = Conv3D(256, (3, 3, 3), padding="same")(up6)
    conv6 = Activation("relu")(fun(conv6))
    conv6 = Conv3D(256, (3, 3, 3), padding="same")(conv6)
    conv6 = Activation("relu")(fun(conv6))

    up7 = concatenate(
        [
            Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv6),
            conv2,
        ],
        axis=4,
    )
    conv7 = Conv3D(128, (3, 3, 3), padding="same")(up7)
    conv7 = Activation("relu")(fun(conv7))
    conv7 = Conv3D(128, (3, 3, 3), padding="same")(conv7)
    conv7 = Activation("relu")(fun(conv7))

    up8 = concatenate(
        [
            Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv7),
            conv1,
        ],
        axis=4,
    )
    conv8 = Conv3D(64, (3, 3, 3), padding="same")(up8)
    conv8 = Activation("relu")(fun(conv8))
    conv8 = Conv3D(64, (3, 3, 3), padding="same")(conv8)
    conv8 = Activation("relu")(fun(conv8))

    conv10 = Conv3D(feature_num, (1, 1, 1), activation="sigmoid")(conv8)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=["mse"])

    return model

def unet3d_big(
    lossfunc,
    lr,
    input_dim,
    feature_num,
    num_cams,
    norm_method="layer",
    include_top=True,
    last_kern_size=(1, 1, 1),
    gridsize=None,
):
    # Gridsize unused, necessary for argument consistency with other nets
    fun = norm_fun(norm_method)

    inputs = Input((64, 64, 64, input_dim * num_cams))
    conv1 = Conv3D(64, (3, 3, 3), padding="same")(inputs)
    conv1 = Activation("relu")(fun(conv1))
    conv1 = Conv3D(64, (3, 3, 3), padding="same")(conv1)
    conv1 = Activation("relu")(fun(conv1))
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), padding="same")(pool1)
    conv2 = Activation("relu")(fun(conv2))
    conv2 = Conv3D(128, (3, 3, 3), padding="same")(conv2)
    conv2 = Activation("relu")(fun(conv2))
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(256, (3, 3, 3), padding="same")(pool2)
    conv3 = Activation("relu")(fun(conv3))
    conv3 = Conv3D(256, (3, 3, 3), padding="same")(conv3)
    conv3 = Activation("relu")(fun(conv3))
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(512, (3, 3, 3), padding="same")(pool3)
    conv4 = Activation("relu")(fun(conv4))
    conv4 = Conv3D(512, (3, 3, 3), padding="same")(conv4)
    conv4 = Activation("relu")(fun(conv4))

    up6 = concatenate(
        [
            Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv4),
            conv3,
        ],
        axis=4,
    )
    conv6 = Conv3D(256, (3, 3, 3), padding="same")(up6)
    conv6 = Activation("relu")(fun(conv6))
    conv6 = Conv3D(256, (3, 3, 3), padding="same")(conv6)
    conv6 = Activation("relu")(fun(conv6))

    up7 = concatenate(
        [
            Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv6),
            conv2,
        ],
        axis=4,
    )
    conv7 = Conv3D(128, (3, 3, 3), padding="same")(up7)
    conv7 = Activation("relu")(fun(conv7))
    conv7 = Conv3D(128, (3, 3, 3), padding="same")(conv7)
    conv7 = Activation("relu")(fun(conv7))

    up8 = concatenate(
        [
            Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv7),
            conv1,
        ],
        axis=4,
    )
    conv8 = Conv3D(64, (3, 3, 3), padding="same")(up8)
    conv8 = Activation("relu")(fun(conv8))
    conv8 = Conv3D(64, (3, 3, 3), padding="same")(conv8)
    conv8 = Activation("relu")(fun(conv8))

    if "gaussian_cross_entropy_loss" in str(lossfunc):
        conv10 = Conv3D(feature_num, last_kern_size, activation="linear")(conv8)
    else:
        conv10 = Conv3D(feature_num, last_kern_size, activation="sigmoid")(conv8)

    if include_top:
        model = Model(inputs=[inputs], outputs=[conv10])
    else:
        model = Model(inputs=[inputs], outputs=[conv8])

    model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=["mse"])

    return model

def finetune_AVG(
    lossfunc,
    lr,
    input_dim,
    feature_num,
    num_cams,
    new_last_kern_size,
    new_n_channels_out,
    weightspath,
    num_layers_locked=2,
    norm_method="layer",
    gridsize=(64, 64, 64),
):
    """
    makes necessary calls to network constructors to set up nets for fine-tuning
    the spatial average version of the network.

    num_layers_locked (int) is the number of layers, starting from the input layer,
    that will be locked (non-trainable) during fine-tuning.
    """
    # model = netobj()
    model = unet3d_big_expectedvalue(
        lossfunc,
        lr,
        input_dim,
        feature_num,
        num_cams,
        gridsize,
        norm_method,
        include_top=False,
    )
    pre = model.get_weights()
    # Load weights
    model = renameLayers(model, weightspath)

    post = model.get_weights()

    print("evaluating weight deltas in the first conv layer")

    print("pre-weights")
    print(pre[1][0])
    print("post-weights")
    print(post[1][0])
    print("delta:")
    print(np.sum(pre[1][0] - post[1][0]))

    # Lock desired number of layers
    for layer in model.layers[:num_layers_locked]:
        layer.trainable = False

        # Do forward pass all the way until end
    input_ = Input((*gridsize, input_dim * num_cams), name="image_input")

    old_out = model(input_)

    # Add new output conv. layer
    new_conv = Conv3D(
        new_n_channels_out, new_last_kern_size, activation="linear", padding="same"
    )(old_out)

    grid_centers = Input((None, 3), name="grid_input")

    new_conv2 = Lambda(lambda x: ops.spatial_softmax(x), name="normed_map")(new_conv)

    output = Lambda(lambda x: ops.expected_value_3d(x[0], x[1]), name="final_output")(
        [new_conv2, grid_centers]
    )

    model = Model(inputs=[input_, grid_centers], outputs=[output])

    return model

def finetune_fullmodel_AVG(
    lossfunc,
    lr,
    input_dim,
    feature_num,
    num_cams,
    new_last_kern_size,
    new_n_channels_out,
    weightspath,
    num_layers_locked=2,
    norm_method="layer",
    gridsize=(64, 64, 64),
):
    """
    makes necessary calls to network constructors to set up nets for fine-tuning
    the spatial average version of the network, but here starting from a full model
    file, which enables finetuning of a finetuned model.

    num_layers_locked (int) is the number of layers, starting from the input layer,
    that will be locked (non-trainable) during fine-tuning.
    """

    model = load_model(
                weightspath,
                custom_objects={
                    "ops": ops,
                    "slice_input": slice_input,
                    "mask_nan_keep_loss": losses.mask_nan_keep_loss,
                    "mask_nan_l1_loss": losses.mask_nan_l1_loss,
                    "euclidean_distance_3D": losses.euclidean_distance_3D,
                    "centered_euclidean_distance_3D": losses.centered_euclidean_distance_3D,
                },
                compile=False,
            )

    # Unlock all layers so they can be locked later according to num_layers_locked
    for layer in model.layers[1].layers:
        layer.trainable = True
        # Lock desired number of layers
    for layer in model.layers[1].layers[:num_layers_locked]:
        layer.trainable = False

        # Do forward pass all the way until end
    input_ = Input((*gridsize, input_dim * num_cams), name="image_input")

    old_out = model.layers[1](input_)

    # Add new output conv. layer
    new_conv = Conv3D(
        new_n_channels_out, new_last_kern_size, activation="linear", padding="same"
    )(old_out)

    grid_centers = Input((None, 3), name="grid_input")

    new_conv2 = Lambda(lambda x: ops.spatial_softmax(x), name="normed_map")(new_conv)

    output = Lambda(lambda x: ops.expected_value_3d(x[0], x[1]), name="final_output")(
        [new_conv2, grid_centers]
    )

    model = Model(inputs=[input_, grid_centers], outputs=[output])

    return model

def add_exposed_heatmap(model):
    """
    Given a normal AVG model, add an extra output for supervision of the penultimate heatmap representation
    """
    lay = [l.name for l in model.layers]
    if "exposed_heatmap" not in lay:
        model.layers[-1]._name = "final_output"
        model.layers[-4]._name = "exposed_heatmap"
        sigmoid_output = Activation(activations.sigmoid,
                                    name="sigmoid_exposed_hetmap")
        model = Model(
            inputs=[model.layers[0].input, model.layers[-2].input],
            outputs=[model.layers[-1].output, sigmoid_output(model.layers[-4].output)],
        )

    return model

def remove_exposed_heatmap(model):
    """
    Given an AVG+MAX model, removes the exposes heatmap output so that only the continuous AVG output is
        generated.

    To fully support "continued" mode training, this should only be called during dannce-predict, and before
        the p_max output is added to the network.
    """

    lay = [l.name for l in model.layers]
    if "exposed_heatmap" in lay:
        model = Model(
            inputs=[model.get_layer("image_input").input, model.get_layer("grid_input").input],
            outputs=[model.get_layer("final_output").output],
        )

    return model

def heatmap_reg(hmap, inds):
    """
    Returns the value of the 3D hmap at inds
    """
    nvox = K.int_shape(hmap)[1]
    samp = tf.gather_nd(K.reshape(tf.transpose(hmap,
                                               (0, 4, 1, 2, 3)),
                                  (-1, nvox, nvox, nvox)),
                        K.reshape(tf.transpose(K.cast(inds, "int32"),
                                               (0, 2, 1)),
                                  (-1, 1, 3)),
                        batch_dims=1)
    return K.reshape(samp,
                     (-1, K.int_shape(inds)[2]))

def add_heatmap_output(model):
    """
    Given an AVG model, splice on a new input (GT voxel index) and output (amplitude of normalized heatmap at that GT index)
    """
    lay = [l.name for l in model.layers]
    if "heatmap_output" not in lay:
        print("Adding heatmap regularization arm")
        norm_layer = "normed_map"
        image_input_layer = "image_input"
        grid_input_layer = "grid_input"
        final_output_layer = "final_output"
        if norm_layer not in lay:
            # for backwards compatibility with older models running in "continued" mode
            model.get_layer("lambda_4")._name = norm_layer
            model.get_layer("input_3")._name = image_input_layer
            model.get_layer("input_4")._name = grid_input_layer
            model.get_layer("lambda_5")._name = final_output_layer

        gt_vox_ind = Input((3,
                            K.int_shape(model.get_layer(norm_layer).output)[-1]),
                           name="heatmap_input")

        normed_out = model.get_layer(norm_layer).output
        gt_vox_val = Lambda(lambda x: heatmap_reg(x[0], x[1]), name="heatmap_output")(
            [normed_out, gt_vox_ind]
            )

        #import pdb;pdb.set_trace()
        model = Model(
            inputs=[model.get_layer(image_input_layer).input, model.get_layer(grid_input_layer).input, gt_vox_ind],
            outputs=[model.get_layer(final_output_layer).output, gt_vox_val])

        #import pdb;pdb.set_trace()

    return model

def remove_heatmap_output(model, params):
    """
    Remove any heatmap regularizer layers so saved models can be run normally with dannce predict.
    """
    lay = [l.name for l in model.layers]

    if "heatmap_output" in lay:
        print("Removing heatmap regularization arm")

        opt = Adam(lr=float(params["lr"]))
        mets = get_metrics(params)
        l = params["loss"]
        model = Model(
            inputs=[model.get_layer("image_input").input, model.get_layer("grid_input").input],
            outputs=[model.get_layer("final_output").output])

        # Compile so that this model can be loaded in subsequent "continued" mode without needing to recompile.
        # Note that optimizer state will be lost.
        model.compile(
            optimizer=opt,
            loss=l,
            metrics=mets)

    return model

def load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.
    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.
    Arguments:
      group: A pointer to a HDF5 group.
      name: A name of the attributes to load.
    Returns:
      data: Attributes data.

    From the TF/keras hdf5_format.py
    """
    if name not in group.attrs:
        group = group["model_weights"]
    data = [n if isinstance(n, str) else n.decode("utf8") for n in group.attrs[name]]

    return data


def renameLayers(model, weightspath):
    """
    Rename layers in the model if we detect differences from the layer names in
        the weights file.
    """
    with h5py.File(weightspath, "r") as f:
        lnames = load_attributes_from_hdf5_group(f, "layer_names")

    tf2_names = []
    for (i, layer) in enumerate(model.layers):
        tf2_names.append(layer.name)
        if layer.name != lnames[i]:
            print(
                "Correcting mismatch in layer name, model: {}, weights: {}".format(
                    layer.name, lnames[i]
                )
            )
            layer._name = lnames[i]

    import traceback
    try:
        model.load_weights(weightspath, by_name=True, skip_mismatch=True)
    except ValueError:
        print("Loading model weights failed")
        print(traceback.format_exc())


    # We need to change the model layer names back to the TF2 version otherwise the model
    # won't save
    # If no layer names were changed, this won't do anything.
    for (i, layer) in enumerate(model.layers):
        layer._name = tf2_names[i]

    return model


def finetune_MAX(
    lossfunc,
    lr,
    input_dim,
    feature_num,
    num_cams,
    new_last_kern_size,
    new_n_channels_out,
    weightspath,
    num_layers_locked=2,
    norm_method="layer",
    gridsize=(64, 64, 64),
):
    """
    makes necessary calls to network constructors to set up nets for fine-tuning
    the argmax version of the network.
    """

    model = unet3d_big(
        lossfunc,
        lr,
        input_dim,
        feature_num,
        num_cams,
        norm_method="layer",
        include_top=False,
    )

    # If a model was created with TF1, it will not load by name into a TF2
    # model because TF2 changing the naming convention.
    # here, we call a function to change the names of the layers in the model
    # to match what's contained in the weights file
    model = renameLayers(model, weightspath)

    # Lock desired number of layers
    for layer in model.layers[:num_layers_locked]:
        layer.trainable = False

        # Do forward pass all the way until end
    input_ = Input((None, None, None, input_dim * num_cams))

    old_out = model(input_)

    # Add new output conv. layer
    if "gaussian_cross_entropy_loss" in str(lossfunc):
        new_conv = Conv3D(
        new_n_channels_out, new_last_kern_size, activation="linear", padding="same"
        )(old_out)
    else:
        new_conv = Conv3D(
        new_n_channels_out, new_last_kern_size, activation="sigmoid", padding="same"
        )(old_out)

    model = Model(inputs=[input_], outputs=[new_conv])

    return model
