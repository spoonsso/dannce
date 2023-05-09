import tensorflow.keras as keras
from tensorflow.keras import backend as K
import os
import scipy.io as sio
import numpy as np
import dannce.engine.processing as processing
from dannce.engine import losses
import gc

class savePredTargets(keras.callbacks.Callback):
    def __init__(
        self, total_epochs, td, tgrid, vd, vgrid, tID, vID, odir, tlabel, vlabel
    ):
        self.td = td
        self.vd = vd
        self.tID = tID
        self.vID = vID
        self.total_epochs = total_epochs
        self.val_loss = 1e10
        self.odir = odir
        self.tgrid = tgrid
        self.vgrid = vgrid
        self.tlabel = tlabel
        self.vlabel = vlabel

    def on_epoch_end(self, epoch, logs=None):
        lkey = "val_loss" if "val_loss" in logs else "loss"
        if (
            epoch == self.total_epochs - 1
            or logs[lkey] < self.val_loss
        ):
            print(
                "Saving predictions on train and validation data, after epoch {}".format(
                    epoch
                )
            )
            self.val_loss = logs[lkey]
            pred_t = self.model.predict([self.td, self.tgrid], batch_size=1)
            pred_v = self.model.predict([self.vd, self.vgrid], batch_size=1)
            ofile = os.path.join(
                self.odir, "checkpoint_predictions_e{}.mat".format(epoch)
            )
            sio.savemat(
                ofile,
                {
                    "pred_train": pred_t,
                    "pred_valid": pred_v,
                    "target_train": self.tlabel,
                    "target_valid": self.vlabel,
                    "train_sampleIDs": self.tID,
                    "valid_sampleIDs": self.vID,
                },
            )

class saveMaxPreds(keras.callbacks.Callback):
    """
    This callback fully evaluates MAX predictions and logs the euclidean
        distance error to a file.
    """

    def __init__(self, vID, vData, vLabel, odir, com, params):
        self.vID = vID
        self.vData = vData
        self.odir = odir
        self.com = com
        self.param_mat = params
        self.total_epochs = params["epochs"]

        fn = os.path.join(odir, "max_euclid_error.csv")
        self.fn = fn

        self.vLabel = np.zeros((len(vID), 3, params["new_n_channels_out"]))

        # Now run thru sample IDs, pull out the correct COM, and add it in
        for j in range(len(self.vID)):
            id_ = self.vID[j]
            self.vLabel[j] = vLabel[id_]

        with open(fn, "w") as fd:
            fd.write("epoch,error\n")

    def on_epoch_end(self, epoch, logs=None):
        pred_v = self.model.predict([self.vData], batch_size=1)
        d_coords = np.zeros((pred_v.shape[0], 3, pred_v.shape[-1]))
        for j in range(pred_v.shape[0]):
            xcoord, ycoord, zcoord = processing.plot_markers_3d(pred_v[j])
            d_coords[j] = np.stack([xcoord, ycoord, zcoord])

        vsize = (self.param_mat["vmax"] - self.param_mat["vmin"]) / self.param_mat[
            "nvox"
        ]
        # # First, need to move coordinates over to centers of voxels
        pred_out_world = self.param_mat["vmin"] + d_coords * vsize + vsize / 2

        # Now run thru sample IDs, pull out the correct COM, and add it in
        for j in range(len(self.vID)):
            id_ = self.vID[j]
            tcom = self.com[id_]
            pred_out_world[j] = pred_out_world[j] + tcom[:, np.newaxis]

        # Calculate euclidean_distance_3d
        e3d = K.eval(losses.euclidean_distance_3D(self.vLabel, pred_out_world))

        print("epoch {} euclidean_distance_3d: {}".format(epoch, e3d))
        with open(self.fn, "a") as fd:
            fd.write("{},{}\n".format(epoch, e3d))

class saveCheckPoint(keras.callbacks.Callback):
    def __init__(self, odir, total_epochs):
        self.odir = odir
        self.saveE = np.arange(0, total_epochs, 250)

    def on_epoch_end(self, epoch, logs=None):
        lkey = "val_loss" if "val_loss" in logs else "loss"
        val_loss = logs[lkey]
        if epoch in self.saveE:
            # Do a garbage collect to combat keras memory leak
            gc.collect()
            print("Saving checkpoint weights at epoch {}".format(epoch))
            savename = "weights.checkpoint.epoch{}.{}{:.5f}.hdf5".format(
                epoch, lkey, val_loss
            )
            self.model.save(os.path.join(self.odir, savename))