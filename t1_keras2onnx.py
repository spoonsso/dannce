# python t1_keras2onnx.py ./models/dannce_model.h5 | xargs -i python t2_onnx2trt.py "{}"
import argparse
from tensorflow.keras.models import load_model
import tensorflow as tf
import onnx
import tf2onnx
import sys
import os
import os.path as osp
modelfile = '/home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x10_mono/DANNCE/train_results/MAX/fullmodel_weights/fullmodel_end.hdf5'

batchsize = None  #[None = Dynamic, 1|2|... = Fix shape]
def main(modelfile):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #use cpu to load model
    model = load_model(modelfile, compile=False)

    nchannel_fix = model.input.shape[-1]
    input_signature = [tf.TensorSpec([batchsize,64,64,64,nchannel_fix], model.input.dtype, name='input_1')]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
    onnx_file = osp.splitext(modelfile)[0] + ('.onnx' if batchsize in (None, 1) else f'_batch{batchsize}.onnx')
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_file)
    print('Saved model to: {}'.format(onnx_file), file=sys.stderr)
    print(onnx_file)
    return onnx_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile", type=str,
                        help="Path to the model file.")
    args = parser.parse_args()
    main(args.modelfile)
