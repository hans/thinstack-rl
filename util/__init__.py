from util.blocks import *
from util.data import *


# Load custom TF module.
import os
import tensorflow as tf
floaty_path = os.path.join(os.path.dirname(__file__), "..", "lib", "floaty_ops.so")
try:
    floaty = tf.load_op_library(floaty_path)

    # HACK: Set up gradient of the ScatterUpdate op. In this script ScatterUpdate
    # is only used non-destructively.
    @tf.RegisterGradient("FloatyScatterUpdate")
    def _floaty_scatter_update_grad(op, grad):
        idxs = op.inputs[1]
        return None, None, floaty.floaty_gather(grad, idxs)

except:
    print "Warning: Compiled floaty library could not be loaded. Falling back to slow floaty ops."

    class floaty:
        @staticmethod
        def floaty_scatter_update(ref, indices, updates, **kwargs):
            return tf.scatter_update(ref, tf.to_int32(indices), updates, **kwargs)

        @staticmethod
        def floaty_gather(arg, idxs, **kwargs):
            return tf.gather(arg, tf.to_int32(idxs), **kwargs)

    # HACK: Set up gradient of the ScatterUpdate op. In this script ScatterUpdate
    # is only used non-destructively.
    @tf.RegisterGradient("ScatterUpdate")
    def _scatter_update_grad(op, grad):
        idxs = op.inputs[1]
        return None, None, tf.gather(grad, idxs)
