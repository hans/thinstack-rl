from util import floaty_ops
from util.blocks import *
from util.data import *


# HACK: Set up gradient of the ScatterUpdate op. In this script ScatterUpdate
# is only used non-destructively.
@tf.RegisterGradient("ScatterUpdate")
def _scatter_update_grad(op, grad):
    idxs = op.inputs[1]
    return None, None, tf.gather(grad, idxs)
