import tensorflow as tf

from util.floaty_ops import _library


floaty_gather = _library.floaty_gather
floaty_scatter_update = _library.floaty_scatter_update


@tf.RegisterShape("FloatyGather")
def _floaty_gather_shape(op):
    params_shape = op.inputs[0].get_shape()
    indices_shape = op.inputs[1].get_shape()
    return [indices_shape.concatenate(params_shape[1:])]


@tf.RegisterGradient("FloatyGather")
def _floaty_gather_grad(op, grad):
    if op.inputs[0].get_shape().is_fully_defined():
        dense_shape = tf.constant(op.inputs[0].get_shape().as_list())
        values_shape = [-1] + op.inputs[0].get_shape()[1:].as_list()
    else:
        # op.inputs[0] can be large, so colocate the shape calculation with it.
        with ops.colocate_with(op.inputs[0]):
            dense_shape = tf.shape(op.inputs[0])
            values_shape = tf.concat(0, [[-1], dense_shape[1:]])

    values = tf.reshape(grad, values_shape)
    indices = tf.to_int32(tf.reshape(op.inputs[1], [-1]))
    return [tf.IndexedSlices(values, indices, dense_shape), None]



@tf.RegisterShape("FloatyScatterUpdate")
def _floaty_scatter_update_shape(op):
    var_shape = op.inputs[0].get_shape()
    indices_shape = op.inputs[1].get_shape()
    unused_updates_shape = op.inputs[2].get_shape().merge_with(
            indices_shape.concatenate(var_shape[1:]))
    return [var_shape]


@tf.RegisterGradient("FloatyScatterUpdate")
def _floaty_scatter_update_grad(op, grad):
    idxs = op.inputs[1]
    with tf.device("/gpu:0"):
        return None, None, floaty_gather(grad, idxs)
