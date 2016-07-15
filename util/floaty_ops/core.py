import tensorflow as tf

from util.floaty_ops import _library


floaty_gather = _library.floaty_gather
floaty_scatter_update = _library.floaty_scatter_update
unsafe_floaty_gather = _library.unsafe_floaty_gather


@tf.RegisterShape("FloatyGather")
def _floaty_gather_shape(op):
    params_shape = op.inputs[0].get_shape()
    indices_shape = op.inputs[1].get_shape()
    return [indices_shape.concatenate(params_shape[1:])]


@tf.RegisterGradient("FloatyGather")
def _floaty_gather_grad(op, grad):
    # Easier if we just assume this. Look back at Gather grad if you need
    # dynamic shaping support
    assert op.inputs[0].get_shape().is_fully_defined()

    dense_shape = tf.constant(op.inputs[0].get_shape().as_list())
    values_shape = [-1] + op.inputs[0].get_shape()[1:].as_list()

    values = tf.reshape(grad, values_shape)
    indices = tf.to_int32(tf.reshape(op.inputs[1], [-1]))
    return [tf.IndexedSlices(values, indices, dense_shape), None]


@tf.RegisterShape("UnsafeFloatyGather")
def _unsafe_floaty_gather_shape(op):
    params_shape = op.inputs[0].get_shape()
    indices_shape = op.inputs[1].get_shape()
    return [indices_shape.concatenate(params_shape[1:])]


@tf.RegisterGradient("UnsafeFloatyGather")
def _unsafe_floaty_gather_grad(op, grad):
    params, idxs, grad_container = op.inputs
    assert params.get_shape().is_fully_defined()

    with tf.op_scope([grad_container, idxs, grad], None, "UnsafeFloatyGatherGrad"):
        update_container = floaty_scatter_update(grad_container, idxs, grad)
        with tf.control_dependencies([update_container]):
            return floaty_gather(update_container, idxs)



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
    grad_updates = floaty_gather(grad, idxs)
    return None, None, grad_updates
