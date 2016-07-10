import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib.layers.python import layers

def Linear(args, output_dim, bias=True, bias_init=0.0, scope=None):
    if not isinstance(args, (list, tuple)):
        args = [args]

    input_dim = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2d arguments: %s" % str(shapes))
        elif not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            input_dim += shape[1]

    with tf.variable_scope(scope or "linear"):
        W = tf.get_variable("W", (input_dim, output_dim))

        if len(args) == 1:
            result = tf.matmul(args[0], W)
        else:
            result = tf.matmul(tf.concat(1, args), W)

        if not bias:
            return result

        b = tf.get_variable("b", (output_dim,),
                            initializer=tf.constant_initializer(bias_init))

    return result + b

def HeKaimingInitializer(seed=None, dtype=dtypes.float32):
    # This is the default behavior: 
    return layers.initializers.variance_scaling_initializer(seed=seed, dtype=dtype)
