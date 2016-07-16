import numpy as np
import tensorflow as tf
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


def HeKaimingInitializer(seed=None, dtype=tf.float32):
    # This is the default behavior:
    return layers.initializers.variance_scaling_initializer(seed=seed, dtype=dtype)


def TreeLSTMBiasInitializer():
    def init(shape, dtype):
        hidden_dim = shape[0] / 5
        value = np.zeros(shape, dtype=dtype.as_numpy_dtype())
        value[hidden_dim:3*hidden_dim] = 1
        return value
    return init


def LSTMBiasInitializer():
    def init(shape, dtype):
        hidden_dim = shape[0] / 4
        value = np.zeros(shape, dtype=dtype.as_numpy_dtype())
        value[hidden_dim:2*hidden_dim] = 1
        return value
    return init


def ReLULayer(inp, inp_dim, outp_dim, vs, name="relu_layer", use_bias=True, initializer=None):
    return tf.nn.relu(Linear(inp, inp_dim, outp_dim, vs, name, use_bias, initializer))


def LSTMLayer(lstm_prev, input_t, scope=None):
    full_memory_dim = lstm_prev.get_shape().as_list()[1]
    assert full_memory_dim % 2 == 0, \
            "LSTM memories are concatenated (h,c); full dim must be even (found %i)" \
            % full_memory_dim
    hidden_dim = full_memory_dim / 2

    def slice_gate(gate_data, i):
        return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

    with tf.variable_scope(scope or "lstm", initializer=HeKaimingInitializer()):
        b = tf.get_variable("b", (hidden_dim * 4,), initializer=LSTMBiasInitializer())

        # Decompose previous LSTM value into hidden and cell value
        h_prev = lstm_prev[:, :hidden_dim]
        c_prev = lstm_prev[:,  hidden_dim:]

        gates_dim = hidden_dim * 4
        gates = Linear(h_prev, gates_dim, bias=False, scope="hid_linear")
        gates += Linear(input_t, gates_dim, bias=False, scope="inp_linear")
        gates += b

        # Compute and slice gate values
        i_gate, f_gate, o_gate, cell_inp = [slice_gate(gates, i) for i in range(4)]

        # Apply nonlinearities
        i_gate = tf.nn.sigmoid(i_gate)
        f_gate = tf.nn.sigmoid(f_gate)
        o_gate = tf.nn.sigmoid(o_gate)
        cell_inp = tf.nn.tanh(cell_inp)

        # Compute new cell and hidden value
        c_t = f_gate * c_prev + i_gate * cell_inp
        h_t = o_gate * tf.nn.tanh(c_t)

        return tf.concat(1, [h_t, c_t])


def TreeLSTMLayer(lstm_prev, external_state, scope=None):
    assert isinstance(lstm_prev, tuple)
    l_prev, r_prev = lstm_prev

    assert l_prev.get_shape() == r_prev.get_shape(), "Child memories must be of same dimension"
    full_memory_dim = l_prev.get_shape().as_list()[1]
    assert full_memory_dim % 2 == 0, \
            "LSTM memories are concatenated (h,c); full dim must be even (found %i)" \
            % full_memory_dim
    hidden_dim = full_memory_dim / 2

    with tf.variable_scope(scope or "tree_lstm", initializer=HeKaimingInitializer()):
        W_l = tf.get_variable("W_l", (hidden_dim, hidden_dim * 5))
        W_r = tf.get_variable("W_r", (hidden_dim, hidden_dim * 5))
        if external_state is not None:
            W_ext = tf.get_variable("W_ext", (external_state.get_shape()[1], hidden_dim * 5))
        b = tf.get_variable("b", (hidden_dim * 5,), initializer=TreeLSTMBiasInitializer())

    def slice_gate(gate_data, i):
        return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

    # Decompose previous LSTM value into hidden and cell value
    l_h_prev = l_prev[:, :hidden_dim]
    l_c_prev = l_prev[:,  hidden_dim:]
    r_h_prev = r_prev[:, :hidden_dim]
    r_c_prev = r_prev[:,  hidden_dim:]

    gates = tf.matmul(l_h_prev, W_l) + tf.matmul(r_h_prev, W_r) + b
    if external_state is not None:
        gates += tf.matmul(external_state, W_ext)

    # Compute and slice gate values
    i_gate, fl_gate, fr_gate, o_gate, cell_inp = [slice_gate(gates, i) for i in range(5)]

    # Apply nonlinearities
    i_gate = tf.nn.sigmoid(i_gate)
    fl_gate = tf.nn.sigmoid(fl_gate)
    fr_gate = tf.nn.sigmoid(fr_gate)
    o_gate = tf.nn.sigmoid(o_gate)
    cell_inp = tf.nn.tanh(cell_inp)

    # Compute new cell and hidden value
    c_t = fl_gate * l_c_prev + fr_gate * r_c_prev + i_gate * cell_inp
    h_t = o_gate * tf.nn.tanh(c_t)

    return tf.concat(1, [h_t, c_t])




