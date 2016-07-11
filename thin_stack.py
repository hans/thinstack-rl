
import random

import numpy as np
import tensorflow as tf

import util


# HACK: Set up gradient of the ScatterUpdate op. In this script ScatterUpdate
# is only used non-destructively.
@tf.RegisterGradient("ScatterUpdate")
def _scatter_update_grad(op, grad):
    idx = op.inputs[1]
    return None, None, tf.gather(grad, idxs)


class ThinStack(object):

    def __init__(self, compose_fn, tracking_fn, transition_fn, batch_size,
                 vocab_size, num_timesteps, model_dim, embedding_dim,
                 tracking_dim, embeddings=None, embedding_initializer=None,
                 scope=None):
        assert num_timesteps % 2 == 1, "Require odd number of timesteps for binary SR parser"

        self.compose_fn = compose_fn
        self.tracking_fn = tracking_fn

        self.transition_fn = transition_fn

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.stack_size = num_timesteps # HACK: Not true
        self.buff_size = (num_timesteps + 1) / 2
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.tracking_dim = tracking_dim

        # Helpers
        self.batch_range = tf.range(self.batch_size)

        with tf.variable_scope(scope or "ts") as scope:
            self._scope = scope

            self._create_params(embeddings, embedding_initializer)
            self._create_placeholders()
            self._create_state()

            # Run the forward op to compute a final self.stack representation
            self.forward()

            # Look up stack tops based on per-example length input
            top_idxs = (self.num_transitions - 1) * self.batch_size + self.batch_range
            self.final_representations = tf.gather(self.stack, top_idxs)

            # Reshape the stack into a more intuitive 3D for indexing.
            self.indexable_stack = tf.reshape(self.stack, (self.stack_size, self.batch_size, self.model_dim))

    def _create_params(self, embeddings, embedding_initializer):
        embedding_shape = (self.vocab_size, self.embedding_dim)
        if embeddings is None:
            embeddings = tf.get_variable("embeddings", embedding_shape,
                    initializer=embedding_initializer)
        else:
            shape = embeddings.get_shape()
            shape.assert_same_rank(embedding_shape)
            assert shape.as_list() == embedding_shape, \
                    "Provided embeddings must be of shape %s" % embedding_shape

        self.embeddings = embeddings

    def _create_placeholders(self):
        # int embedding index batch, buff_size * batch_size
        self.buff = tf.placeholder(tf.int32, (self.buff_size, self.batch_size),
                                     name="buff")
        # list of num_timesteps-many (batch_size) int batches
        # Used for loss computation only.
        self.transitions = [tf.placeholder(tf.int32, (self.batch_size,), name="transitions_%i" % t)
                            for t in range(self.num_timesteps)]

        self.num_transitions = tf.placeholder(tf.int32, (self.batch_size,), name="num_transitions")

    def _create_state(self):
        """Prepare stateful variables modified during the recurrence."""

        # Both the queue and the stack are flattened stack_size * batch_size
        # tensors. `stack_size` many blocks of `batch_size` values
        self.stack = tf.Variable(tf.zeros((self.stack_size * self.batch_size, self.model_dim), dtype=tf.float32),
                                 trainable=False, name="stack")
        self.queue = tf.Variable(tf.zeros((self.stack_size * self.batch_size,), dtype=tf.int32),
                                 trainable=False, name="queue")

        self.buff_cursors = tf.Variable(tf.zeros((self.batch_size,), dtype=tf.int32),
                                          trainable=False, name="buff_cursors")
        self.cursors = tf.Variable(tf.ones((self.batch_size,), dtype=tf.int32) * - 1,
                                   trainable=False, name="cursors")

        # TODO make parameterizable
        self.tracking_value = tf.Variable(tf.zeros((self.batch_size, self.tracking_dim), dtype=tf.float32),
                                          trainable=False, name="tracking_value")

        # Create an Op which will (re-)initialize the auxiliary variables
        # declared above.
        aux_vars = [self.stack, self.queue, self.buff_cursors, self.cursors,
                    self.tracking_value]
        self.variable_initializer = tf.initialize_variables(aux_vars)

    def _update_stack(self, t, shift_value, reduce_value, transitions_t):
        mask = tf.to_float(tf.expand_dims(transitions_t, 1))
        top_next = mask * reduce_value + (1 - mask) * shift_value

        stack_idxs = t * self.batch_size + self.batch_range
        stack_next = tf.scatter_update(self.stack, stack_idxs, top_next)

        cursors_next = self.cursors + (transitions_t * -1 + (1 - transitions_t) * 1)

        queue_idxs = cursors_next * self.batch_size + self.batch_range
        # TODO: enforce transition validity instead of this hack
        queue_idxs = tf.maximum(queue_idxs, 0)
        queue_next = tf.scatter_update(self.queue, queue_idxs, tf.fill((self.batch_size,), t))

        return stack_next, queue_next, cursors_next

    def _lookup(self, t):
        stack1_ptrs = (t - 1) * self.batch_size + self.batch_range
        stack1 = tf.gather(self.stack, tf.maximum(0, stack1_ptrs))

        queue_ptrs = (self.cursors - 1) * self.batch_size + self.batch_range
        stack2_ptrs = tf.to_int32(tf.gather(self.queue, tf.maximum(0, queue_ptrs))) * self.batch_size + self.batch_range
        stack2 = tf.gather(self.stack, stack2_ptrs)

        buff_idxs = (self.buff_cursors * self.batch_size) + self.batch_range
        # TODO: enforce transition validity instead of this hack
        buff_idxs = tf.maximum(0, tf.minimum(buff_idxs, (self.buff_size * self.batch_size) - 1))
        buff_top = tf.gather(self.buff_embeddings, buff_idxs)
        return stack1, stack2, buff_top

    def _step(self, t, transitions_t):
        stack1, stack2, buff_top = self._lookup(t)
        # stack1 = tf.Print(stack1, [stack1, t])

        # Compute new recurrent and recursive values.
        tracking_value_ = self.tracking_fn([self.tracking_value, stack1, stack2, buff_top])
        reduce_value = self.compose_fn(stack1, stack2, tracking_value_)

        if self.transition_fn is not None:
            p_transitions_t = self.transition_fn([tracking_value_, stack1, stack2, buff_top])
            sample_t = tf.multinomial(p_transitions_t, 1)

            must_shift = tf.to_int32(self.cursors < 1)
            must_reduce = tf.to_int32(self.buff_cursors >= (self.num_transitions + 1) / 2)
            sample_mask = 1 - must_reduce - must_shift

            transitions_t = tf.to_int32(tf.squeeze(sample_t)) * sample_mask + must_reduce
        else:
            p_transitions_t = None

        transitions_t = tf.Print(transitions_t, [transitions_t, t])

        stack_, queue_, cursors_ = \
                self._update_stack(t, buff_top, reduce_value, transitions_t)
        buff_cursors_ = self.buff_cursors + 1 - transitions_t

        return stack_, queue_, cursors_, buff_cursors_, tracking_value_, \
                p_transitions_t, transitions_t

    def forward(self):
        # Look up word embeddings and flatten for easy indexing with gather
        self.buff_embeddings = tf.nn.embedding_lookup(self.embeddings, self.buff)
        self.buff_embeddings = tf.reshape(self.buff_embeddings, (-1, self.model_dim))
        # TODO: embedding projection / dropout / BN / etc.

        # Storage for p(transition_t) and sampled transition information
        self.p_transitions = [None] * self.num_timesteps
        self.sampled_transitions = [None] * self.num_timesteps

        # TODO: deep! just rerun with a new compose fn and use multiple stacks,
        # reading from previous and writing to next, with fixed transitions on
        # higher layers
        for t, transitions_t in enumerate(self.transitions):
            if t > 0:
                self._scope.reuse_variables()

            with tf.control_dependencies([self.stack, self.queue]):
                ret = self._step(t, transitions_t)

            self.stack, self.queue, self.cursors, self.buff_cursors = ret[:4]
            self.tracking_value, self.p_transitions[t], self.sampled_transitions[t] = ret[4:]


    def reset(self, session):
        session.run(self.variable_initializer)


def main():
    s = tf.Session()

    batch_size = 3
    num_timesteps = 9
    buff_size = (num_timesteps + 1) / 2
    embedding_dim = 7
    model_dim = 7
    tracking_dim = 2
    vocab_size = 5

    def integer_embedding_initializer(*args, **kwargs):
        return [[i for j in range(embedding_dim)] for i in range(vocab_size)]

    with tf.variable_scope("m", initializer=util.HeKaimingInitializer()):
        compose_fn = lambda x, y, h: x + y
        tracking_fn = lambda *xs: xs[0]
        def transition_fn(*xs):
            return [[-10., -10.] for i in range(3)]

        ts = ThinStack(compose_fn, tracking_fn, transition_fn, batch_size,
                       vocab_size, num_timesteps, model_dim, embedding_dim,
                       tracking_dim, embedding_initializer=integer_embedding_initializer)

    data = [{'label': 10,
          'len': 5,
          'sentence': '( 2 ( 4 4 ) )',
          'tokens': [2, 4, 4, 0, 0],
          'transitions': [0, 0, 0, 1, 1, 0, 0, 0, 0]},
        {'label': 10,
          'len': 5,
          'sentence': '( ( 4 4 ) 2 )',
          'tokens': [4, 4, 2, 0, 0],
          'transitions': [0, 0, 1, 0, 1, 0, 0, 0, 0]},
        {'label': 10,
          'len': 9,
          'sentence': '( 1 ( 2 ( 3 ( 3 1 ) ) ) )',
          'tokens': [1, 2, 3, 3, 1],
          'transitions': [0, 0, 0, 0, 0, 1, 1, 1, 1]}]

    X, transitions, y, lengths = util.data.BucketToArrays(data, 9)
    buff = np.concatenate([xt[:, np.newaxis] for xt in X], axis=1)

    s.run(tf.initialize_variables(tf.trainable_variables()))
    ts.reset(s)

    feed = {ts.transitions[t]: transitions[:, t] for t in range(num_timesteps)}
    feed[ts.buff] = buff
    feed[ts.num_transitions] = lengths
    print s.run([ts.indexable_stack, ts.final_representations], feed)


if __name__ == '__main__':
    main()
