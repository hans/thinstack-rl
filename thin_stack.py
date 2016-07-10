
import random

import numpy as np
import tensorflow as tf


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
        self.num_transitions = 2

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.stack_size = num_timesteps # HACK: Not true
        self.buffer_size = (num_timesteps + 1) / 2
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

            # Convenience member pointing to top of stack at final timestep
            # TODO: Handle variable sequence lengths
            final_representation_idx = self.batch_size * self.num_timesteps - self.batch_size
            self.final_representations = self.stack[final_representation_idx:, :]

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
        # int embedding index batch, buffer_size * batch_size
        self.buffer = tf.placeholder(tf.int32, (self.buffer_size, self.batch_size),
                                     name="buffer")
        # list of num_timesteps-many (batch_size) int batches
        self.transitions = [tf.placeholder(tf.int32, (self.batch_size,), name="transitions_%i" % t)
                            for t in range(self.num_timesteps)]

    def _create_state(self):
        """Prepare stateful variables modified during the recurrence."""

        # Both the queue and the stack are flattened stack_size * batch_size
        # tensors. `stack_size` many blocks of `batch_size` values
        self.stack = tf.Variable(tf.zeros((self.stack_size * self.batch_size, self.model_dim), dtype=tf.float32),
                                 trainable=False, name="stack")
        self.queue = tf.Variable(tf.zeros((self.stack_size * self.batch_size,), dtype=tf.int32),
                                 trainable=False, name="queue")

        self.buffer_cursors = tf.Variable(tf.zeros((self.batch_size,), dtype=tf.int32),
                                          trainable=False, name="buffer_cursors")
        self.cursors = tf.Variable(tf.ones((self.batch_size,), dtype=tf.int32) * - 1,
                                   trainable=False, name="cursors")

        # TODO make parameterizable
        self.tracking_value = tf.Variable(tf.zeros((self.batch_size, self.tracking_dim), dtype=tf.float32),
                                          trainable=False, name="tracking_value")

        # Create an Op which will (re-)initialize the auxiliary variables
        # declared above.
        aux_vars = [self.stack, self.queue, self.buffer_cursors, self.cursors,
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

        buffer_idxs = self.buffer_cursors * self.buffer_size + self.batch_range
        # TODO: enforce transition validity instead of this hack
        buffer_idxs = tf.minimum(0, tf.maximum(buffer_idxs, self.buffer_size * self.buffer_size + self.batch_range))
        buffer_top = tf.gather(self.buffer_embeddings, buffer_idxs)

        return stack1, stack2, buffer_top

    def _step(self, t, transitions_t):
        stack1, stack2, buffer_top = self._lookup(t)

        # Compute new recurrent and recursive values.
        tracking_value_ = self.tracking_fn([self.tracking_value, stack1, stack2, buffer_top])
        reduce_value = self.compose_fn([stack1, stack2, tracking_value_])

        if self.transition_fn is not None:
            p_transitions_t = self.transition_fn([tracking_value_, stack1, stack2, buffer_top])
            transitions_t = tf.to_int32(tf.squeeze(tf.multinomial(p_transitions_t, 1)))
            transitions_t = tf.Print(transitions_t, [transitions_t])
        else:
            p_transitions_t = None

        stack_, queue_, cursors_ = \
                self._update_stack(t, buffer_top, reduce_value, transitions_t)
        buffer_cursors_ = self.buffer_cursors + 1 - transitions_t

        return stack_, queue_, cursors_, buffer_cursors_, tracking_value_, \
                p_transitions_t, transitions_t

    def forward(self):
        # Look up word embeddings and flatten for easy indexing with gather
        self.buffer_embeddings = tf.nn.embedding_lookup(self.embeddings, self.buffer)
        self.buffer_embeddings = tf.reshape(self.buffer_embeddings, (-1, self.model_dim))
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

            ret = self._step(t, transitions_t)

            self.stack, self.queue, self.cursors, self.buffer_cursors = ret[:4]
            self.tracking_value, self.p_transitions[t], self.sampled_transitions[t] = ret[4:]

        return self.stack

    def reset(self, session):
        session.run(self.variable_initializer)


def main():
    s = tf.Session()

    batch_size = 3
    num_timesteps = 3
    buffer_size = (num_timesteps + 1) / 2
    embedding_dim = 7
    model_dim = 7
    tracking_dim = 2
    vocab_size = 10

    compose_fn = lambda (x, y, h): x + y
    tracking_fn = lambda *xs: xs[0]
    def transition_fn(*xs):
        """Return random logits."""
        return tf.random_uniform((batch_size, 2), minval=-10, maxval=10)

    ts = ThinStack(compose_fn, tracking_fn, transition_fn, batch_size,
                   vocab_size, num_timesteps, model_dim, embedding_dim,
                   tracking_dim)

    X = [np.ones((batch_size,)) * random.randint(0, vocab_size - 1)
         for t in range(buffer_size)]
    buffer = np.concatenate([xt[np.newaxis, :] for xt in X])
    transitions = [np.zeros((batch_size,), np.int32), np.zeros((batch_size,), np.int32),
                   np.ones((batch_size,), np.int32)]

    s.run(tf.initialize_variables(tf.trainable_variables()))
    ts.reset(s)

    feed = {ts.transitions[t]: transitions_t for t, transitions_t in enumerate(transitions)}
    feed[ts.buffer] = buffer
    print s.run(ts.stack, feed)


if __name__ == '__main__':
    main()
