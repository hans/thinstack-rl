import unittest

import tensorflow as tf
import numpy as np

from thin_stack import ThinStack


class ThinStackTestCase(tf.test.TestCase):

    """Basic functional tests for ThinStack with dummy data."""

    def _make_stack(self, batch_size=2, seq_length=5):
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim = 3
        self.vocab_size = vocab_size = 10
        self.seq_length = seq_length

        compose_fn = lambda (x, y), *ext: x + y
        tracking_fn = lambda *xs: xs[0]
        transition_fn = None

        # Swap in our own dummy embeddings and weights.
        initial_embeddings = np.arange(vocab_size).reshape(
            (vocab_size, 1)).repeat(embedding_dim, axis=1).astype(np.float32)
        initial_embeddings = tf.Variable(initial_embeddings, name="embeddings")

        self.stack = ThinStack(compose_fn, tracking_fn, transition_fn, batch_size,
                               vocab_size, seq_length, embedding_dim, embedding_dim,
                               10, embeddings=initial_embeddings)

    def test_basic_ff(self):
        self._make_stack(seq_length=5)

        X = np.array([
            [3, 1,  2],
            [3, 2,  4]
        ], dtype=np.int32).T

        transitions = np.array([
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 1]
        ], dtype=np.float32)

        num_transitions = np.array([4, 4], dtype=np.int32)

        expected = np.array([[ 3.,  3.,  3.],
                             [ 3.,  3.,  3.],
                             [ 1.,  1.,  1.],
                             [ 2.,  2.,  2.],
                             [ 2.,  2.,  2.],
                             [ 5.,  5.,  5.],
                             [ 3.,  3.,  3.],
                             [ 4.,  4.,  4.],
                             [ 6.,  6.,  6.],
                             [ 9.,  9.,  9.]])

        # Run twice to make sure first state is properly erased
        with self.test_session() as s:
            s.run(tf.initialize_variables(tf.trainable_variables()))
            ts = self.stack

            feed = {ts.transitions[t]: transitions[:, t]
                    for t in range(self.seq_length)}
            feed[ts.buff] = X
            feed[ts.num_transitions] = num_transitions

            for _ in range(2):
                ts.reset(s)

                ret = s.run(ts.stack, feed)
                np.testing.assert_almost_equal(ret, expected)


if __name__ == '__main__':
    tf.test.main()
