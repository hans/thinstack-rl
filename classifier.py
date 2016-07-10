import sys

import gflags
import tensorflow as tf

from thin_stack import ThinStack
import util

FLAGS = gflags.FLAGS


def build_thin_stack_classifier(ts, num_classes, mlp_dims=(256,),
                                scope=None):
    with tf.variable_scope(scope or "classifier"):
        dims = (ts.model_dim,) + mlp_dims
        x = ts.final_representations
        for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            with tf.variable_scope("mlp%i" % i):
                x = util.Linear(x, out_dim, bias=True)
                x = tf.tanh(x)

        with tf.variable_scope("logits"):
            logits = util.Linear(x, num_classes, bias=True)

    return logits


def build_model():
    with tf.variable_scope("m", initializer=util.HeKaimingInitializer()):
        ys = tf.placeholder(tf.int32, (FLAGS.batch_size,), "ys")

        tracking_fn = lambda *xs: xs[0]
        compose_fn = lambda x, y, h: util.Linear([x, y, h], FLAGS.model_dim)
        def transition_fn(*xs):
            """Return random logits."""
            return tf.random_uniform((batch_size, 2), minval=-10, maxval=10)

        ts = ThinStack(compose_fn, tracking_fn, transition_fn, FLAGS.batch_size, FLAGS.vocab_size,
                       FLAGS.seq_length, FLAGS.model_dim, FLAGS.embedding_dim,
                       FLAGS.tracking_dim)

        logits = build_thin_stack_classifier(ts, FLAGS.num_classes)

        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys)

    return ts, logits, ys, xent_loss


def main():
    ts, logits, ys, xent_loss = build_model()

    sess = tf.Session()
    sess.run(tf.initialize_variables(tf.trainable_variables()))
    ts.reset(sess)

    # TODO generate data and train on xent loss


if __name__ == '__main__':
    gflags.DEFINE_integer("batch_size", 64, "")
    gflags.DEFINE_integer("vocab_size", 100, "")
    gflags.DEFINE_integer("seq_length", 29, "")
    gflags.DEFINE_integer("num_classes", 3, "")

    gflags.DEFINE_integer("model_dim", 128, "")
    gflags.DEFINE_integer("embedding_dim", 64, "")
    gflags.DEFINE_integer("tracking_dim", 32, "")

    FLAGS(sys.argv)
    main()
