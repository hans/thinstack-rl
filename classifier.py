import sys

import gflags
import tensorflow as tf

from data.arithmetic import load_simple_data as load_arithmetic_data
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


def build_model(num_timesteps):
    with tf.variable_scope("m", initializer=util.HeKaimingInitializer()):
        ys = tf.placeholder(tf.int32, (FLAGS.batch_size,), "ys")

        tracking_fn = lambda *xs: xs[0]
        compose_fn = util.TreeLSTMLayer
        def transition_fn(*xs):
            """Return random logits."""
            return tf.random_uniform((FLAGS.batch_size, 2), minval=-10, maxval=10)

        ts = ThinStack(compose_fn, tracking_fn, transition_fn, FLAGS.batch_size, FLAGS.vocab_size,
                       num_timesteps, FLAGS.model_dim, FLAGS.embedding_dim,
                       FLAGS.tracking_dim)

        logits = build_thin_stack_classifier(ts, FLAGS.num_classes)

        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys)

    return ts, logits, ys, xent_loss


def prepare_data():
    if FLAGS.data_type == "arithmetic":
        data_manager = load_arithmetic_data

    sentence_pair_data = data_manager.SENTENCE_PAIR_DATA

    raw_data, vocabulary = data_manager.load_data(FLAGS.training_data_path)
    data = util.data.TokensToIDs(vocabulary, raw_data,
                                 sentence_pair_data=sentence_pair_data)

    # TODO customizable
    buckets = [9, 21]
    bucketed_data = util.data.PadAndBucket(data, buckets,
                                           sentence_pair_data=sentence_pair_data)

    # Convert each bucket into TF-friendly arrays
    bucketed_data = {length: util.data.BucketToArrays(bucket, length,
                                                      sentence_pair_data=sentence_pair_data)
                     for length, bucket in bucketed_data.iteritems()}

    iterator = util.data.MakeBucketedTrainingIterator(bucketed_data, FLAGS.batch_size)

    return iterator, buckets, vocabulary


def build_training_graphs(buckets):
    graphs = {}
    opt = tf.train.MomentumOptimizer(0.001, 0.9)

    for i, num_timesteps in enumerate(buckets):
        with tf.variable_scope("train/", reuse=i > 0):
            ts, logits, ys, xent_loss = build_model(num_timesteps)
            train_op = opt.minimize(xent_loss)

            graphs[num_timesteps] = (ts, logits, ys, xent_loss, train_op)

    return graphs


def main():
    training_iterator, training_buckets, vocabulary = prepare_data()
    training_graphs = build_training_graphs(training_buckets)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for bucket, (X_batch, transitions_batch, y_batch, num_transitions_batch) in training_iterator:
        ts, logits, ys, xent_loss, train_op = training_graphs[bucket]
        ts.reset(sess)

        X_batch, transitions_batch = X_batch.T, transitions_batch.T

        feed = {ts.transitions[t]: transitions_batch[t]
                for t in range(ts.num_timesteps)}
        feed[ts.buff] = X_batch
        feed[ts.num_transitions] = num_transitions_batch
        feed[ys] = y_batch

        xent_loss_batch, _ = sess.run([tf.reduce_mean(xent_loss), train_op], feed)
        print xent_loss_batch



if __name__ == '__main__':
    gflags.DEFINE_integer("batch_size", 64, "")
    gflags.DEFINE_integer("vocab_size", 100, "")
    gflags.DEFINE_integer("seq_length", 29, "")
    gflags.DEFINE_integer("num_classes", 3, "")

    gflags.DEFINE_integer("model_dim", 128, "")
    gflags.DEFINE_integer("embedding_dim", 128, "")
    gflags.DEFINE_integer("tracking_dim", 32, "")

    gflags.DEFINE_enum("data_type", "arithmetic", ["arithmetic"], "")
    gflags.DEFINE_string("training_data_path", None, "")

    FLAGS(sys.argv)
    main()
