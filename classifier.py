import sys

import gflags
import tensorflow as tf

from data.arithmetic import load_simple_data as load_arithmetic_data
from reinforce import reinforce_episodic_gradients
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


def build_rewards(classifier_logits, ys):
    """
    Build 0-1 classification reward for REINFORCE units within model.
    """
    return tf.to_float(tf.equal(tf.to_int32(tf.argmax(classifier_logits, 1)),
                                ys))


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
        xent_loss = tf.reduce_mean(xent_loss)
        tf.scalar_summary("xent_loss", xent_loss)

        rewards = build_rewards(logits, ys)
        tf.scalar_summary("avg_reward", tf.reduce_mean(rewards))

        params = tf.trainable_variables()
        xent_gradients = zip(tf.gradients(xent_loss, params), params)
        rl_gradients = reinforce_episodic_gradients(
                ts.p_transitions, ts.sampled_transitions, rewards,
                params=params)
        # TODO store magnitudes in summaries?
        gradients = xent_gradients + rl_gradients

    return ts, logits, ys, gradients


def prepare_data():
    if FLAGS.data_type == "arithmetic":
        data_manager = load_arithmetic_data

    sentence_pair_data = data_manager.SENTENCE_PAIR_DATA

    raw_data, vocabulary = data_manager.load_data(FLAGS.training_data_path)
    data = util.data.TokensToIDs(vocabulary, raw_data,
                                 sentence_pair_data=sentence_pair_data)

    # TODO customizable
    buckets = [21]#[9, 21]
    bucketed_data = util.data.PadAndBucket(data, buckets,
                                           sentence_pair_data=sentence_pair_data)

    # Convert each bucket into TF-friendly arrays
    bucketed_data = {length: util.data.BucketToArrays(bucket, length,
                                                      sentence_pair_data=sentence_pair_data)
                     for length, bucket in bucketed_data.iteritems()}

    iterator = util.data.MakeBucketedTrainingIterator(bucketed_data, FLAGS.batch_size)

    return iterator, buckets, vocabulary


def build_training_graphs(buckets, global_step):
    graphs = {}
    opt = tf.train.MomentumOptimizer(0.001, 0.9)

    for i, num_timesteps in enumerate(buckets):
        summaries_so_far = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        with tf.variable_scope("train/", reuse=i > 0):
            ts, logits, ys, gradients = build_model(num_timesteps)
            train_op = opt.apply_gradients(gradients, global_step)

            new_summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
            new_summary_ops = set(new_summary_ops) - summaries_so_far
            summary_op = tf.merge_summary(list(new_summary_ops))

            graphs[num_timesteps] = (ts, logits, ys, train_op, summary_op)

    return graphs


def run_batch(sess, graph, batch_data, do_summary=True):
    ts, logits, ys, train_op, summary_op = graph
    ts.reset(sess)

    X_batch, transitions_batch, y_batch, num_transitions_batch = batch_data
    X_batch, transitions_batch = X_batch.T, transitions_batch.T

    feed = {ts.transitions[t]: transitions_batch[t]
            for t in range(ts.num_timesteps)}
    feed[ts.buff] = X_batch
    feed[ts.num_transitions] = num_transitions_batch
    feed[ys] = y_batch

    # Sub in a no-op for summary op if we don't want to compute summaries.
    if not do_summary:
        summary_op = train_op

    fetches = [train_op, summary_op]
    _, summary = sess.run(fetches, feed)
    return summary


def main():
    global_step = tf.Variable(0, trainable=False, name="global_step")

    training_iterator, training_buckets, vocabulary = prepare_data()
    training_graphs = build_training_graphs(training_buckets, global_step)

    summary_op = tf.merge_all_summaries()
    no_op = tf.constant(0.0)

    sv = tf.train.Supervisor(logdir=FLAGS.logdir, global_step=global_step,
                             summary_op=None)

    with sv.managed_session(FLAGS.master) as sess:
        for step, (bucket, batch_data) in enumerate(training_iterator):
            if sv.should_stop():
                break

            do_summary = step % FLAGS.summary_step_interval == 0
            ret = run_batch(sess, training_graphs[bucket], batch_data,
                            do_summary)

            if do_summary:
                sv.summary_computed(sess, ret)



if __name__ == '__main__':
    gflags.DEFINE_string("master", "", "")
    gflags.DEFINE_string("logdir", "/tmp/rl-stack", "")
    gflags.DEFINE_integer("summary_step_interval", 100, "")

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
