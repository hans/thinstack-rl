from collections import namedtuple
from functools import partial
from pprint import pprint
import sys

import gflags
import tensorflow as tf
from tensorflow.contrib import layers

from data.arithmetic import load_simple_data as load_arithmetic_data
from data.snli import load_snli_data
from reinforce import reinforce_episodic_gradients
from thin_stack import ThinStack
import util

FLAGS = gflags.FLAGS

Data = namedtuple("Data", ["train_iter", "eval_iters", "buckets", "vocabulary",
                           "is_pair_data", "train_embeddings", "num_classes"])
Graph = namedtuple("Graph", ["stacks", "logits", "ys", "gradients",
                             "num_timesteps", "learning_rate", "train_op",
                             "summary_op", "is_training"])


def mlp_classifier(x, num_classes, mlp_dims=(1024,1024), scope=None):
    with tf.variable_scope(scope or "classifier"):
        dims = (x.get_shape()[1],) + mlp_dims
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


def build_model(num_timesteps, vocab_size, classifier_fn, is_training,
                train_embeddings=True, initial_embeddings=None, num_classes=3):
    with tf.variable_scope("Model", initializer=util.HeKaimingInitializer()):
        ys = tf.placeholder(tf.int32, (FLAGS.batch_size,), "ys")

        tracking_fn = lambda *xs: xs[0]
        compose_fn = util.TreeLSTMLayer
        def transition_fn(*xs):
            """Return random logits."""
            return tf.random_uniform((FLAGS.batch_size, 2), minval=-10, maxval=10)

        ts = ThinStack(compose_fn, tracking_fn, transition_fn, FLAGS.batch_size,
                       vocab_size, num_timesteps, FLAGS.model_dim,
                       FLAGS.embedding_dim, FLAGS.tracking_dim, is_training,
                       embeddings=initial_embeddings)

        logits = classifier_fn(ts.final_representations)
        assert logits.get_shape()[1] == num_classes

        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys)
        xent_loss = tf.reduce_mean(xent_loss)
        tf.scalar_summary("xent_loss", xent_loss)

        rewards = build_rewards(logits, ys)
        tf.scalar_summary("avg_reward", tf.reduce_mean(rewards))

        params = tf.trainable_variables()

        if not train_embeddings:
            params.remove(ts.embeddings)
        xent_gradients = zip(tf.gradients(xent_loss, params), params)
        rl_gradients = reinforce_episodic_gradients(
                ts.p_transitions, ts.sampled_transitions, rewards,
                params=params)

        gradients = xent_gradients + rl_gradients
    return (ts,), logits, ys, gradients


def build_sentence_pair_model(num_timesteps, vocab_size, classifier_fn, is_training, num_classes,
                              train_embeddings=True, initial_embeddings=None):
    initializer = tf.random_uniform_initializer(-0.005, 0.005)
    with tf.variable_scope("PairModel", initializer=initializer):
        ys = tf.placeholder(tf.int32, (FLAGS.batch_size,), "ys")

        assert FLAGS.model_dim % 2 == 0, "model_dim must be even; we're using LSTM memory cells which are divided in half"

        def embedding_project_fn(embeddings):
            if FLAGS.embedding_dim != FLAGS.model_dim:
                # Need to project embeddings to model dimension.
                embeddings = util.Linear(embeddings, FLAGS.model_dim, bias=False)
            if FLAGS.embedding_batch_norm:
                embeddings = layers.batch_norm(embeddings, center=True, scale=True,
                                            is_training=True)
            if FLAGS.embedding_keep_rate < 1.0:
                embeddings = tf.cond(is_training,
                        lambda: tf.nn.dropout(embeddings, FLAGS.embedding_keep_rate),
                        lambda: embeddings / FLAGS.embedding_keep_rate)
            return embeddings

        # Share scope across the two models. (==> shared embedding projection /
        # BN weights)
        embedding_project_fn = tf.make_template("embedding_project", embedding_project_fn)

        ts_args = {
            "compose_fn": util.TreeLSTMLayer,
            "tracking_fn": util.LSTMLayer,
            "transition_fn": None,
            "embedding_project_fn": embedding_project_fn,
            "batch_size": FLAGS.batch_size,
            "vocab_size": vocab_size,
            "num_timesteps": num_timesteps,
            "model_dim": FLAGS.model_dim,
            "embedding_dim": FLAGS.embedding_dim,
            "tracking_dim": FLAGS.tracking_dim,
            "is_training": is_training,
            "embeddings": initial_embeddings,
        }

        with tf.variable_scope("s1"):
            ts_1 = ThinStack(**ts_args)
        with tf.variable_scope("s2"):
            ts_2 = ThinStack(**ts_args)

        # Extract just the hidden value of the LSTM (not cell state)
        repr_dim = FLAGS.model_dim / 2
        ts_1_repr = ts_1.final_representations[:, :repr_dim]
        ts_2_repr = ts_2.final_representations[:, :repr_dim]

        # Now prep return representations
        mlp_inputs = [ts_1_repr, ts_2_repr]

        if FLAGS.use_difference_feature:
            mlp_inputs.append(ts_2_repr - ts_1_repr)
        if FLAGS.use_product_feature:
            mlp_inputs.append(ts_1_repr * ts_2_repr)

        mlp_input = tf.concat(1, mlp_inputs)

        if FLAGS.sentence_repr_batch_norm:
            mlp_input = layers.batch_norm(mlp_input, center=True, scale=True,
                                          is_training=True, scope="sentence_repr_bn")
        if FLAGS.sentence_repr_keep_rate < 1.0:
            mlp_input = tf.cond(is_training,
                    lambda: tf.nn.dropout(mlp_input, FLAGS.sentence_repr_keep_rate,
                                          name="sentence_repr_dropout"),
                    lambda: mlp_input / FLAGS.sentence_repr_keep_rate)

        logits = classifier_fn(mlp_input)
        assert logits.get_shape()[1] == num_classes

        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys)
        xent_loss = tf.reduce_mean(xent_loss)
        tf.scalar_summary("xent_loss", xent_loss)

        rewards = build_rewards(logits, ys)
        tf.scalar_summary("avg_reward", tf.reduce_mean(rewards))

        params = tf.trainable_variables()
        if not train_embeddings:
            params.remove(ts_1.embeddings)
            try:
                params.remove(ts_2.embeddings)
            except: pass

        l2_loss = tf.add_n([tf.reduce_sum(tf.square(param))
                            for param in params])
        tf.scalar_summary("l2_loss", l2_loss)
        total_loss = xent_loss + FLAGS.l2_lambda * l2_loss

        xent_gradients = zip(tf.gradients(total_loss, params), params)
        # TODO enable for transition_fn != None
        # rl1_gradients = reinforce_episodic_gradients(
        #         ts_1.p_transitions, ts_1.sampled_transitions, rewards,
        #         params=params)
        # rl2_gradients = reinforce_episodic_gradients(
        #         ts_2.p_transitions, ts_2.sampled_transitions, rewards,
        #         params=params)
        rl1_gradients, rl2_gradients = [], []
        # TODO store magnitudes in summaries?
        gradients = xent_gradients + rl1_gradients + rl2_gradients

    return (ts_1, ts_2), logits, ys, gradients


def prepare_data():
    if FLAGS.data_type == "arithmetic":
        data_manager = load_arithmetic_data
    elif FLAGS.data_type == "snli":
        data_manager = load_snli_data

    sentence_pair_data = data_manager.SENTENCE_PAIR_DATA

    raw_data, vocabulary = data_manager.load_data(FLAGS.training_data_path)

    raw_eval_data = []
    if FLAGS.eval_data_path:
        for eval_filename in FLAGS.eval_data_path.split(":"):
            eval_data, _ = data_manager.load_data(eval_filename)
            raw_eval_data.append((eval_filename, eval_data))

    train_embeddings = True
    if not vocabulary:
        vocabulary = util.BuildVocabulary(raw_data, raw_eval_data,
                FLAGS.embedding_data_path, sentence_pair_data=sentence_pair_data)
        # Don't train embeddings on open vocabulary
        tf.logging.warn("Training on open vocabulary, so not training embeddings.")
        train_embeddings = False

    data = util.data.TokensToIDs(vocabulary, raw_data,
                                 sentence_pair_data=sentence_pair_data)
    eval_data = []
    for filename, eval_data_i in raw_eval_data:
        eval_data_i = util.data.TokensToIDs(vocabulary, eval_data_i,
                                            sentence_pair_data=sentence_pair_data)
        eval_data.append((filename, eval_data_i))

    #### TRAINING DATA

    tf.logging.info("Preprocessing training data.")
    # TODO customizable
    #
    # Sort of even bucketing for SNLI train: 15, 17, 19, 21, 23, 25, 29, 33, 39, 49, 171
    # ~50k in each bucket
    #
    # ~100k in each bucket, fewer buckets:
    # 17, 21, 25, 33, 49, 71, 171
    # DEV TESTING [17, 21, 25, 33, 49, 71, 171]
    buckets = [int(arg) for arg in FLAGS.buckets.split(",")]
    bucketed_data = util.data.PadAndBucket(data, buckets, FLAGS.batch_size,
                                           sentence_pair_data=sentence_pair_data,
                                           discard_long_examples=FLAGS.discard_long_examples)
    tf.logging.info("Bucket distribution:\n\t" +
                    "\n\t".join("Length %3i: %7i examples"
                                % (bucket, len(bucketed_data[bucket]))
                                for bucket in sorted(bucketed_data.keys())))

    # Convert each bucket into TF-friendly arrays
    bucketed_data = {length: util.data.BucketToArrays(bucket, data_manager)
                     for length, bucket in bucketed_data.iteritems()}

    iterator = util.data.MakeBucketedTrainingIterator(bucketed_data, FLAGS.batch_size)

    #### EVAL DATA

    tf.logging.info("Preprocessing eval data.")
    eval_iterators = []
    for name, eval_data_i in eval_data:
        eval_data_i = util.data.PadDataset(eval_data_i, buckets[-1],
                                           sentence_pair_data=sentence_pair_data)
        eval_data_i = util.data.BucketToArrays(eval_data_i, data_manager)
        iterator = util.data.MakeEvalIterator(eval_data_i, FLAGS.batch_size)
        eval_iterators.append((name, iterator))

    return Data(iterator, eval_iterators, buckets, vocabulary,
                sentence_pair_data, train_embeddings, data_manager.NUM_CLASSES)


def build_graphs(model_fn, buckets):
    graphs = {}
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    is_training = tf.placeholder(tf.bool, (), name="is_training")
    opt = tf.train.RMSPropOptimizer(learning_rate)

    for i, num_timesteps in enumerate(buckets):
        summaries_so_far = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        with tf.variable_scope("train/", reuse=i > 0):
            stacks, logits, ys, gradients = model_fn(num_timesteps,
                                                     is_training=is_training)

            # Set up histogram displays
            params = set()
            for gradient, param in gradients:
                params.add(param)
                if gradient is not None:
                    tf.histogram_summary(gradient.name + "b%i" % num_timesteps, gradient)
            for param in params:
                tf.histogram_summary(param.name + "b%i" % num_timesteps, param)

            # Clip gradients.
            clipped_gradients = tf.clip_by_global_norm(
                    [grad for grad, param in gradients], FLAGS.grad_clip)
            clipped_gradients = zip(clipped_gradients,
                                    [param for _, param in gradients])

            train_op = opt.apply_gradients(clipped_gradients, global_step)

            new_summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
            new_summary_ops = set(new_summary_ops) - summaries_so_far
            summary_op = tf.merge_summary(list(new_summary_ops))

            graphs[num_timesteps] = Graph(stacks, logits, ys, clipped_gradients,
                                          num_timesteps, learning_rate,
                                          train_op, summary_op,
                                          is_training)

    return graphs, global_step


def run_batch(sess, graph, batch_data, learning_rate, do_summary=True,
              is_training=True, profiler=None):
    for stack in graph.stacks:
        stack.reset(sess)

    # each batch data element has leading batch axis
    # X: (B, num_stacks, buffer_size)
    # transitions: (B, num_stacks, num_timesteps)
    # num_transitions: (B, num_stacks)
    X, transitions, num_transitions, ys = batch_data

    # Prepare feed dict
    feed = {
        graph.ys: ys,
        graph.learning_rate: learning_rate,
        graph.is_training: is_training,
    }
    for i, stack in enumerate(graph.stacks):
        # Swap batch axis to front.
        X_i = X[:, i].T
        transitions_i = transitions[:, i, :].T

        feed.update({stack.transitions[t]: transitions_i[t]
                     for t in range(graph.num_timesteps)})
        feed[stack.buff] = X_i
        feed[stack.num_transitions] = num_transitions[:, i]

    # Sub in a no-op for summary op if we don't want to compute summaries.
    summary_op_ = graph.summary_op
    if not do_summary:
        summary_op_ = graph.train_op

    kwargs = {}
    if profiler is not None:
        kwargs["options"] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        kwargs["run_metadata"] = profiler

    _, summary = sess.run([graph.train_op, summary_op_], feed, **kwargs)
    return summary


def main():
    pprint(FLAGS.FlagValuesDict())
    sys.stdout.flush()

    tf.logging.info("Loading and preparing data.")
    data = prepare_data()

    if FLAGS.embedding_data_path:
        embeddings = util.LoadEmbeddingsFromASCII(
                data.vocabulary, FLAGS.embedding_dim, FLAGS.embedding_data_path)
        with tf.device("/cpu:0"):
            embeddings = tf.Variable(embeddings, name="embeddings")
    else:
        embeddings = None

    tf.logging.info("Building training graphs.")
    classifier_fn = partial(mlp_classifier, num_classes=data.num_classes)
    model_fn = build_sentence_pair_model if data.is_pair_data else build_model
    model_fn = partial(model_fn, vocab_size=len(data.vocabulary),
                       classifier_fn=classifier_fn,
                       train_embeddings=data.train_embeddings,
                       initial_embeddings=embeddings,
                       num_classes=data.num_classes)
    graphs, global_step = build_graphs(model_fn, data.buckets)

    summary_op = tf.merge_all_summaries()
    no_op = tf.constant(0.0)

    tf.logging.info("Preparing to run training.")
    savable_variables = set(tf.all_variables())
    for graph in graphs.values():
        for stack in graph.stacks:
            savable_variables -= set(stack._aux_vars)
    saver = tf.train.Saver(savable_variables)
    sv = tf.train.Supervisor(logdir=FLAGS.logdir, global_step=global_step,
                             saver=saver, summary_op=None)

    run_metadata = tf.RunMetadata()
    with sv.managed_session(FLAGS.master) as sess:
        tf.logging.info("Training.")
        for step, (bucket, batch_data) in zip(xrange(FLAGS.training_steps), data.train_iter):
            if step % 100 == 0:
                tf.logging.info("%i", step)
            if sv.should_stop():
                break

            learning_rate = FLAGS.learning_rate * (FLAGS.learning_rate_decay_per_10k_steps ** (step / 10000.0))

            do_summary = step % FLAGS.summary_step_interval == 0
            profiler = run_metadata if FLAGS.profile and do_summary else None
            ret = run_batch(sess, graphs[bucket], batch_data, learning_rate,
                            do_summary, profiler)

            if do_summary:
                sv.summary_computed(sess, ret)

    if FLAGS.profile:
        from tensorflow.python.client import timeline
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        with open("timeline.ctf.json", "w") as timeline_f:
            timeline_f.write(trace.generate_chrome_trace_format())



if __name__ == '__main__':
    gflags.DEFINE_string("master", "", "")
    gflags.DEFINE_string("logdir", "/tmp/rl-stack", "")
    gflags.DEFINE_integer("summary_step_interval", 100, "")
    gflags.DEFINE_integer("training_steps", 10000, "")
    gflags.DEFINE_boolean("profile", False, "")

    gflags.DEFINE_integer("batch_size", 64, "")
    gflags.DEFINE_string("buckets", "17,171", "")
    gflags.DEFINE_boolean("discard_long_examples", True, "")

    gflags.DEFINE_integer("model_dim", 128, "")
    gflags.DEFINE_integer("embedding_dim", 128, "")
    gflags.DEFINE_integer("tracking_dim", 32, "")

    gflags.DEFINE_float("embedding_keep_rate", 1.0, "")
    gflags.DEFINE_float("sentence_repr_keep_rate", 1.0, "")
    gflags.DEFINE_boolean("embedding_batch_norm", False, "")
    gflags.DEFINE_boolean("sentence_repr_batch_norm", False, "")

    gflags.DEFINE_boolean("use_difference_feature", True, "")
    gflags.DEFINE_boolean("use_product_feature", True, "")

    gflags.DEFINE_float("learning_rate", 0.01, "")
    gflags.DEFINE_float("learning_rate_decay_per_10k_steps", 0.75, "")
    gflags.DEFINE_float("grad_clip", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 0.0, "")

    gflags.DEFINE_enum("data_type", "arithmetic", ["arithmetic", "snli"], "")
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string("eval_data_path", None, "")
    gflags.DEFINE_string("embedding_data_path", None, "")

    FLAGS(sys.argv)
    tf.logging.set_verbosity(tf.logging.INFO)

    main()
