import tensorflow as tf


def ewma_baseline(rewards, tau=0.9):
    """
    Baseline observed rewards using an exponential weighted moving average.

    Args:
        rewards: batch of batch_size reward floats

    Returns:
        baselined_rewards: batch of batch_size reward floats
    """

    avg_reward = tf.Variable(0.0, trainable=False, name="avg_reward")
    new_avg_reward = tau * avg_reward + (1. - tau) * tf.reduce_mean(rewards)
    tf.scalar_summary("baseline", new_avg_reward)

    assign_avg_reward = avg_reward.assign(new_avg_reward)

    with tf.control_dependencies([assign_avg_reward]):
        return avg_reward


def reinforce_episodic_gradients(logits, sampled_outputs, rewards,
                                 baseline_fn=ewma_baseline, params=None):
    """
    Calculate REINFORCE gradients given a batch of single episodic rewards.

    Args:
        logits: list of `num_timesteps` batches, each of size
            `batch_size * num_classes`: logits for distribution over actions
            at each timestep
        sampled_outputs: list of `num_timesteps` batches, each of size
            `batch_size`: ints describing sampled action at each timestep
            for each example
        rewards: float batch `batch_size` describing episodic reward per
            example
        baseline_fn:
        params:

    Return:
        updates: list of (gradient, param) update tuples
    """

    if params is None:
        params = tf.trainable_variables()

    batch_size = tf.shape(logits[0])[0]
    num_classes = tf.shape(logits[0])[1]
    num_timesteps = len(logits)

    # Baseline empirical rewards
    rewards = baseline_fn(rewards)

    # Feed logits through log softmax.
    log_softmaxes = [tf.nn.log_softmax(logits_t) for logits_t in logits]

    # Fetch p(sampled_output) for each timestep.
    # This is a bit awkward -- need to pick a single element out of each
    # example softmax vector.
    # Output is (batch_size, 1) per timestep
    flat_softmaxes = [tf.reshape(log_softmax_t, (-1,))
                      for log_softmax_t in log_softmaxes]
    lookup_offset = tf.range(batch_size) * num_classes
    log_p_sampled = [tf.gather(log_softmax_t, sampled_outputs_t + lookup_offset)
                     for log_softmax_t, sampled_outputs_t
                     in zip(flat_softmaxes, sampled_outputs)]

    # Merge into single (batch_size, num_timesteps) batch
    log_p_sampled = [tf.expand_dims(log_p_sampled_t, 1)
                     for log_p_sampled_t in log_p_sampled]
    log_p_sampled = tf.concat(1, log_p_sampled)
    # Calculate p(sampled_output) by chain rule. We can merge these ahead of
    # time, since we only have a single episode-level reward.
    log_p_sampled = tf.reduce_sum(log_p_sampled, 1)

    # Main REINFORCE gradient equation.
    # Apply rewards on batch + mean beforehand for efficiency.
    log_p_sampled *= -1 * rewards
    log_p_sampled = tf.reduce_mean(log_p_sampled)
    gradients = tf.gradients(log_p_sampled, params)

    return zip(gradients, params)

