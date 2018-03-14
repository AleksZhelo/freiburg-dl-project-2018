import tensorflow as tf


def exponential_decay(learning_rate_start, learning_rate_end,
                      global_step, decay_steps, decay_in_epochs, name=None):
    if global_step is None:
        raise ValueError("global_step is required for exponential decay.")
    with tf.name_scope(name, "ExpDecay",
                       [learning_rate_start, learning_rate_end, global_step,
                        decay_steps, decay_in_epochs]) as name:  # TODO: needed?
        learning_rate_start = tf.convert_to_tensor(learning_rate_start, name="learning_rate")
        dtype = learning_rate_start.dtype
        learning_rate_end = tf.cast(learning_rate_end, dtype)
        global_step = tf.cast(global_step, dtype)
        decay_steps = tf.cast(decay_steps, dtype)

        k = -1.0 / decay_in_epochs * tf.log(learning_rate_end / learning_rate_start)
        current_epoch = tf.floordiv(global_step, decay_steps)
        return tf.cond(current_epoch < decay_in_epochs,
                       lambda: learning_rate_start * tf.exp(-k * current_epoch),
                       lambda: learning_rate_end)
