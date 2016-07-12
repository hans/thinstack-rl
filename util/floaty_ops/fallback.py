"""Fallback implementations of floaty ops."""

import tensorflow as tf


def floaty_scatter_update(ref, indices, updates, **kwargs):
    return tf.scatter_update(ref, tf.to_int32(indices), updates, **kwargs)


def floaty_gather(arg, idxs, **kwargs):
    return tf.gather(arg, tf.to_int32(idxs), **kwargs)

