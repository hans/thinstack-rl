from util.blocks import *
from util.data import *


# Load custom TF module.
import tensorflow as tf
floaty = tf.load_op_library("floaty_ops.so")
