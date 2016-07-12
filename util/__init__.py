from util.blocks import *
from util.data import *


# Load custom TF module.
import os
import tensorflow as tf
floaty_path = os.path.join(os.path.dirname(__file__), "..", "lib", "floaty_ops.so")
floaty = tf.load_op_library(floaty_path)
