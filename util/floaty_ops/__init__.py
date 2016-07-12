# Load custom TF module.
import os
import sys

import tensorflow as tf

floaty_path = os.path.join(os.path.dirname(__file__), "floaty_ops_impl.so")
try:
    _library = tf.load_op_library(floaty_path)
    from util.floaty_ops.core import *

except:
    print >> sys.stderr, "Warning: compiled floaty library could not be loaded. Falling back to slow ops"
    from util.floaty_ops.fallback import *
