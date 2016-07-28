import os
import sys

import tensorflow as tf

so_path = os.path.join(os.path.dirname(__file__), "libthin_stack_ops_impl_gpu.so")
try:
    _library = tf.load_op_library(so_path)

except tf.errors.AlreadyExistsError:
    pass
except Exception, e:
    print >> sys.stderr, "Warning: compiled custom op library could not be loaded. Falling back to slow ops."
    print >> sys.stderr, "\t%s" % e
    from util.custom_ops.fallback import *

else:
    from util.custom_ops.core import *
