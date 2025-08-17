import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

from tensorflow.python.platform import build_info as tf_build_info
print(tf_build_info.build_info)
