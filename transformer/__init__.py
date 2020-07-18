import tensorflow as tf
import os

cur_dir = os.path.dirname(__file__)

lib_filename = None
for _ in os.listdir(cur_dir):
    print ('==============================', _)
    if _.endswith('.so') or _.endswith('.dylib'):
        lib_filename = _
        break

print('---------------------------', cur_dir, lib_filename)
Decode = tf.load_op_library(os.path.join(cur_dir, lib_filename))

