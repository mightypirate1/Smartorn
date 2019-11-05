import tensorflow as tf
import numpy as np

#TF-abbreviations
def remove_nan(x, value=0.0):
    return tf.where(tf.math.is_nan(x), value * tf.ones_like(x), x)

def dot(x,y):
    return tf.reduce_sum(x*y, axis=-1, keepdims=True)

#Smartorn aux-fcns
def init_pos_from_shape(shape, is_input=True, dim=3):
    s = shape[1:]
    assert len(s) == 1 , "We only do 1D inputs so far..."
    assert dim == 3, "Only 3D brains so far..."
    n = np.prod(s)
    ret = np.zeros((1,n,dim))
    increment = 2.0 / n
    z = -1.0 if is_input else 1.0
    for i in range(n):
        ret[0,i,:] = [-1+i*increment, 0, z]
    return ret

def init_dir_from_shape(shape, is_input=True, dim=3):
    s = shape[1:]
    assert len(s) == 1 , "We only do 1D inputs so far..."
    assert dim == 3, "Only 3D brains so far..."
    n = np.prod(s)
    ret = np.zeros((n,dim))
    ret[:,2] = 1
    return ret
