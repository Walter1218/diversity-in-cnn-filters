import os
os.environ['KERAS_BACKEND'] = "tensorflow"
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.layers import ZeroPadding2D, Lambda, GlobalMaxPooling2D, Concatenate
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
np.seed(2)

def tf_roll(x, shift, axis, axis_shape):
    n = axis_shape
    shift[0] %= n[0]
    shift[1] %= n[1]
    indexes = [None]*2
    indexes[0] = np.concatenate([np.arange(n[0] - shift[0], n[0]), np.arange(n[0] - shift[0])])
    indexes[1] = np.concatenate([np.arange(n[1] - shift[1], n[1]), np.arange(n[1] - shift[1])])
    res = tf.gather(x, indexes[0], axis=axis[0])
    res = tf.gather(res, indexes[1], axis=axis[1])
    return res


def fft_shift(x, filter_shape, axis):
    return tf_roll(x, [filter_shape[0]//2,filter_shape[1]//2], axis, [filter_shape[0], filter_shape[1]])


def get_fftshift(x, filter_shape):
    x_fft = tf.fft2d(tf.cast(x,tf.complex64))
    # x_fft_mag = tf.log(1+tf.abs(x_fft))
    x_fft_mag = tf.abs(x_fft)
    # return fft_shift(x_fft_mag, (filter_shape[0]*2, filter_shape[1]*2), (2,3))
    return x_fft_mag


base_filter = np.random.normal(0., 1., (1,1,3,3))
fltr_1 = np.zeros((1,1,6,6)).astype(base_filter.dtype)
fltr_2 = np.zeros((1,1,6,6)).astype(base_filter.dtype)
fltr_1[0, 0, 1:4, 1:4] = base_filter
fltr_2[0, 0, 2:5, 2:5] = base_filter


plt.imshow(np.concatenate([fltr_1[0,0,:,:], fltr_2[0,0,:,:]], axis=0))
plt.show()

fft_1 = get_fftshift(fltr_1, (3,3))
fft_1 = sess.run(fft_1)
fft_2 = get_fftshift(fltr_2, (3,3))
fft_2 = sess.run(fft_2)
print("fft_1_shape", fft_1.shape)
print("fft_2_shape", fft_2.shape)

plt.imshow(np.concatenate([fft_1[0,0,:,:], fft_2[0,0,:,:]], axis=0))
plt.show()