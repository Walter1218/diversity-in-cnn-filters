from keras.regularizers import Regularizer
from keras import backend as K
from keras.layers import ZeroPadding2D, Lambda, GlobalMaxPooling2D, Concatenate
import numpy as np
import tensorflow as tf

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


def get_sim_mat_fft(x, n_filter, n_input, filter_shape):
    x_ = tf.transpose(x, [3,2,0,1])
    x_fft = get_fftshift(x_, filter_shape)
    x_fft_ = tf.transpose(x_fft, [2,3,1,0])
    denom = tf.reduce_sum(x_fft*x_fft, axis=(2,3))
    denom = tf.sqrt(tf.reduce_sum(denom, axis=1, keepdims=True))
    denom = K.dot(denom, K.transpose(denom))
    sim_mat = []
    filter_list = []
    for i in range(n_filter):
        filter_i = Lambda(lambda y : y[i:(i+1),:,:,:])(x_fft)
        filter_list.append(filter_i)
    for i in range(n_filter):
        filter_i = filter_list[i]
        sim_mat_i = []
        for j in range(n_filter):
            # d_ij = tf.reduce_sum(filter_i*filter_list[j], axis=(1,2,3))
            d_ij = K.exp(-tf.reduce_sum(K.square(filter_i-filter_list[j]), axis=(1,2,3)))
            sim_mat_i.append(d_ij)
        sim_mat.append(K.reshape(Concatenate()(sim_mat_i), (1,n_filter)))
    sim_mat_ = Concatenate(axis=0)(sim_mat)
    return sim_mat_, denom


def get_sim_mat(x, n_filter, n_input, filter_shape):
    x_padded = tf.transpose(x, [3,2,0,1])
    x_padded = ZeroPadding2D(filter_shape)(x_padded)
    sim_mat = []
    for i in range(n_filter):
        filter_i = Lambda(lambda y : y[i:(i+1),:,:,:])(x_padded)
        fmap_i = K.conv2d(filter_i, x)
        sim_i = GlobalMaxPooling2D()(fmap_i)
        sim_mat.append(sim_i)
    sim_mat_ = Concatenate(axis=0)(sim_mat)
    return sim_mat_


def get_norm_mat(x, n_filter, n_input, filter_shape):
    x_norm = K.reshape(K.square(x), (np.prod(filter_shape)*n_input, n_filter))
    x_norm = K.sqrt(K.sum(x_norm, axis=0, keepdims=True))
    x_norm = K.dot(K.transpose(x_norm), x_norm)
    return x_norm


def get_reg_term(sim_mat_, n_filter, eps):
    sim_mat_p_I = tf.add(sim_mat_, tf.eye(n_filter))
    sim_mat_p_eps_I = tf.add(sim_mat_, eps*tf.eye(n_filter))
    det_1 = tf.matrix_determinant(sim_mat_p_eps_I)
    det_2 = tf.matrix_determinant(sim_mat_p_I)
    return det_1, det_2


def get_stats(x, n_filter, n_input, filter_shape, eps):
    # sim_mat = get_sim_mat(x, n_filter, n_input, filter_shape)
    # denom = get_norm_mat(x, n_filter, n_input, filter_shape)
    sim_mat, denom = get_sim_mat_fft(x, n_filter, n_input, filter_shape)
    # sim_mat_ = sim_mat/denom
    sim_mat_ = sim_mat
    det1, det2 = get_reg_term(sim_mat_, n_filter, eps)
    return sim_mat_, sim_mat, denom, det1, det2


class Diversity2D(Regularizer):
    def __init__(self, n_filter, n_input, filter_shape, l1=0., l2=0., eps=1e-8):
        self.filter_shape = filter_shape
        self.n_filter = n_filter
        self.n_input = n_input
        self.l1 = l1
        self.l2 = l2
        self.eps = eps

    def __call__(self, x):
        regularization = 0.
        # denom = get_norm_mat(x, self.n_filter, self.n_input, self.filter_shape)
        # sim_mat_ = get_sim_mat(x, self.n_filter, self.n_input, self.filter_shape)
        sim_mat_, denom = get_sim_mat_fft(x, self.n_filter, self.n_input, self.filter_shape)
        # sim_mat_ = tf.divide(sim_mat_, denom)
        det_1, det_2 = get_reg_term(sim_mat_, self.n_filter, 0.)
        regularization += self.l2 * (K.square(tf.log(det_1 + self.eps)-tf.log(det_2))-tf.log(det_1))
        # regularization += self.l2 * (K.abs(tf.log(det_1)-tf.log(det_2)))
        # regularization += self.l2 * (K.abs((det_1)-(det_2)))
        return regularization

    def get_config(self):
        return {'filter_shape': self.filter_shape,
                'n_filter': self.n_filter,
                'n_input': self.n_input,
                'l1': self.l1,
                'l2': self.l2,
                'eps': self.eps}