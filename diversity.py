from keras.regularizers import Regularizer
from keras import backend as K
from keras.layers import ZeroPadding2D, Lambda, GlobalMaxPooling2D, Concatenate
import numpy as np
import tensorflow as tf

def get_sim_mat_fft(x, n_filter, n_input, filter_shape):
    x_ = tf.transpose(x, [3,2,0,1])
    x_fft = tf.abs(tf.fft2d(tf.cast(x_,tf.complex64)))
    # x_fft_ = tf.transpose(x_fft, [2,3,1,0])
    # denom = tf.reduce_sum(x_fft*x_fft, axis=(2,3))
    # denom = tf.sqrt(tf.reduce_sum(denom, axis=1, keepdims=True))
    # denom = K.dot(denom, K.transpose(denom))
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
    return sim_mat_


def get_reg_term(sim_mat, n_filter):
    sim_mat_p_I = tf.add(sim_mat, tf.eye(n_filter))
    det_1 = tf.matrix_determinant(sim_mat)
    det_2 = tf.matrix_determinant(sim_mat_p_I)
    return det_1, det_2


def get_stats(x, n_filter, n_input, filter_shape):
    sim_mat = get_sim_mat_fft(x, n_filter, n_input, filter_shape)
    det1, det2 = get_reg_term(sim_mat, n_filter)
    return sim_mat, det1, det2


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
        sim_mat = get_sim_mat_fft(x, self.n_filter, self.n_input, self.filter_shape)
        det_1, det_2 = get_reg_term(sim_mat, self.n_filter)
        p_term = tf.log(det_1 + self.eps)-tf.log(det_2)
        if self.l2:
            regularization += self.l2 * (K.square(p_term))
        if self.l1:
            regularization += self.l1 * (K.abs(p_term))
        return regularization

    def get_config(self):
        return {'filter_shape': self.filter_shape,
                'n_filter': self.n_filter,
                'n_input': self.n_input,
                'l1': self.l1,
                'l2': self.l2,
                'eps': self.eps}