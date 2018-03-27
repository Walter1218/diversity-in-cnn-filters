from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = "tensorflow"
################################################################################################
################################################################################################
################################################################################################
import numpy as np
import tensorflow as tf
import random as rn

os.environ['PYTHONHASHSEED'] = '0'

from keras import backend as K
K.set_image_data_format('channels_first')

np.random.seed(27)
rn.seed(27)
tf.set_random_seed(27)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
################################################################################################
################################################################################################
################################################################################################
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input, Lambda, Maximum, MaxoutDense, GlobalMaxPooling2D, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.constraints import max_norm
from diversity import Diversity2D, get_stats
import matplotlib.cm as cm
import matplotlib.pyplot as pl
from scipy import misc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
import functools
from sklearn.metrics import confusion_matrix
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

########################################
# Plotting functions
########################################
def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


########################################
########################################

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_channels, img_rows, img_cols = 1, 28, 28
def load_data(fpath):
    X = np.loadtxt(fpath)
    y = X[:,-1].astype(np.int)
    X = X[:,:-1]
    return (X.reshape(X.shape[0], img_channels, img_rows, img_cols), y)


print("Loading data...")
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_test, y_test), (x_train, y_train) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
input_shape = (x_train.shape[1],img_rows,img_cols)

# custom loss
def get_custom_loss():
    def custom_loss(y_true, y_pred):
        # Compute loss
        return (y_true-y_pred)*0
    return custom_loss

def print_stats(W, n_filter, n_input, filter_shape, eps, my_name=""):
    print(my_name)
    sim_mat_, sim_mat, denom, det_1, det_2 = get_stats(W, n_filter, n_input, filter_shape, eps)
    print(my_name, "sim_mat_:")
    sim_mat_ = sess.run(sim_mat_)
    print(sim_mat_)
    print(my_name, "sim_mat:")
    print(sess.run(sim_mat))
    print(my_name, "denom:")
    print(sess.run(denom))
    print(my_name, "eigenvalues of sim_mat:")
    evals, evecs = np.linalg.eig(sim_mat_)
    print(evals)
    print(my_name, "det_1, det_2: ", sess.run(det_1), sess.run(det_2))


def plot_filters(W, n_filter, n_input, filter_shape, n_epochs, my_name=""):
    f_side_0 = filter_shape[0]
    f_side_1 = filter_shape[1]
    pl.figure(figsize=(15, 15))
    nice_imshow(pl.gca(), make_mosaic(W.transpose((2,3,0,1)).reshape(n_filter*n_input, f_side_0, f_side_1), n_filter, n_input), vmin=-1.0, vmax=1.0, cmap=cm.binary)
    pl.savefig("figures/ep_" + my_name + "_" + str(n_epochs)+".png")
    pl.close()

# model
n_fmap = 6
f_side = 2
l2_ = 0.0
mnorm = 2.0
inputs = Input(shape=input_shape)
conv_1 = Conv2D(n_fmap, (f_side,f_side),kernel_initializer='random_uniform', activation="relu", kernel_regularizer=Diversity2D(n_fmap, input_shape[0], (f_side,f_side), 0., l2_), kernel_constraint=max_norm(mnorm))(inputs)
# conv_1 = Conv2D(n_fmap, (f_side,f_side),kernel_initializer='random_uniform', activation="relu", kernel_constraint=max_norm(mnorm))(inputs)
mpool_2 = MaxPooling2D((2,2))(conv_1)
conv_2 = Conv2D(n_fmap, (f_side,f_side),kernel_initializer='random_uniform', activation="relu", kernel_regularizer=Diversity2D(n_fmap, n_fmap, (f_side,f_side), 0., l2_), kernel_constraint=max_norm(mnorm))(mpool_2)
# conv_2 = Conv2D(n_fmap, (f_side,f_side),kernel_initializer='random_uniform', activation="relu", kernel_constraint=max_norm(mnorm))(mpool_2)
mpool_3 = MaxPooling2D((2,2))(conv_2)
conv_3 = Conv2D(n_fmap, (f_side,f_side),kernel_initializer='random_uniform', activation="relu", kernel_regularizer=Diversity2D(n_fmap, n_fmap, (f_side,f_side), 0., l2_), kernel_constraint=max_norm(mnorm))(mpool_3)
# conv_3 = Conv2D(n_fmap, (f_side,f_side),kernel_initializer='random_uniform', activation="relu", kernel_constraint=max_norm(mnorm))(mpool_3)
gmp = GlobalMaxPooling2D()(conv_3)
# gmp = GlobalMaxPooling2D()(conv_1)
dense_1 = Dense(512, activation="relu")(gmp)
predictions = Dense(num_classes, activation="softmax")(dense_1)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
n_epochs = 0
for i in range(50):
    model.fit(x_train, y_train/1.25+0.1,
            batch_size=batch_size,
            epochs=1,
            verbose=1,
            shuffle=False,
            steps_per_epoch=None)
    y_pred = np.argmax(model.predict(x_train), axis=1)
    conf_mat = confusion_matrix(np.argmax(y_train,axis=1), y_pred)
    n_epochs += 1
    print(i, ": stats: ")
    print("conf_mat:")
    print(conf_mat)
    print_stats(model.layers[1].weights[0], n_fmap, input_shape[0], (f_side,f_side), 0., "W_0:")
    print_stats(model.layers[3].weights[0], n_fmap, n_fmap, (f_side,f_side), 0., "W_1:")
    print_stats(model.layers[5].weights[0], n_fmap, n_fmap, (f_side,f_side), 0., "W_2:")
    plot_filters(model.layers[1].get_weights()[0], n_fmap, input_shape[0], (f_side, f_side), n_epochs, "W_0")
    plot_filters(model.layers[3].get_weights()[0], n_fmap, n_fmap, (f_side, f_side), n_epochs, "W_1")
    plot_filters(model.layers[5].get_weights()[0], n_fmap, n_fmap, (f_side, f_side), n_epochs, "W_2")    


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = "keras_mnist_basic_trained_model.h5"
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)