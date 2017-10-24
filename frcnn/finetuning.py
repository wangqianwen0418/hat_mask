'''
fine tuning the faster r-cnn model for hat/mask detection
'''

from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time

from keras import backend as K
from kereas import layers
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils


from keras_frcnn import config
from keras_frcnn import roi_helpers
import keras_frcnn.resnet as nn

from test_frcnn import format_img
from ulti import load_data
ulti.data_dir = "data"
ulti.num_imgs = 500
ulti.ratio_train = 0.8
ulti.w = 224
ulti.h = 224

x_train, y_train = load_data(dset="train", target="no_hat")
x_test, y_test = load_data(dset="test", target="no_hat")

epoch_length = 1000
num_epochs = 200
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
    
with open("model_frcnn/config.pickle", 'rb') as f_in:
	C = pickle.load(f_in)
f_in.close()

C.use_horizontal_flips = True
C.use_vertical_flips = False
C.rot_90 = False
C.num_rois = 32 #Number of ROIs per iteration
num_features = 1024 # 1024 for resnet, 512 for vgg
input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# the base network, i use a resnet50 here
shared_layers = nn.nn_base(img_input, trainable=False)

# the rpn, built upon the base network
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=False)

model_rpn = Model(img_input, rpn_layers)

model_classifier = Model([feature_map_input, roi_input], classifier)

# load pretrained weights
model_rpn.load_weights("model_frcnn/model_frnn.hdf5", by_name=True)
model_classifier.load_weights("model_frcnn/model_frnn.hdf5", by_name=True)

x = model_classifier.get_layer("time_distributed_1").output
x = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
x = layers.Lambda(lambda x: K.max(x))(x)
model_tuning = Model([img_input, roi_input], x)

# model_rpn.compile(optimizer='sgd', loss='mse')
model_tuning.compile(optimizer='sgd', loss='mse')

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True, save_weights_only=False)
tf_log = keras.callbacks.TensorBoard(log_dir=tflog_dir, batch_size=batch_size)
callbacks = [checkpoint, tf_log]

for epo in num_epochs:
