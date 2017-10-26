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
from keras import layers
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils


from keras_frcnn import config
from keras_frcnn import roi_helpers
from keras_frcnn.my_parser import get_data, my_generator 
import keras_frcnn.resnet as nn

# for reproducible training
seed = 7
np.random.seed(seed)
PYTHONHASHSEED=0
from tensorflow import set_random_seed 
set_random_seed(2)

# from test_frcnn import format_img
# load the model config
with open("model_frcnn/config.pickle", 'rb') as f_in:
	C = pickle.load(f_in)
f_in.close()

C.use_horizontal_flips = True
C.use_vertical_flips = False
C.rot_90 = True
C.num_rois = 32 #Number of ROIs per iteration
C.model_path = "model_frcnn/model_tuning_exp2.h5"

class_mapping = C.class_mapping
if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)
class_mapping = {v: k for k, v in class_mapping.items()}

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

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)

model_classifier_only = Model([feature_map_input, roi_input], classifier)

# load pretrained weights
model_rpn.load_weights("model_frcnn/model_frnn.hdf5", by_name=True)
model_classifier_only.load_weights("model_frcnn/model_frnn.hdf5", by_name=True)

x = model_classifier_only.get_layer("time_distributed_1").output
x = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
# x = layers.Lambda(lambda x: K.max(x))(x)
# x = layers.Flatten()(x)
x = layers.Lambda(lambda x: K.max(x, axis=1))(x)
x = layers.Activation("sigmoid")(x)
model_tuning = Model([feature_map_input, roi_input], x)

# model_rpn.compile(optimizer='sgd', loss='mse')
model_tuning.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
model_tuning.summary()

train_imgs, train_labels = get_data("../data/", mode="train", target="no_hat")
val_imgs, val_labels = get_data("../data/", mode="val", target="no_hat")

data_gen_train = my_generator(train_imgs, train_labels, C, mode='train')
data_gen_val = my_generator(val_imgs, val_labels, C, mode='val')

# train the model_tuning
epoch_length = 500
num_epochs = 200
iter_num = 0

losses = np.zeros((epoch_length, 2))
start_time = time.time()

best_loss = np.Inf
bbox_threshold = 0.8

class_mapping_inv = {v: k for k, v in class_mapping.items()}
for epoch_num in range(num_epochs):
  progbar = generic_utils.Progbar(epoch_length)
  print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
  while True:
    try:
      # not efficient, one image is a batch
      img, label = next(data_gen_train)
      [Y1, Y2, F] = model_rpn.predict(img)
      R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7) # R.shape = [max_boxes, 4]
      # convert from (x1,y1,x2,y2) to (x,y,w,h)
      R[:, 2] -= R[:, 0]
      R[:, 3] -= R[:, 1]
      person_roi = np.zeros((1, C.num_rois, 4))
      person_i = 0
      for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0) # shape [1, num_rois, 4]
        if ROIs.shape[1] == 0:
          break
        if jk == R.shape[0]//C.num_rois:
          curr_shape = ROIs.shape
          target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
          ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
          ROIs_padded[:, :curr_shape[1], :] = ROIs
          ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
          ROIs = ROIs_padded
        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
        for ii in range(P_cls.shape[1]):
          if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
            continue
          cls_id = np.argmax(P_cls[0, ii, :])
          cls_name = class_mapping[cls_id]
          if cls_name == "person":
            person_roi[0, person_i, :] = ROIs[0, ii, :]
            person_i += 1
      if person_i == 0:
        continue
      while person_i < C.num_rois:
        index = np.random.randint(0, person_i)
        person_roi[0, person_i, :] = ROIs[0, index, :]
        person_i += 1
      loss, acc = model_tuning.train_on_batch([F, person_roi], label)
      losses[iter_num, 0] = loss
      losses[iter_num, 1] = acc
      progbar.update(iter_num, [('loss', np.mean(losses[:iter_num, 0])), (' acc', np.mean(losses[:iter_num, 1]))])
      iter_num += 1
      
      if iter_num == epoch_length:
        epo_loss = np.mean(losses[:, 0])
        epo_acc = np.mean(losses[:, 1])
				# curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
        iter_num = 0
        start_time = time.time()
        if epo_loss < best_loss:
          if C.verbose:
            print('Total loss decreased from {:.4f} to {:.4f}, saving model'.format(best_loss, loss))
          best_loss = epo_loss
          model_tuning.save(C.model_path)
        break
        
    except Exception as e:
      print('Exception: {}'.format(e))
      continue

print('Training complete, exiting.')

