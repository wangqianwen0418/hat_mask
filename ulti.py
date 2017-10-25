import tarfile
import os
import numpy as np
from PIL import Image
import random

import keras
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input

from frcnn.test_frcnn import format_img

def pretrained_model():
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

    return model_rpn, model_classifier
	

def load_imgs(dset="train"):
    global data_dir, num_imgs, ratio_train, w, h
    model_rpn, model_classifier = pretrained_model()
    if dset=="train":
        index = range(1, 1+int(num_imgs*ratio_train))
    else:
        index = range(1+int(num_imgs*ratio_train),num_imgs)
    imgs = np.zeros((len(index), 32, 2048))
    for i, v in enumerate(index):
        try:
            fname = '{:03d}.png'.format(v)
            fpath = os.path.join(data_dir, fname)
            img = img_process(fpath)
        except Exception:
            fname = '{:03d}.jpg'.format(v)
            fpath = os.path.join(data_dir, fname)
            img = img_process(fpath, model_rpn, model_classifier)
        imgs[i] = img
    return imgs

def load_labels(dset="train", target="no_mask"):
    global data_dir,num_imgs, ratio_train
    fpath = os.path.join(data_dir, "labels.txt")
    names = ["index", "total", "no_cloth", "no_mask", "no_hat"]
    labels = np.zeros((num_imgs))
    with open(fpath,"r") as f:
        for line in f:
            line = line.split()
            # print(line)
            label_i = names.index(target)
            label = line[label_i]
            labels[int(line[0])-1] = bool(int(label)>0)
    if dset =="train":
        return labels[0: int(num_imgs*ratio_train)]  
    else:
        return labels[int(num_imgs*ratio_train): num_imgs-1]




def img_process(fpath):
    global w, h
    img = Image.open(fpath)
    _PIL_INTERPOLATION_METHODS = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'hamming': Image.HAMMING,
        'box': Image.BOX,
        "lanczos": Image.LANCZOS
    }
    
    if (img.mode!='RGB'):
        img = img.convert("RGB")
    # resize and corp
    if img.size != (w, h):
        short_edg = min(img.size)
        resample = _PIL_INTERPOLATION_METHODS['bilinear']
        resized = img.resize((w, h), resample)
        # xx = random.randint(0, scaled_w - w)
        # yy = random.randint(0, scaled_h - h)
        # crop = resized.crop((xx, yy, w+xx, h+yy))
    else: 
        resized = img

    # resized.show()

    arr = np.asarray(resized)
    arr = arr.astype(dtype=K.floatx())
    arr = np.expand_dims(arr, axis=0)
    # arr = preprocess_input(arr, data_format='channels_last', mode='tf')
    ## zero-center between [-1, 1]
    arr /=255
    arr -= 0.5
    arr *=2
    return arr

def load_data(dset="train", target="no_mask"):
    return load_imgs(dset), load_labels(dset, target)


if __name__=="__main__":
    data_dir = "data"
    num_imgs = 500
    ratio_train = 0.8
    w = 224
    h = 224

    img_process("data/001.png")

    # imgs, labels = load_data()
    # print(labels[1:5])

