import tarfile
import os
import numpy as np
from PIL import Image
import random

import keras
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input

def load_imgs(dset="train"):
    global data_dir, num_imgs, ratio_train, w, h
    imgs = np.zeros((num_imgs, w, h, 3))
    if dset=="train":
        index = range(1, 1+int(num_imgs*ratio_train))
    else:
        index = range(1+int(num_imgs*ratio_train),num_imgs)
    for i in index:
        try:
            fname = '{:03d}.png'.format(i)
            fpath = os.path.join(data_dir, fname)
            img = img_process(fpath)
        except Exception:
            fname = '{:03d}.jpg'.format(i)
            fpath = os.path.join(data_dir, fname)
            img = img_process(fpath)
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

