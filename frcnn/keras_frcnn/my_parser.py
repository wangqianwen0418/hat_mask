import cv2
import numpy as np
import os

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	if img.shape[2]!=3:
		img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

def load_imgs(data_dir, mode="train"):
    imgs = []
    for idx, img_name in enumerate(sorted(os.listdir(data_dir))):
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        img_path = os.path.join(data_dir, img_name)
        if mode=="train" and idx<400:
            imgs.append(img_path)
        elif mode == "val" and idx>399:
            imgs.append(img_path)
        elif mode =="all":
            imgs.append(img_path)
    return imgs

def load_labels(data_dir, mode="train", target="no_hat"):
    fpath = os.path.join(data_dir, "labels.txt")
    names = ["index", "total", "no_cloth", "no_mask", "no_hat"]
    labels = np.zeros(500)
    with open(fpath,"r") as f:
        for line in f:
            line = line.split()
            # print(line)
            label_i = names.index(target)
            label = line[label_i]
            labels[int(line[0])-1] = int(int(label)>0)
    if mode =="train":
        return labels[0: 400]  
    elif mode == "all":
        return labels
    else:
        return labels[400:500]

def get_data(data_dir, mode = "train", target="no_hat"):
    return load_imgs(data_dir, mode), load_labels(data_dir, mode, target)

def my_generator(all_img_data, all_label, C):
    while True:
        # if (mode == "train"):
        #     np.random.shuffle(all_img_data)
        for i, img_path in enumerate(all_img_data):
            img = cv2.imread(img_path)
            X, ratio = format_img(img, C)
            yield np.copy(X), np.array([all_label[i]])



