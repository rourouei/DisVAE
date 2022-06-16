import cv2
import dlib
import numpy as np
import os
import random
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

IMG_JPGS = ['.jpg', '.jpeg', '.JPG', '.JPEG']
IMG_PNGS = ['.png', '.PNG']

NUMPY_EXTENSIONS = ['.npy', '.NPY']

data_dir = '/home/njuciairs/zmy/data'

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

ck_dir = data_dir + '/ck'
dirOutput = data_dir + '/ck_aug/ck'
try:
    os.makedirs(dirOutput)
except OSError:
    pass
def resize_ck():
    for root, _, files in os.walk(ck_dir):
        for file in files:
            if(is_image_file(file)):
                path = os.path.join(root, file)
                img = Image.open(path)
                out = img.resize((70, 70))
                outpath = dirOutput + '/' + root.split('/')[-2] + '/' + root.split('/')[-1]
                print(outpath)
                try:
                    os.makedirs(outpath)
                except OSError:
                    pass
                out.save(os.path.join(outpath, file))

# resize_ck()
# exit()

ck_dir = data_dir + '/ck_aug/ck_rdown'
dirOutput = data_dir + '/ck_aug/ck_rdownf'
try:
    os.makedirs(dirOutput)
except OSError:
    pass
def corner_crop():
    for root, _, files in os.walk(ck_dir):
        for file in files:
            if (is_image_file(file)):
                path = os.path.join(root, file)
                img = Image.open(path)
                out = img.transpose(Image.FLIP_LEFT_RIGHT)
                # out = img.crop((6, 6, 70, 70))
                outpath = dirOutput + '/' + root.split('/')[-2] + '/' + root.split('/')[-1]
                print(outpath)
                try:
                    os.makedirs(outpath)
                except OSError:
                    pass
                out.save(os.path.join(outpath, file))


# corner_crop()

def print_ck():
    for root, dirs, files in os.walk(ck_dir):
        array = dirs
        if array:
            return array

arry = print_ck()
print(len(arry))


# label_dir = data_dir +'/Emotion'
# def count_label():
#

def find_except():
    for root, _, files in os.walk(ck_dir):
        for file in files:
            if (is_image_file(file) == False):
                path = os.path.join(root, file)
                print(path)

# find_except()


def get_label(img_path):
    emo_dir = data_dir + '/Emotion/'
    label_dir = emo_dir + '/' + img_path.split('/')[-3]  + '/' + img_path.split('/')[-2]
    label = -1
    for root, _, files in os.walk(label_dir):
        if len(files) == 0:
            return -1
        for file in files:
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                label_str = f.read()
            label = float(label_str)
    return int(label)


dirpath = data_dir + '/ck_aug/ck_center'
def test_label():
    images = []
    labels = []
    for root, _, fnames in os.walk(dirpath):
        fnames.sort()
        for fname in fnames[-1:]:
            img = os.path.join(root, fname)
            images.append(img)
            # print(img)
            labels.append(get_label(img))
    # print(labels)
    label_ = []
    for l in labels:
        if l != -1:
            label_.append(l-1)
    print(len(labels))
    print(len(label_))

    count = [0,0,0,0,0,0,0]
    for i in label_:
        print(i)
        count[i] += 1
    print(count)


# print(test_label())


# data_dir = '/home/njuciairs/zmy/data'
# oulu = data_dir + '/ck_aug'
# oulu_dirs = []
# for path in os.listdir(oulu):
#     oulu_dirs.append(oulu+'/'+path)
#
# angles = [-15, -12, -9, -6, -3, 3, 6, 9, 12, 15]
#
# for oulu_dir in oulu_dirs:
#     for angle in angles:
#         dirOutput = oulu_dir + str(angle)
#         # print(dirOutput)
#         try:
#             os.makedirs(dirOutput)
#         except OSError:
#             pass
#         for root, _, files in os.walk(oulu_dir):
#             for file in files:
#                 path = os.path.join(root, file)
#                 img = Image.open(path)
#                 out = img.rotate(angle)
#
#                 outpath = dirOutput + '/' + root.split('/')[-2] + '/' + root.split('/')[-1]
#                 try:
#                     os.makedirs(outpath)
#                 except OSError:
#                     pass
#                 out.save(os.path.join(outpath, file))