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

oulu = data_dir + '/Oulu_aug'
oulu_dirs = []
for path in os.listdir(oulu):
    oulu_dirs.append(oulu+'/'+path)

angles = [-15, -12, -9, -6, -3, 3, 6, 9, 12, 15]

for oulu_dir in oulu_dirs:
    for angle in angles:
        dirOutput = oulu_dir + str(angle)
        # print(dirOutput)
        try:
            os.makedirs(dirOutput)
        except OSError:
            pass
        for root, _, files in os.walk(oulu_dir):
            for file in files:
                path = os.path.join(root, file)
                img = Image.open(path)
                out = img.rotate(angle)

                outpath = dirOutput + '/' + root.split('/')[-2] + '/' + root.split('/')[-1]
                try:
                    os.makedirs(outpath)
                except OSError:
                    pass
                out.save(os.path.join(outpath, file))