#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
import sys
import random
import tensorflow as tf

# proj_home_path="C:/vrsk.psk.family/Selva/BitsPilani/azure/semester4/semester4/SuperPoint"
# sys.path.append(proj_home_path)

from superpoint.settings import EXPER_PATH
# from utils import plot_imgs
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

def _read_image(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    return tf.cast(image, tf.float32)

# Python function
def _read_points(filename):
    return np.load(filename.decode('utf-8')).astype(np.float32)

def draw_keypoints(img, corners, color):
    keypoints = [cv2.KeyPoint(c[1], c[0], 1) for c in corners]
    return cv2.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)
def display(d):
    return draw_keypoints(d['image'], d['keypoints'], (0, 255, 0))

def getDataIter(shape_dir_list,num_images_per_shape,data_dir, img_dir,p_dir):
    '''
        shape_dir_list : directory name that holds specific type of shapes
        num_images_per_shape: number of images randomly picked from the shapes directory
        data_dir: base directory for both images and corner points
        idir: full directory path for specifc shape generation
        pdir: full directory path for specifc corner points generation
    '''
    #================= LIST RANDOM FILES =============================
    ifiles =  []
    pfiles = []
 
    # Randomly pick num_images_per_shape count of images
    for sdir in shape_dir_list:
        idir = data_dir+sdir+img_dir
        pdir = data_dir+sdir+p_dir
        ifiles_list = [f for f in os.listdir(idir)]
        random_ifiles = np.random.choice(ifiles_list, num_images_per_shape)
        random_pfiles = [ f.replace(".png",".npy") for f in random_ifiles]
        random_ifiles = [os.path.join(idir, f) for f in random_ifiles]
        random_pfiles = [os.path.join(pdir, f) for f in random_pfiles]
            
        # Accumulate images in list for later operations.
        if ifiles == None:
            ifiles = random_ifiles
            pfiles = random_pfiles
        else:    
            ifiles = ifiles + random_ifiles
            pfiles = pfiles + random_pfiles

    #================ READ LISTED FILES ===========================
    data = tf.data.Dataset.from_tensor_slices(
            (ifiles, pfiles))
    data = data.map(
            lambda image, points:
            (_read_image(image), tf.py_func(_read_points, [points], tf.float32)))
    data = data.map(lambda image, points: (image, tf.reshape(points, [-1, 2])))
    data = data.map(lambda image, kp: {'image': image, 'keypoints': kp})
    tf_next = data.make_one_shot_iterator().get_next()
    sess = tf.Session()
    while True:
        yield sess.run(tf_next)