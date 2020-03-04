#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
###############################################################################
#                                  IMPORTS
###############################################################################

import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
import cv2 as cv

DATASET_PATH = '/' + os.path.join(*os.getcwd().split('/')[:-1],'dataset', 'data') 
RESIZE_PATH = os.path.join(os.getcwd(),'resize') 

if not os.path.exists(RESIZE_PATH):
    os.mkdir(RESIZE_PATH)

###############################################################################
#                       RESIZE EVERY IMAGE AND DEFINE MIN BB
###############################################################################

def resize_image(image, h, w, max_side = 300):
    new_width = w
    new_height = h
    if h > w and h != max_side:
        new_height = max_side
        new_width = int(w * (new_height/h))
    elif w > h and w != max_side:
        new_width = max_side
        new_height = int(h * (new_width/w))
    
    resized_image = cv.resize(image, (new_width, new_height))
    return resized_image

def validate_and_resize_bounding_box(h, w, xmin, ymin, xmax, ymax, max_side = 300):
    new_width = w
    new_height = h
    if h > w and int(h) != max_side:
        new_height = max_side
        new_width = int(w * (new_height/h))
    elif w > h and int(w) != max_side:
        new_width = max_side
        new_height = int(h * (new_width/w))


    xmin = 0 if xmin < 0 else int((xmin * new_width)/w)
    ymin = 0 if ymin < 0 else int((ymin * new_height)/h)
    xmax = new_width if xmax > w else int((xmax * new_width)/w)
    ymax = new_height if ymax > h else int((ymax * new_height)/h)

    
    if ((xmax - xmin) * (ymax - ymin)) < (new_height * new_width *  0.03):
        return None, None, xmin, ymin, xmax, ymax
    
    return new_height, new_width, xmin, ymin, xmax, ymax

#%%
###############################################################################
#                                  CONVERT XML TO NPY
###############################################################################

datasets = ['farm', 'flickr', 'mixed', 'wsid'] 
for dataset in datasets:
    path= os.path.join(DATASET_PATH, dataset)
    
    classes_names = []
    xml_list = []
    all_npy = []
    
    n_frontal = 0
    n_right = 0
    n_left = 0
    
    for i, xml_file in enumerate(glob.glob(path + "/*.xml")):
        image_npy = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        found_obj = False
        for j, member in enumerate(root.findall('object')):
            found_obj = True
            filename = root.find('filename').text
            class_ = member.find('name').text
            
            img_path = os.path.join(DATASET_PATH,dataset, filename)
            img = cv.imread(img_path)
            
            if class_ == 'right':
                n_right += 1
            elif class_ == 'left':
                n_left += 1
            elif class_ == 'frontal':
                n_frontal += 1
                
            width = int(root.find('size').find('width').text)
            height = int(root.find("size").find('height').text)
            xmlbox = member.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            ymin = int(xmlbox.find('ymin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymax = int(xmlbox.find('ymax').text)
            
            resized_img = resize_image(img, height, width)                          
            n_height,n_width, n_xmin, n_ymin, n_xmax, n_ymax =  validate_and_resize_bounding_box(height, width, xmin, ymin, xmax, ymax)
            
            if n_height is not None:
                image_npy.append((filename,class_,n_width,n_height, (n_xmin,n_ymin,n_xmax,n_ymax))) #name, label, width, height, bbox
        if len(image_npy) != 0:
            np.save(os.path.join(RESIZE_PATH, filename.replace('png','npy')), image_npy)
            cv.imwrite(os.path.join(RESIZE_PATH, filename), resized_img)
            all_npy.append(image_npy)
           
    np.save(os.path.join(RESIZE_PATH,dataset + '_info.npy'), np.asarray(all_npy))
