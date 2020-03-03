#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:13:50 2019

@author: franciscapessanha
"""
###############################################################################
#                                  IMPORTS
###############################################################################

import os
import numpy as np
import cv2 as cv
import copy


DATASET_PATH = '/' + os.path.join(*os.getcwd().split('/')[:-1],'dataset', 'pose_lms_info') 
N_FOLDS = 5
PER_TEST = 1/N_FOLDS

def flip_sheep(image, points, ocs=[]):
    h, w = image.shape[:2]
    new_image = cv.flip(image, 1)
    new_points = []
    new_points.append(points[5])
    new_points.append(points[4])
    new_points.append(points[3])
    new_points.append(points[2])
    new_points.append(points[1])
    new_points.append(points[0])
    new_points.append(points[6])
    new_points.append(points[7])
    new_points.append(points[8])
    new_points.append(points[10])
    new_points.append(points[9])
    new_points.append(points[12])
    new_points.append(points[11])
    new_points.append(points[14])
    new_points.append(points[13])
    new_points.append(points[16])
    new_points.append(points[15])
    new_points.append(points[18])
    new_points.append(points[17])
    new_points.append(points[20])
    new_points.append(points[19])
    new_points.append(points[23])
    new_points.append(points[24])
    new_points.append(points[21])
    new_points.append(points[22])
    for i, p in enumerate(new_points):
        new_points[i] = [w - p[0], p[1]]
   
    # draw('initial', image, points, 0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # draw('flipped', new_image, new_points, 0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    n_ocs = []
    n_ocs.append(5) if 0 in ocs else ()
    n_ocs.append(0) if 5 in ocs else ()
    n_ocs.append(4) if 1 in ocs else ()
    n_ocs.append(1) if 4 in ocs else ()
    n_ocs.append(3) if 2 in ocs else ()
    n_ocs.append(2) if 3 in ocs else ()
    n_ocs.append(10) if 9 in ocs else ()
    n_ocs.append(9) if 10 in ocs else ()
    n_ocs.append(12) if 11 in ocs else ()
    n_ocs.append(11) if 12 in ocs else ()
    n_ocs.append(14) if 13 in ocs else ()
    n_ocs.append(13) if 14 in ocs else ()
    n_ocs.append(16) if 15 in ocs else ()
    n_ocs.append(15) if 16 in ocs else ()
    n_ocs.append(18) if 17 in ocs else ()
    n_ocs.append(17) if 18 in ocs else ()
    n_ocs.append(20) if 19 in ocs else ()
    n_ocs.append(19) if 20 in ocs else ()
    n_ocs.append(22) if 21 in ocs else ()
    n_ocs.append(21) if 22 in ocs else ()
    n_ocs.append(24) if 23 in ocs else ()
    n_ocs.append(23) if 24 in ocs else ()
    n_ocs.append(6) if 6 in ocs else ()
    n_ocs.append(7) if 7 in ocs else ()
    n_ocs.append(8) if 8 in ocs else ()
    new_points = np.array(new_points)
    
    
    return new_image, new_points, n_ocs

#=============================================================================
def crop_img(im, lms, out_path):
    lms = np.asarray(lms).astype(np.float64)
    x, y, w, h = [np.min(lms[:, 0]),
                  np.min(lms[:, 1]),
                  np.max(lms[:, 0]) - np.min(lms[:, 0]),
                  np.max(lms[:, 1]) - np.min(lms[:, 1])]                    
    cx, cy = (x + w / 2, y + h /2)
    s = max(w, h)
    
    #ew = 0.15
    #eh = 0.15
    
    ew = 0.05
    eh = 0.05
    #left: xmin top: ymin right: xmax bottom: ymax
    img_h, img_w = im.shape[:2]
    l = int(max(int(cx - s / 2)  - w * ew, 0))
    t = int(max(int(cy - s / 2) - h * eh, 0))
    r = int(min(int(cx + s / 2) + h * eh, img_w))
    b = int(min(int(cy + s / 2)  + w * ew, img_h))
    im_sq = im[t :b, l:r]
    sq_h, sq_w = im_sq.shape[:2]
    im_sq = cv.resize(im_sq, (500, 500))
    sq_lms = []
    
    #image_o = im_sq.copy()
    for pt in lms:
        # have into account the cropped and the resize!
        new_pt = (int((pt[0] - l) * 500/sq_w), int((pt[1] - t) * 500/sq_h))
        sq_lms.append(new_pt)
        r = 5
        cv.circle(im_sq, new_pt, r, (255, 0, 0), -1)
    
    out_path = os.path.join(out_path.split('/')[0], 'gt_' + out_path.split('/')[-1])
    cv.imwrite(out_path, im_sq)
    """ 
    cv.imshow('bboxes', image_o)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
    
    return sq_lms


def create_pts(lms, ocs, pts_path):
    with open(pts_path, 'w') as pts_file:
        pts_file.write('version: 1\n')
        pts_file.write('n_points:  ' + str(len(lms)) + '\n')
        pts_file.write('{\n')
        for (x, y) in [lms[i] for i in range(len(lms)) if i not in ocs]:
            pts_file.write(str(int(x)) + ' ' + str(int(y)) + '\n')
        pts_file.write('}')
    
            
#%%############################################################################
#                               INITIALIZE DATASETS
###############################################################################
farm = np.load(os.path.join(DATASET_PATH, 'farm.npy'), allow_pickle=True).item()

flickr = np.load(os.path.join(DATASET_PATH, 'flickr.npy'),allow_pickle=True).item()

mixed = np.load(os.path.join(DATASET_PATH, 'mixed.npy'),allow_pickle=True).item()

wsid = np.load(os.path.join(DATASET_PATH, 'wsid.npy'),allow_pickle=True).item()


if not os.path.exists('absolute_yaw'):
    os.mkdir('absolute_yaw')
    
path_0_30 = os.path.join('absolute_yaw_copy', '0_30')
path_30_60 = os.path.join('absolute_yaw_copy', '30_60')
path_60_90 = os.path.join('absolute_yaw_copy','60_90')

ocs_30 = []
ocs_60 = []
ocs_90 = []

if not os.path.exists(path_0_30):
    os.makedirs(path_0_30)
if not os.path.exists(path_30_60):
    os.makedirs(path_30_60)
if not os.path.exists(path_60_90):
    os.makedirs(path_60_90)
    
for dataset in [farm, flickr, mixed, wsid]:
    for d in range(len(dataset)):
        (im_path, _, lms, ocs, (roll, pitch, yaw)) = dataset[d]
        im_path = '/' + os.path.join(*os.getcwd().split('/')[:-1],im_path)
        im = cv.imread(im_path)
        if yaw < 0:
            im, lms, ocs = flip_sheep(im, lms, ocs)
            yaw = abs(yaw)
        
        
        if yaw < 0:
            print("Error: Invalid yaw (< 0)")
        if yaw < 30:
            out_path = os.path.join(path_0_30, str(d) + '_' + im_path.split('/')[-1])
            ocs_30.append(ocs)
        elif yaw < 60:
            out_path = os.path.join(path_30_60, str(d) + '_' + im_path.split('/')[-1])
            ocs_60.append(ocs)
        elif yaw < 90:
            out_path = os.path.join(path_60_90, str(d) + '_' + im_path.split('/')[-1])
            ocs_90.append(ocs)
        else:
            print("Error: Invalid yaw (> 90)")
            
        lms = crop_img(im,lms, out_path)
         
        pts_path = out_path.replace('png','pts')
        if yaw < 30: 
            ocs = []
        elif yaw < 60: 
            ocs = []
            #ocs = [2,11,13,15,19]
        elif yaw < 90:
            ocs = []
            #ocs = [0, 1, 2,9,11,13,15,17,19, 21, 22]
        create_pts(lms, ocs, pts_path)
 
#%%############################################################################
#                                 LANDMARK DISTRIBUTION
###############################################################################
import matplotlib.pyplot as plt
counts_30, bins_30, _ = plt.hist(np.hstack(ocs_30), bins=25, range=(0,24))  
counts_60, bins_60, _ = plt.hist(np.hstack(ocs_60), bins=25, range=(0,24)) 
counts_90, bins_90, _ = plt.hist(np.hstack(ocs_90), bins=25, range=(0,24))      