###############################################################################
#                                  IMPORTS
###############################################################################

import os
import numpy as np
from sklearn.model_selection import KFold
from aug_data import aug_data
import cv2 as cv
import matplotlib.pyplot as plt
import math
import copy

DATASET_PATH = '/' + os.path.join(*os.getcwd().split('/')[:-1],'dataset', 'pose_lms_info') 
N_FOLDS = 5
PER_TEST = 1/N_FOLDS

###############################################################################
#                               INITIALIZE DATASETS
###############################################################################
farm = np.load(os.path.join(DATASET_PATH, 'farm.npy'), allow_pickle=True).item()

flickr = np.load(os.path.join(DATASET_PATH, 'flickr.npy'),allow_pickle=True).item()

mixed = np.load(os.path.join(DATASET_PATH, 'mixed.npy'),allow_pickle=True).item()

wsid = np.load(os.path.join(DATASET_PATH, 'wsid.npy'),allow_pickle=True).item()

#Info about images from each range on the subset 
intervals = []
for i in range(0,90,6):
    intervals.append([i, i + 6])

all_dataset_tests = [] # all folds test group

alpha = 0.6
###############################################################################
#                       DEFINE FOLDS (5-FOLD CROSS VALIDATION)
###############################################################################
def get_cross_val():
    
    if not os.path.exists('cross_val'):
        os.makedirs('cross_val')
    
    info_by_angle = [[]] * len(intervals)
    x = 0
    for dataset in [farm, flickr, mixed, wsid]:
        for d in dataset:
            (im_path, bbox, lms, ocs, (roll, pitch, yaw)) = dataset[d]
            for count,i in enumerate(intervals):
                if abs(int(yaw)) in range(*i):
                    x += 1
                    info_by_angle[count] = [*info_by_angle[count],((im_path, bbox, lms, ocs, roll, pitch, yaw))]
                    """
                    print('yaw: ', yaw)
                    print('interval: ', i)
                    print('count: ', count)
                    """
    
    kf = KFold(n_splits=N_FOLDS)
    
    all_train_sets = []
    all_test_sets = []
    
    subsets = info_by_angle
    for subset in subsets:
        subset = np.asarray(subset)
        train = []
        test = []
        for train_index, test_index in kf.split(subset):
            X_train, X_test = subset[train_index], subset[test_index]
            test.append(X_test)
            train.append(X_train)
        
        all_train_sets.append(train)
        all_test_sets.append(test)
    
    train = []
    test = []
    
    # Divide between train and test - test group won't be augmentated
    for n in range(N_FOLDS): 
        train_fold = []
        test_fold = []
        for i in range((len(subsets))):
            train_fold.append(all_train_sets[i][n])
            test_fold.append(all_test_sets[i][n])
            
        train.append(np.vstack(train_fold))
        test.append(np.vstack(test_fold))
        
        test_path = os.path.join('cross_val', 'pose_info_test_' + str(n)) 
        train_path = os.path.join('cross_val', 'pose_info_train_' + str(n)) 
        
        np.save(train_path, train[n])
        np.save(test_path, test[n])
    
return train, test    

#=============================================================================
def crop_img(im, box, lms, out_path):
    new_lms = copy.copy(lms)
    new_box = copy.copy(box)
    
    x,y,w,h = new_box
    cx, cy = (x + w / 2, y + h /2)
    s = max(w, h)
        
    ew = 0.15
    eh = 0.15
        
    #left: xmin top: ymin right: xmax bottom: ymax
    img_h, img_w = im.shape[:2]
    l = int(max(int(cx - s / 2)  - w * ew, 0))
    t = int(max(int(cy - s / 2) - h * eh, 0))
    r = int(min(int(cx + s / 2) + h * eh, img_w))
    b = int(min(int(cy + s / 2)  + w * ew, img_h))
    
    im_sq = im[t :b, l:r]
    
    
    for lm in new_lms:
        lm[0] -= l
        lm[1] -= t
    x -= l
    y -= t
    new_box = (x,y,w,h)
    # resize to square
    ih, iw = im_sq.shape[:2]
    # resize output image (some are very big) and fix lms
    s = 500 / max(iw, ih)
    im_sq = cv.resize(im_sq, (int(s * iw), int(s * ih)))
    new_lms = [x * s for x in new_lms]
    
    new_box = copy.copy(box)
    new_box = [int(x * s) for x in new_box]
    
    
    """
    #TO DEBUG
    #=========
    image_o = im_sq.copy()
    for pt in new_lms:
        print(pt)
        r = 2
        point = tuple(pt.astype(int))
        cv.circle(image_o, point, r, (255, 0, 0), -1)
    cv.imshow('image from the crop', image_o)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
    cv.imwrite(out_path, im_sq)
    

    return new_box, new_lms

train, test = get_cross_val() 

###############################################################################
#         SAVE FOLD - SAVE IMAGES AND INFORMATION OF A FOLD FOR TRAINING
###############################################################################

#def save_folds(train, test, folds):
#train, test = get_cross_val()  
#def save_folds(train, test, folds):

folds = [0,1,2,3,4]
for f in folds:
    fold = '%0d' % f 
    test_path = os.path.join('cross_val', 'pose_info_test_aug_' + str(fold)) 
    train_path = os.path.join('cross_val', 'pose_info_train_aug_' + str(fold)) 

    train_folder = os.path.join('cross_val', str(fold) + '_train')
    test_folder = os.path.join('cross_val', str(fold) + '_test')
    
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        
    # Augment train group 
    #print(train[f][i])
    #print(len(train[f]))
    all_yaws = [np.abs(train[f][i][6]) for i in range(len(train[f]))]
    yaw_counts, yaw_bins, _ = plt.hist(all_yaws, bins=15, range=(0,90))
    print('FOLD: ', f)
    print('yaw_counts: ', yaw_counts)
    print('yaw_bins: ', yaw_counts)
    cn = yaw_counts.max()
    yaw_mod = [np.ceil((cn // y) ** alpha) for y in yaw_counts]
    for i,y in enumerate(yaw_mod):
      if math.isnan(y):
        yaw_mod[i] = 0  
    print('yaw mod: ', yaw_mod)
    eff_n = np.sum([yaw_mod[i] * yaw_counts[i] for i in range(len(yaw_counts))])
    
    
    print('Dataset size: ', int(2 * eff_n))
    print('Effective aug:', 2 * eff_n / len(train[f]))
    print('Aug Max: ', 2 * np.max(yaw_mod))
    print('\n')
  
    index_list = list(range(len(train[f])))
    