###############################################################################
#                                  IMPORTS
###############################################################################

import os
import numpy as np

N_FOLDS = 5
PER_TEST = 1/N_FOLDS
RESIZE_PATH = os.path.join(os.getcwd(),'resize') 

if not os.path.exists(RESIZE_PATH):
    os.mkdir(RESIZE_PATH)

   
#%%###############################################################################
#                     DEFINE FOLDS (5-FOLD CROSS VALIDATION)
###############################################################################
farm = list(np.load(os.path.join(RESIZE_PATH, 'farm_info.npy'), allow_pickle=True))
farm.sort()

flickr = list(np.load(os.path.join(RESIZE_PATH, 'flickr_info.npy'),allow_pickle=True))
flickr.sort()

mixed = list(np.load(os.path.join(RESIZE_PATH, 'mixed_info.npy'),allow_pickle=True))
mixed.sort()

wsid = list(np.load(os.path.join(RESIZE_PATH, 'wsid_info.npy'),allow_pickle=True))
wsid.sort()

#Write txt file with all names
#%%%
files = [i for i in os.listdir(RESIZE_PATH) if i.endswith('.png')]
all_dataset_tests = []
for dataset in [farm, flickr, mixed, wsid]:
    n_frontal = 0
    n_left = 0
    n_right = 0
    
    for info in dataset:
        for j in info:  
            #Distribution of classes on the image
            if j[1] == 'frontal':
                n_frontal += 1
            if j[1] == 'left':
                n_left += 1
            if j[1] == 'right':
                n_right += 1
  
    test_size = int(0.10 * (n_frontal + n_left + n_right))
    all_test = []    
            
    for k in range(N_FOLDS):
        test = [] 
        n_f_test = int(n_frontal *PER_TEST)
        n_l_test = int(n_left * PER_TEST)
        n_r_test = int(n_right * PER_TEST)
        
        for info in dataset:
            #Distribution of classes on the image
            l = 0
            r = 0
            f = 0
            for j in info:
                if j[1] == 'frontal':
                    f += 1
                elif j[1] == 'left':
                    l += 1
                elif j[1] == 'right':
                    r += 1

            if f + l + r > 0 and n_f_test - f > -1 and n_l_test - l > -1 and n_r_test - r > -1:
                test.append(info)
                n_f_test = n_f_test - f
                n_l_test = n_l_test - l
                n_r_test = n_r_test - r

        for t in test:
            dataset.remove(t)

        all_test.append(test)
    all_dataset_tests.append(all_test)

file_path = os.path.join(os.getcwd(), 'cross_val', 'all.txt') 

with open(file_path, 'w') as f:
    for i in files:
        f.write("%s\n" % i)

#Write txt file with the test names
for j in range(N_FOLDS):
    fold = '%0d' % j
    file_path = os.path.join(os.getcwd(), 'cross_val', fold + '_test.txt') 
    with open(file_path, 'w') as f:
        for set_ in all_dataset_tests:
            fold = set_[j]
            for image in fold:
                for i in image:
                    f.write("%s\n" % image[0][0])
    

