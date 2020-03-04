

#%%
###############################################################################
#                                   IMPORTS
###############################################################################

import os
import shutil
DATASET_PATH = os.path.join(os.getcwd(), 'resize') 

#%%
###############################################################################
#                        GET SUBSETS - TRAIN AND TEST
###############################################################################

def get_fold(folds):
    
    for fold in folds:
        n_fold = str(fold)
        test = n_fold + '_test.txt'
           
        test_path = os.path.join(os.getcwd(),'cross_val',test)
        all_path = os.path.join(os.getcwd(),'cross_val','all.txt')
        
        test = open(test_path,'r').readlines()
        all_ = open(all_path,'r').readlines()
        
        test_folder = os.path.join(os.getcwd(),'data_' + n_fold , 'images', 'test')
        train_folder = os.path.join(os.getcwd(),'data_' + n_fold, 'images', 'train')
        
        # Create directory if it doesn't exist
        
        if not os.path.exists( os.path.join(os.getcwd(),'data_' + n_fold)):
            os.mkdir( os.path.join(os.getcwd(),'data_' + n_fold))
            
        if not os.path.exists( os.path.join(os.getcwd(),'data_' + n_fold, 'images')):
            os.mkdir( os.path.join(os.getcwd(),'data_' + n_fold, 'images'))

        if not os.path.exists( os.path.join(os.getcwd(),'data_' + n_fold, 'annotations')):
            os.mkdir( os.path.join(os.getcwd(),'data_' + n_fold, 'annotations'))
            
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)
        if not os.path.exists(train_folder):
            os.mkdir(train_folder)
            
        
        for image in all_:
            image = image.strip() #remove \n from string
            dataset = image.split('_')[0] 
            if dataset == 'internet':
                dataset = 'wsid'
            img_path = os.path.join(DATASET_PATH,image)
            
            in_test = False
            for t in test:
                if t.strip() in image:
                    npy_path = img_path.replace('png','npy')
                    shutil.copy2(img_path, os.path.join(test_folder, image))
                    shutil.copy2(img_path, os.path.join(test_folder, image))
                    shutil.copy2(npy_path, os.path.join(test_folder, image.replace('png','npy')))
                    in_test = True
                    break
                
            if not in_test:
                npy_path = img_path.replace('png','npy')
                shutil.copy2(img_path, os.path.join(train_folder, image))
                shutil.copy2(npy_path, os.path.join(train_folder, image.replace('png','npy')))
        

get_fold([0,1,2,3,4])     