#%%
##############################################################################
#                                  IMPORTS
###############################################################################
import os
import cv2 as cv
import numpy as np
from HOGs_features import get_HOG_features
from geometric_features import get_geometric_features
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import shutil

DATASET_PATH = '/' + os.path.join(*os.getcwd().split('/')[:-1],'dataset')
FINAL_FOLDER = os.path.join(DATASET_PATH, 'scored','first_frames')
ABS_FOLDER = os.path.join(os.getcwd(), 'abs_yaw_pain')
if not os.path.exists(os.path.join(ABS_FOLDER)):
    os.makedirs(ABS_FOLDER)

FOLDER_10 = os.path.join(ABS_FOLDER, '0_10')
if not os.path.exists(os.path.join(FOLDER_10)):
    os.makedirs(FOLDER_10)

FOLDER_60 = os.path.join(ABS_FOLDER, '10_60')
if not os.path.exists(os.path.join(FOLDER_60)):
    os.makedirs(FOLDER_60)

FOLDER_90 = os.path.join(ABS_FOLDER, '60_90')
if not os.path.exists(os.path.join(FOLDER_90)):
    os.makedirs(FOLDER_90)

            
info_10 = []
info_60 = []
info_90 = []

all_precision = []
all_recall = []
all_f1 = [] 

PAIN_THRESH = 0.5
N_FOLDS = 5
N_N_FOLDS = 4 #nested
N_COMP = 65

#%%
##############################################################################
#                               UPDATE LANDMARKS
###############################################################################
def update_lms(points, yaw):

    if abs(yaw) < 10:
        lms = points 
    elif abs(yaw) < 60:
        if yaw < 0:
            lms = []
            lms.append(points[0])
            lms.append(points[1]) 
            lms.append((0,0))
            lms.append(points[2]) 
            lms.append(points[3])
            lms.append(points[4])
            lms.append(points[5])
            lms.append(points[6])
            lms.append(points[7])
            lms.append(points[8])
            lms.append(points[9])
            lms.append((0,0))
            lms.append(points[10])
            lms.append((0,0))
            lms.append(points[11])
            lms.append((0,0))
            lms.append(points[12])
            lms.append(points[13])
            lms.append(points[14])            
            lms.append((0,0))
            lms.append(points[15])
            lms.append(points[16]) 
            lms.append(points[17])
            lms.append(points[18]) 
            lms.append(points[19])
            
        else:
            ocs = [2, 11, 13, 15, 19]
            lms = [(0,0) for i in range(25)]
            j = 0
            for i in range(len(lms)):
                if i not in ocs:
                    lms[i] = tuple(points[j])
                    j +=1

    elif abs(yaw) < 90:
        if yaw < 0:
            lms = []
            lms.append((0,0))
            lms.append((0,0))
            lms.append((0,0))
            lms.append(points[0]) 
            lms.append(points[1])
            lms.append(points[2])
            lms.append(points[3])
            lms.append(points[4])
            lms.append(points[5])
            lms.append((0,0))
            lms.append(points[6])
            lms.append((0,0))
            lms.append(points[7])
            lms.append((0,0))
            lms.append(points[8])
            lms.append((0,0))
            lms.append(points[9])
            lms.append((0,0))
            lms.append(points[10])           
            lms.append((0,0))
            lms.append(points[11])
            lms.append((0,0))
            lms.append((0,0))
            lms.append(points[12])
            lms.append(points[13])
            
        else:
            ocs = [0, 1, 2, 9, 11, 13, 15, 17, 19, 21, 22]
            lms = [(0,0) for i in range(25)]
            j = 0
            for i in range(len(lms)):
                if i not in ocs:
                    lms[i] = tuple(points[j])
                    j +=1

    return lms
            
#%%
###############################################################################
#                                  FLIP IMAGE
###############################################################################

def flip_sheep(image,points):                
    h, w = image.shape[:2]
    new_image = cv.flip(image, 1)
    for i, p in enumerate(points):
        points[i] = [w - p[0], p[1]]
    
    return new_image, points


#%%############################################################################
#                               CROSS VALIDATION
###############################################################################

def cross_validation(info):
    folders = ['ears', 'eyes', 'nose', 'cheek']
    all_X_train = []
    all_X_test = []
    all_y_train = []
    all_y_test = []
    
    skf = StratifiedKFold(n_splits=N_FOLDS)
    label = []
    for i in info[:,0:5]:
        comp_folders = []
        path,_,_,_, av_score = i
        #print(av_score)
        
        if av_score >= PAIN_THRESH:
            label.append(1)
        else:
            label.append(0)
            #print('path: ', path.split('/')[-1], ' - ', 0)
        
        folder_path = os.path.join('/', *path.split('/')[0:-1])
        name = path.split('/')[-1]
        pain_folder = os.path.join(folder_path, 'pain')
        no_pain_folder = os.path.join(folder_path, 'no_pain')
        
        comp_folders.append(folder_path)
        for folder in folders:
            comp_folders.append(os.path.join(folder_path, folder))
    
        #print('comp folders: ', comp_folders)
        for folder in comp_folders:
            for file in os.listdir(folder):
                if name in file:
                    if label[-1] == 1: # pain
                        shutil.copy(os.path.join(folder,file), os.path.join(pain_folder, file))
                    elif label[-1] == 0: # no pain
                        shutil.copy(os.path.join(folder,file), os.path.join(no_pain_folder, file))
                
            
         
    info = np.asarray(info)
    label = np.asarray(label)
    for train_index, test_index in skf.split(info, label):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = info[train_index], info[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        all_X_train.append(X_train)
        all_X_test.append(X_test)
        all_y_train.append(y_train)
        all_y_test.append(y_test)
        
    return all_X_train, all_y_train, all_X_test,all_y_test
        
          
def nested_cross_val(x, y):
    skf = StratifiedKFold(n_splits=N_N_FOLDS)
    all_X_train = []
    all_X_test = []
    all_y_train = []
    all_y_test = []
    
    for train_index, test_index in skf.split(x, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        all_X_train.append(X_train)
        all_X_test.append(X_test)
        all_y_train.append(y_train)
        all_y_test.append(y_test)
        
    return all_X_train, all_y_train, all_X_test,all_y_test


#%%
###############################################################################
#                              FEATURE EXTRACTION
###############################################################################   

def get_features(info):
    all_HOG = []
    all_geometry = []
    all_pose = []
    all_ratio_ears = []
    all_ratio_nose = []
    for x in info:
        new_path, lms, pose, ocs, average_score = x
        roll, pitch, yaw = pose
        img = cv.imread(new_path)
        image_name = new_path.split('/')[-1]
        HOG, ratio_ears, ratio_nose = get_HOG_features(image_name,img,lms, yaw) 
        geometry = get_geometric_features(lms, yaw)

        all_HOG.append(np.hstack(HOG))
        all_geometry.append(geometry)
        all_pose.append(pose)
        all_ratio_ears.append(ratio_ears)
        all_ratio_nose.append(ratio_nose)

    #print('EARS: %0.2f (+/-) %0.2f' % (np.mean(all_ratio_ears), np.std(all_ratio_ears)))
    #print('NOSE: %0.2f (+/-) %0.2f' % (np.mean(all_ratio_nose), np.std(all_ratio_nose)))
    return all_HOG, all_geometry, all_pose


#%%
###############################################################################
#                             TRAIN MODEL
###############################################################################
          
def train_model(all_x_train, all_x_test, all_y_train, all_y_test, FOLDS):
    all_precision = []
    all_recall = [] 
    all_f1 = [] 
    all_pred = []
    all_acc = []
    
    for f in range(FOLDS):
        x_train = all_x_train[f]
        x_test = all_x_test[f]
    
        y_train = all_y_train[f]
        y_test = all_y_test[f]
        
        modelSVM = SVC(kernel = 'linear', gamma = 'auto', decision_function_shape= 'ovr',class_weight='balanced')
        modelSVM.fit(x_train, y_train)
        
        y_pred= modelSVM.predict(x_test)
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    
        if TP + FP == 0:
            precision = 0
            
        else:
            precision = TP / (TP + FP)
        
        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
    
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_pred.append(y_pred)
        all_acc.append(accuracy)
        
    
        print('nº pain: %d / %d - MV accuracy: %0.2f' % (list(y_test).count(1), len(y_test),list(y_test).count(0) /len(y_test)))
        """
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F1-score: ',f1)
        print('Accuracy: ', accuracy)
        """
    print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
    print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
    print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))
    print('Accuracy: %0.2f (+/-) %0.2f' % (np.mean(all_acc), np.std(all_acc)))

    
#%%
###############################################################################
#                                      MAIN
###############################################################################
    
for file in os.listdir(FINAL_FOLDER):
    if file.endswith('_s.npy'):
        img_path, lms, pose, ocs, average_score = np.load(os.path.join(FINAL_FOLDER,file), allow_pickle = True)
        roll, pitch, yaw = pose
    
        #print('img_path: ', img_path)
        #print('file: ', file)
        
        img = cv.imread(os.path.join(FINAL_FOLDER, img_path.split('/')[-1]))
    
        #FOR DEBUG
        #=====================
        pose = (roll, pitch, yaw)
        if ocs == -1:
            lms = update_lms(lms, yaw)     
            
            """
            for pt in [lms[0], lms[1], lms[21], lms[22]]:
                r = 2
                cv.circle(img, (int(pt[0]), int(pt[1])), r, (0, 255, 0), -1)
            cv.imshow(img_path.split('/')[-1] + '(before flip) YAW: %0.02f' % yaw  , img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            """ 
        else:
            if yaw < 0:
                n_lms = []
                n_lms.append(lms[5])
                n_lms.append(lms[4])
                n_lms.append(lms[3])
                n_lms.append(lms[2])
                n_lms.append(lms[1])
                n_lms.append(lms[0])
                n_lms.append(lms[6])
                n_lms.append(lms[7])
                n_lms.append(lms[8])
                n_lms.append(lms[10])
                n_lms.append(lms[9])
                n_lms.append(lms[12])
                n_lms.append(lms[11])
                n_lms.append(lms[14])
                n_lms.append(lms[13])
                n_lms.append(lms[16])
                n_lms.append(lms[15])
                n_lms.append(lms[18])
                n_lms.append(lms[17])
                n_lms.append(lms[20])
                n_lms.append(lms[19])
                n_lms.append(lms[23])
                n_lms.append(lms[24])
                n_lms.append(lms[21])
                n_lms.append(lms[22])
                
                lms = n_lms
            
        if yaw < 0:
            
            img,lms = flip_sheep(img, lms)
            yaw = - yaw
            pose  = (roll, pitch, yaw)
            """
            for pt in [lms[3], lms[20], lms[18], lms[6]]:
                r = 2
                cv.circle(img, (int(pt[0]), int(pt[1])), r, (255, 0, 0), -1)
            cv.imshow(img_path.split('/')[-1] + 'YAW: %0.02f' % yaw  , img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            """
        if yaw < 0:
            print('Error: yaw < 0')
        elif yaw < 10:
            new_path = os.path.join(FOLDER_30, str(int(yaw)) + '_' + img_path.split('/')[-1])
            cv.imwrite(new_path, img)
            info_10.append([new_path, lms, pose, ocs, average_score])

        elif yaw < 60:
            new_path = os.path.join(FOLDER_60, str(int(yaw)) + '_' + img_path.split('/')[-1])
            cv.imwrite(new_path, img)
            info_60.append([new_path, lms, pose, ocs, average_score])
           
        elif yaw < 90:
            new_path = os.path.join(FOLDER_90, str(int(yaw)) + '_' + img_path.split('/')[-1])
            cv.imwrite(new_path, img)
            info_90.append([new_path, lms, pose, ocs, average_score])
            
info_10.sort()
info_60.sort()
info_90.sort()

#print("10\n======")
HOG_10, geometry_10, pose_10 = get_features(info_10)

features_10 = np.concatenate((np.asarray(HOG_10), 
                              np.asarray(pose_10), 
                              np.asarray(geometry_10)), axis = 1)

#print("60\n======")
HOG_60, geometry_60, pose_60 = get_features(info_60)

updated_HOG = []

#print("90\n======")
HOG_90, geometry_90, pose_90 = get_features(info_90)

features_10 = np.concatenate((np.asarray(HOG_10), 
                              np.asarray(pose_10), 
                              np.asarray(geometry_10)), axis = 1)


features_60 = np.concatenate((np.asarray(HOG_60), 
                              np.asarray(pose_60), 
                              np.asarray(geometry_60)), axis = 1)


features_90 = np.concatenate((np.asarray(HOG_90), 
                              np.asarray(pose_90), 
                              np.asarray(geometry_90)), axis = 1)


complete_info_10 = np.concatenate((info_10, features_10), axis = 1)
complete_info_60 = np.concatenate((info_60, features_60), axis = 1)
complete_info_90 = np.concatenate((info_90, features_90), axis = 1)

complete_info = np.concatenate((complete_info_30, complete_info_60, complete_info_90), axis = 0)
complete_info = sorted(complete_info, key = lambda x: x[0].split('/')[-1].split('_')[2])

# Group images by videos so they stay in the same group
prev_name = complete_info[0][0].split('/')[-1]
prev_name = prev_name.split('_')[2:]
if len(prev_name) > 1:
    prev_name = prev_name[0] + '_' + prev_name[1]

count = 0
div_info = [[complete_info[0]]]
for i in range(1,len(complete_info)):
    name = complete_info[i][0].split('/')[-1].split('_')[2:]
    
    if len(name) > 1:
        name = name[0] + '_' + name[1]

    if name == prev_name:
        div_info[count].append(complete_info[i])
    else:
        div_info.append([complete_info[i]])
        count += 1
    prev_name = name
        
# Depending on the number of frames from the same video! (from 1 to 4)
X_1_train, y_1_train, X_1_test, y_1_test = cross_validation(np.vstack([i for i in div_info if len(i) == 1]))
X_2_train, y_2_train, X_2_test, y_2_test = cross_validation(np.vstack([i for i in div_info if len(i) == 2]))
X_3_train, y_3_train, X_3_test, y_3_test = cross_validation(np.vstack([i for i in div_info if len(i) == 3]))
X_4_train, y_4_train, X_4_test, y_4_test = cross_validation(np.vstack([i for i in div_info if len(i) == 4]))

"""
# For nested cross validation
for f in range(N_FOLDS):
    cv_x1_train, cv_y1_train, cv_x1_test, cv_y1_test = nested_cross_val(X_1_train[f], y_1_train[f])
    cv_x2_train, cv_y2_train, cv_x2_test, cv_y2_test = nested_cross_val(X_2_train[f], y_2_train[f])
    cv_x3_train, cv_y3_train, cv_x3_test, cv_y3_test = nested_cross_val(X_3_train[f], y_3_train[f])
    cv_x4_train, cv_y4_train, cv_x4_test, cv_y4_test = nested_cross_val(X_4_train[f], y_4_train[f])
    
    all_x_train = [np.concatenate((cv_x1_train[f1][:,6:], cv_x2_train[-f1][:,6:], cv_x3_train[f1][:,6:], cv_x4_train[-f1][:,6:]), 
    axis = 0) for f1 in range(N_N_FOLDS)]
    
    all_x_test = [np.concatenate((cv_x1_test[f1][:,6:], cv_x2_test[-f1][:,6:], cv_x3_test[f1][:,6:], cv_x4_test[-f1][:,6:]),
                            axis = 0) for f1 in range(N_N_FOLDS)]
    
    all_y_train = [np.concatenate(((cv_y1_train[f1],cv_y2_train[-f1],cv_y3_train[f1], cv_y4_train[-f1])), axis = 0) for f1 in range(N_N_FOLDS)]
    all_y_test = [np.concatenate(((cv_y1_test[f1],cv_y2_test[-f1],cv_y3_test[f1], cv_y4_test[-f1])), axis = 0) for f1 in  range(N_N_FOLDS)]
    train_model(all_x_train, all_x_test, all_y_train, all_y_test, N_N_FOLDS)

"""

all_x_train = [np.concatenate((X_1_train[f][:,6:], X_2_train[-f][:,6:], X_3_train[f][:,6:], X_4_train[-f][:,6:]), 
axis = 0) for f in range(N_FOLDS)]

all_x_test = [np.concatenate((X_1_test[f][:,6:], X_2_test[-f][:,6:], X_3_test[f][:,6:], X_4_test[-f][:,6:]),
                        axis = 0) for f in range(N_FOLDS)]

all_y_train = [np.concatenate((y_1_train[f],y_2_train[-f],y_3_train[f], y_4_train[-f]), axis = 0) for f in range(N_FOLDS)]
all_y_test = [np.concatenate(((y_1_test[f],y_2_test[-f],y_3_test[f], y_4_test[-f])), axis = 0) for f in range(N_FOLDS)]

train_model(all_x_train, all_x_test, all_y_train, all_y_test, N_FOLDS)
