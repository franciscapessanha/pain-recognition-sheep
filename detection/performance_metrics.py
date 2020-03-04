
#%%
###############################################################################
#                                  IMPORTS
###############################################################################

import os
import numpy as np
import cv2 as cv
import csv
from operator import itemgetter
import argparse

TP = 0
FP = 0
FN = 0
n_gt = 0

MIN_FUSE = 0.6
MIN_SCORES = [0.7]
#MIN_SCORES = [0.5, 0.6, 0.7,0.8, 0.9]
FOLD = '4'
method = 'mean'
folder = '0.05_data'

"""
parser = argparse.ArgumentParser()
parser.add_argument('fold', action="store", type=str)
parser.add_argument('min_fuse', action="store", type=float)
parser.add_argument('method', action="store", type=str)
results = parser.parse_args()
FOLD = results.fold
MIN_FUSE = results.min_fuse
method = results.method
"""
image_folder = os.path.join(os.getcwd(),folder,'final_data_' + FOLD, 'images', 'test')
                
example_folder = os.path.join(os.getcwd(),folder,'final_data_' + FOLD, 'examples')
if not os.path.exists(example_folder):
    os.mkdir(example_folder)
#%%
###############################################################################
#                            GENERAL UTILS
###############################################################################

def get_txt(directory):
    files = []
    full_path = []
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            files += [file]
            full_path += [os.path.join(directory, file)]     
    files.sort()
    full_path.sort()
    
    return files, full_path

def get_bbox(box):
    
    xmin = int(box[0])
    ymin = int(box[1])
    xmax = int(box[2])
    ymax = int(box[3])
    
    return xmin, ymin, xmax, ymax
    
def get_iou(xmin_0, ymin_0, xmax_0, ymax_0, xmin_1, ymin_1, xmax_1, ymax_1):
    x_left = max(xmin_0, xmin_1)
    y_top = max(ymin_0, ymin_1)
    x_right = min(xmax_0,xmax_1)
    y_bottom = min(ymax_0, ymax_1)
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_0 = (xmax_0 - xmin_0) * (ymax_0 - ymin_0)
    area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)

    iou = intersection_area / float(area_0 + area_1 - intersection_area)

    return iou

#%%%

def fuse_bboxes(pred_bboxes, method):
    #print('len pred_bboxes: ', len(pred_bboxes))
    fused_pred = []
    fused = False
    compare = np.copy(pred_bboxes).tolist()
    for box0 in pred_bboxes:
        if len(compare) > 1:
            xmin_0, ymin_0, xmax_0, ymax_0 = get_bbox(box0[2:]) 
            compare.remove(box0)
            for box1 in compare:
                xmin_1, ymin_1, xmax_1, ymax_1 = get_bbox(box1[2:]) 
                iou = get_iou(xmin_0, ymin_0, xmax_0, ymax_0, xmin_1, ymin_1, xmax_1, ymax_1)
                    
                #fuse bounding boxes if they have a IoU > percentage defined by the user (defaut: 0.5) 
                if iou > MIN_FUSE:
                    if method == 'fuse':
                        x_left = min(xmin_0, xmin_1)
                        y_top = min(ymin_0, ymin_1)
                        x_right = max(xmax_0,xmax_1)
                        y_bottom = max(ymax_0, ymax_1)
                        fused = True
                        fused_pred.append(['sheep', 'intersection', str(x_left), str(y_top), str(x_right),str(y_bottom)])
    
                    elif method == 'mean':
                        x_left = int(np.mean([xmin_0, xmin_1]))
                        y_top = int(np.mean([ymin_0, ymin_1]))
                        x_right = int(np.mean([xmax_0,xmax_1]))
                        y_bottom = int(np.mean([ymax_0, ymax_1]))
                        fused = True
                        fused_pred.append(['sheep', 'intersection', str(x_left), str(y_top), str(x_right),str(y_bottom)])
                    break
                    
            if not fused:
                #print('not fused')
                fused_pred.append(box0)
        else:
            fused_pred.append(box0)        
    return fused_pred
#%%
###############################################################################
#                           GET PERFORMANCE METRICS
###############################################################################

dir_predictions = os.path.join(os.getcwd(),folder, 'final_data_' + FOLD, 'predictions')
csv_gt = os.path.join(os.getcwd(), folder,'final_data_' + FOLD, 'annotations', 'test_labels.csv')

with open(csv_gt) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    gt_list = []
    for i, row in enumerate(csv_reader):
        if i > 0:
            gt_list.append([row[0], row[4], row[5], row[6], row[7]]) #filename, xmin, ymin, xmax, ymax


files, pred_path = get_txt(dir_predictions)
gt_list = sorted(gt_list, key = itemgetter(0))
pred_path = sorted(pred_path,  key = itemgetter(0))

gt_info = []
bbs_img = []
bbs_img.append(gt_list[0])
for i in range(1, len(gt_list)):
    if gt_list[i - 1][0] == gt_list[i][0]:
        bbs_img.append(gt_list[i])
    else:
        gt_info.append(bbs_img)
        bbs_img = []
        bbs_img.append(gt_list[i])

gt_info = sorted(gt_info, key = itemgetter(0))

for MIN_SCORE in MIN_SCORES:
    n_fuse = 0
    index_gt = 0
    
    n_gt = 0
    n_pred = 0
    for pred, name in zip(pred_path, files):
        if index_gt < len(gt_info):
            gt_name = gt_info[index_gt][0][0].split('.')[0]
            #print('ground truth: ', gt_name)
            #print('prediction: ', name)
            if gt_name in name: #gt_info[index_gt][0][0].split('.')[0] is the name of the image
                png_image = os.path.join(image_folder, name.replace('txt','png'))
                img = cv.imread(png_image)
                
                pred_info = open(pred,'r').readlines()
                pred_info = [s.split(' ') for s in pred_info]
                pred_bboxes = []
                
                # Post processing of the prediction (excluding bbs with low confidence and fusing similar ones)
                #==================================================================
                for bbox in pred_info:
                    if float(bbox[1]) > MIN_SCORE:
                        pred_bboxes.append(bbox)  
                fused_pred = fuse_bboxes(pred_bboxes, method)
                
                # Count ground truth bounding boxes
                gt_bboxes = [b[1:] for b in gt_info[index_gt]]
                index_gt += 1
                for gt_bb in gt_bboxes:
                    xmin, ymin, xmax, ymax = get_bbox(gt_bb) 
                    pt1 = (int(xmin), int(ymin))
                    pt2 = (int(xmax),int(ymax))
                    #cv.rectangle(img, pt1, pt2, (0,0,255),2)
                n_gt += len(gt_bboxes)    
                   
                # Count predicted bounding boxes
                pred_bboxes = fused_pred
                #print('fused pred: ', fused_pred)
                for pred_bb in pred_bboxes:
                    xmin, ymin, xmax, ymax = get_bbox(pred_bb[2:]) 
                    pt1 = (int(xmin), int(ymin))
                    pt2 = (int(xmax),int(ymax))
                    cv.rectangle(img, pt1, pt2, (0,255,0),2)
                n_pred += len(pred_bboxes) 
                
                
                pred_bb_copy = np.copy(pred_bboxes).tolist()
                gt_bb_copy = np.copy(gt_bboxes).tolist()
                
                for pred_bb in pred_bboxes:
                    found_box = False
                    if len(gt_bb_copy) > 0:
                        gt = gt_bb_copy
                        for gt_bb in gt:
                            xmin_0, ymin_0, xmax_0, ymax_0 = get_bbox(pred_bb[2:]) 
                            xmin_1, ymin_1, xmax_1, ymax_1 = get_bbox(gt_bb) 
                            iou = get_iou(xmin_0, ymin_0, xmax_0, ymax_0, xmin_1, ymin_1, xmax_1, ymax_1)
                            
                            if iou > 0.5:
                                TP += 1
                                gt_bb_copy.remove(gt_bb)
                                pred_bb_copy.remove(pred_bb)
                                found_box = True
                                break
                        
                        if found_box == False:
                            pred_bb_copy.remove(pred_bb)
                            FP += 1
                            
                if len(gt_bb_copy) > 0:
                    FN += len(gt_bb_copy)
                                
            
                FP += len(pred_bb_copy)
                xmin_det, ymin_det, xmax_det, ymax_det = get_bbox(pred_bb[2:]) 
                
            else:
                #print('Image bboxes excluded because of the size')
                #index_gt += 1
                pass
            
            cv.imwrite(os.path.join(example_folder,name.replace('txt','png')), img)
               
        
    
    print("TOTAL POSITIVES: ", n_gt)
    print("TOTAL PREDICTED: ", n_pred)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1_score = 2 * (precision * recall)/(precision + recall)
    print("\n\nmin_score ", MIN_SCORE)
    print("min_fuse", MIN_FUSE)
    print('TP: %d | FP: %d' % (TP, FP))
    print('FN: %d' % (FN))
    print('Precision: %02f' % (precision * 100))
    print('Recall: %02f' % (recall * 100))
    print('F1 score: %02f' %(f1_score * 100))
    TP = 0
    FN = 0
    FP = 0
    
    
        