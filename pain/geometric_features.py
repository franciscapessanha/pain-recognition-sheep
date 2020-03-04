import os
import cv2 as cv
import numpy as np
from skimage.feature import hog
from ind_HOGs_features import get_lms_per_AU
import cmath as math
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import math
from sklearn.metrics import confusion_matrix


#                               GLOBAL ANGLE EAR
###############################################################################
def global_ear_angle(ear):
    (tx, ty) = ear[0] # top
    (bx, by) = ear[1] # base
     
    if ty != by and bx != tx:
        angle = np.arctan((by - ty)/(bx - tx)) #radians
        if tx > bx: #rotation clockwise
            angle = angle * 180/math.pi
        else: #anticlockwise
            angle = - angle * 180/math.pi
            angle = 360 - angle
    else:
        angle = 0
    return angle
            


#                              INTERNAL ANGLE EARS
###############################################################################


def internal_ear_angle(lms, side):
    base = lms[0]
    tip = lms[1]
    mid_top = lms[2]
    
 
    a = np.sqrt((tip[0]-base[0])**2 + (tip[1]-base[1])**2)
    c = np.sqrt((base[0]-mid_top[0])**2 + (base[1]-mid_top[1])**2)
    b = np.sqrt((tip[0]-mid_top[0])**2 + (mid_top[1]-base[1])**2)
    
    #print('side: ', side)
    #print('a: ', a, 'b: ', b, 'c: ', c)
    angle = np.arccos((b**2 - a**2 - c**2)/(-2*a*c))
    if math.isnan(angle):
        angle = -1
        return angle
    return angle * (180 / math.pi)

def get_geometric_features(lms, yaw):
    (right_ear, left_ear, _, _, _, _) =  get_lms_per_AU(lms, yaw)
    
    geometry = []
    if right_ear == []:
        geometry.append(-1)
        #geometry.append(-1)
    else:
        geometry.append(global_ear_angle(right_ear))
        #geometry.append(internal_ear_angle(right_ear, 'right'))
    
    if left_ear == []:
        geometry.append(-1)
        #geometry.append(-1)
    else:
        geometry.append(global_ear_angle(left_ear))
        #geometry.append(internal_ear_angle(left_ear, 'left'))
        
    if right_ear != [] and left_ear != []:
        lms =np.asarray([lms[i] for i in (list(range(1, 5)) + list(range(6, 21)))]) #without ears
        xmin = np.min(lms[:, 0])
        xmax = np.max(lms[:, 0])
        dist = (left_ear[1][0] - right_ear[1][0]) / (xmax - xmin) 
        geometry.append(dist)
    else:
        geometry.append(-1)
        
    return geometry
    

    
    