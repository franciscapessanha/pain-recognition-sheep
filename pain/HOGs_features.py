###############################################################################
#                                  IMPORTS
###############################################################################

import os
import cv2 as cv

#from skimage.transform import rotate ## Alternatively

import numpy as np
#import matplotlib.pyplot as plt
import math
from skimage.feature import hog
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from skimage import exposure

DATASET_PATH = '/' + os.path.join(*os.getcwd().split('/')[:-1],'dataset')
FINAL_FOLDER = os.path.join(DATASET_PATH, 'scored','final')

ABS_FOLDER = os.path.join(os.getcwd(), 'abs_yaw')
FOLDER_10 = os.path.join(ABS_FOLDER, '0_10')
FOLDER_60 = os.path.join(ABS_FOLDER, '10_60')
FOLDER_90 = os.path.join(ABS_FOLDER, '60_90')


ew_ear = 0.05
eh_ear = 0.05

ew_nose = 0.20
eh_nose = 0.20

ear_dim = (100,80) # W/H = 1.55
eye_dim = (80,80)
nose_dim = (80,120)
cheek_dim = (80,80)
################################################################################
def rotate_image(image, points, angle):
    h, w, _ = np.shape(image)
    centre = (h/2, w/2)    
    M = cv.getRotationMatrix2D(centre, angle, 1)
    h, w = image.shape[:2]
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    new_image = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC)
    #new_points = cv.transform(points_ones, M)
    new_points = M.dot(points_ones.T).T
    min_x = np.min(new_points[:, 0])
    max_x = np.max(new_points[:, 0])
    min_y = np.min(new_points[:, 1])
    max_y = np.max(new_points[:, 1])
    bw = max_x - min_x
    bh = max_y - min_y
    new_box = [int(min_x - 0.08 * bw), int(min_y - 0.1 * bh),
               int(1.16 * bw), int(1.16 * bh)]
    #draw('rotated', new_image, new_box, new_points, 0)
    
    
    new_h, new_w = new_image.shape[:2]
    updated_points = []
    
    for pt in  new_points:
        x = pt[0]
        y = pt[1]
        if x > new_w:
            x = new_w
        elif x < 0:
            x = 0
            
        if y > new_h:
            y = new_h
        elif y < 0:
            y = 0
        updated_points.append([int(x), int(y)])
    
    
    #cv.circle(new_image, (int(pt[0]), int(pt[1])), r, (255, 0, 0), -1)
    """
    cv.namedWindow('image',cv.WINDOW_NORMAL)
    cv.imshow('image' , new_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """         
    return new_image, new_box, np.asarray(updated_points)



#                               GET LMS PER ACTION UNIT
###############################################################################
def get_lms_per_AU(lms, yaw):    
    if abs(yaw) < 30:
        nose = [lms[i] for i in [6, 7, 17, 18, 19, 20]]
        right_eye = [lms[i] for i in [2]]
        left_eye = [lms[i] for i in [3]]
        right_ear = [lms[i] for i in [0, 1, 21, 22]]
        left_ear = [lms[i] for i in [4, 5, 23, 24]]
        left_cheek = []
    elif abs(yaw) < 60:
        nose = [lms[i] for i in [6, 7, 18, 20]]
        right_eye = []
        left_eye = [lms[i] for i in [3]]
        right_ear = [lms[i] for i in [0, 1, 21, 22]]
        left_ear = [lms[i] for i in [4, 5, 23, 24]] # top, base, middle_top, middle_bottom
        left_cheek = [lms[i] for i in [12, 14, 16]]

    elif abs(yaw) < 90:
        nose = [lms[i] for i in [6, 7, 18, 20]]
        right_eye = []
        left_eye = [lms[i] for i in [3]]
        right_ear = []
        left_ear = [lms[i] for i in [4, 5, 23, 24]] # top, base, middle_top, middle_bottom
        left_cheek = [lms[i] for i in [12, 14, 16]]
        
    return (right_ear, left_ear, right_eye, left_eye, left_cheek, nose)

#                               GET EARS 
###############################################################################
def get_ear(img, ear, name, side, EARS):
    
    (x1, y1) = ear[0] 
    (x2, y2) = ear[1] 
     
    if y1 < y2:
        tx = x1 #top
        ty = y1
        bx = x2 #base
        by = y2
    else:
        tx = x2
        ty = y2
        bx = x1
        by = y1
    
    
    if ty != by and bx != tx:
        angle = np.arctan((by - ty)/(bx - tx)) #radians
        if tx > bx: #rotation clockwise
            angle = angle * 180/math.pi
        else: #anticlockwise
            angle = - angle * 180/math.pi
            angle = 360 - angle
    else:
         angle = 0
     
    n_img, n_box, n_ear = rotate_image(img, ear, angle)            
    img_h, img_w = n_img.shape[:2]
    
    medial_y = n_ear[0,1]
    
    x, y, w, h = [np.min(n_ear[:, 0]),
                  np.min(n_ear[:, 1]),
                  np.max(n_ear[:, 0]) - np.min(n_ear[:, 0]),
                  np.max(n_ear[:, 1]) - np.min(n_ear[:, 1])] 
    

    h = 2 * max(abs(medial_y - n_ear[2,1]), abs(medial_y - n_ear[3,1]))
    
    ratio_ear = w/h
    
    l = int(max(x - w * ew_ear, 0))
    r = int(min(x + w * (1 + ew_ear), img_w))

    t = int(max(y - h * eh_ear, 0))
    b = int(min(y + h * (1 + eh_ear),  img_h))

    
    im_sq = n_img[t :b, l:r]
    
    """
    for p in ear:
        r = 2
        cv.circle(img,tuple(p), r, (255, 0, 0), -1)
   
        cv.imshow('ear', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    """
    im_sq = cv.resize(im_sq, ear_dim) 
    cv.imwrite(os.path.join(EARS, side[0] + '_ear_' + name), im_sq)
    
    return im_sq, ratio_ear
#                               GET EYES
###############################################################################
def get_eye(img, dist, eye, name, side, EYES):
    img_h, img_w = img.shape[:2]
    cx, cy = eye[0]
    #print(dist)
    if dist != 0:
        error = (1/6* dist)  #  supossing each eye occupies 1/4 of the full interpupilar dist
    else:
        error = 20
    
    l = int(max(int(cx)  - error, 0))
    t = int(max(int(cy) - error, 0))
    r = int(min(int(cx) + error, img_w))
    b = int(min(int(cy)  + error, img_h))
  
    im_sq = img[t :b, l:r]
    im_sq = cv.resize(im_sq, (80,80)) 
    cv.imwrite(os.path.join(EYES, side[0] + '_eye_' + name), im_sq)
    
    #return (angle, im_sq)
    return im_sq

#                               GET NOSE
###############################################################################
def get_nose(img, nose, name, NOSE):
    (x1, y1) = nose[0]
    (x2, y2) = nose[1]
    
    if x1 != x2:
        angle = np.arctan((y2 - y1)/(x2 - x1)) #radians
        if x1 > x2: #rotation clockwise
            angle = 90 + angle * 180/math.pi

        else: #anticlockwise
            angle = 90 - angle * 180/math.pi
            angle = 360 - angle
    else:
        angle = 0
     
    n_img, n_box, n_nose = rotate_image(img,nose, angle)
    
    n_nose = np.vstack(n_nose)
    
    x, y, w, h = [np.min(n_nose[:, 0]),
                  np.min(n_nose[:, 1]),
                  np.max(n_nose[:, 0]) - np.min(n_nose[:, 0]),
                  np.max(n_nose[:, 1]) - np.min(n_nose[:, 1])] 
    
    medial_x = n_nose[0,0]
    w = 2 * abs(medial_x - n_nose[-1,0])
    
    ratio_nose = w/h
    
    img_h, img_w = n_img.shape[:2]
     
    
    l = int(max(x - w * ew_nose, 0))
    r = int(min(x + w * (1 + ew_nose), img_w))

            
    t = int(max(y - h * eh_nose, 0))
    b = int(min(y + h * (1 + eh_nose),  img_h))


    im_sq = n_img[t :b, l:r]
    im_sq = cv.resize(im_sq, (80,100)) 
    cv.imwrite(os.path.join(NOSE, 'nose_' + name), im_sq)

    
    return im_sq, ratio_nose
#                               GET CHEEK
###############################################################################
def get_cheek(img, cheek, name, CHEEK):
    cheek = np.vstack(cheek)
    x, y, w, h = [np.min( cheek[:, 0]),
                  np.min( cheek[:, 1]),
                  np.max( cheek[:, 0]) - np.min( cheek[:, 0]),
                  np.max( cheek[:, 1]) - np.min( cheek[:, 1])] 
        
    eh = 0.05
    img_h, img_w = img.shape[:2]
    cx, cy = (x + w / 2, y + h / 2)

    ew = 0.15
    l = int(max(int(cx - ((w + w * ew ) / 2)), 0))
    r = int(min(int(cx + ((w + w * ew) / 2)), img_w))        
    b = int(min(int(cy + ((h + h * eh) / 2)), img_h))
    t = int(max(int(cy - ((h + h * eh) / 2)), 0))

    im_sq = img[t :b, l:r]
    im_sq = cv.resize(im_sq, (80,80)) 
    cv.imwrite(os.path.join(CHEEK, 'cheek_' + name), im_sq)
    
    return im_sq
  
        
#%%############################################################################
#                                       HOG 
###############################################################################

def get_HOG_features(image_name, img, lms, yaw):
    (right_ear, left_ear, right_eye, left_eye, left_cheek, nose) =  get_lms_per_AU(lms, yaw)
    
    
    if yaw < 10:
        folder = FOLDER_10
    elif yaw < 60:
        folder = FOLDER_60
    elif yaw < 90:
        folder = FOLDER_90
        
    EARS = os.path.join(folder, 'ears')
    EYES = os.path.join(folder, 'eyes')
    CHEEK = os.path.join(folder, 'cheek')
    NOSE = os.path.join(folder, 'nose')
    
    ratio_ears = []
    AUs = []
    
    # LEFT
    #===============================================
    l_ear, ratio_ear = get_ear(img, left_ear, image_name,'left', EARS)
    AUs.append(l_ear)
    ratio_ears.append(ratio_ear)
    
    if right_eye != []:    
        dist = math.sqrt(((left_eye[0][0]-right_eye[0][0])**2)+((left_eye[0][1]-right_eye[0][1])**2) ).real
    else:
        dist = 0
    l_eye = get_eye(img, dist, left_eye, image_name,'left', EYES)
    AUs.append(l_eye)
    
    nose, ratio_nose = get_nose(img, nose, image_name, NOSE) # lateral nose and mouth
    AUs.append(nose)
    
    if left_cheek != []:
        cheek = get_cheek(img, left_cheek, image_name, CHEEK)
        AUs.append(cheek)
    else:
        AUs.append([-1.0])
     
    # RIGHT
    #===============================================
    if right_ear != []:
        r_ear, ratio_ear = get_ear(img, right_ear, image_name,'right', EARS)
        AUs.append(r_ear)
        ratio_ears.append(ratio_ear)
    else:
        AUs.append([-1.0])

    # EYES
    if right_eye != []:
        r_eye = get_eye(img,dist, right_eye, image_name,'right', EYES)
        AUs.append(r_eye)
    else:
        AUs.append([-1.0])

      
    all_HOG = []
    ear_size = (ear_dim[0],ear_dim[1])
    eye_size = (eye_dim[0],eye_dim[1])
    nose_size = (nose_dim[0],nose_dim[1])
    cheek_size = (cheek_dim[0],cheek_dim[1])
    
    sizes = [ear_size, eye_size, nose_size, cheek_size, ear_size, eye_size]
    labels = [os.path.join(EARS,'l_ear_hog_'),
          os.path.join(EARS,'l_eye_hog_'),
          os.path.join(EYES,'nose_hog_'),
          os.path.join(EYES,'cheek_hog_'),
          os.path.join(NOSE,'r_ear_hog_'),
          os.path.join(CHEEK,'r_eye_hog_')]
        
    """
    labels = [os.path.join(EARS,'l_ear_hog_'),
              os.path.join(EARS,'r_ear_hog_'),
              os.path.join(EYES,'l_eye_hog_'),
              os.path.join(EYES,'r_eye_hog_'),
              os.path.join(NOSE,'nose_hog_'),
              os.path.join(CHEEK,'cheek_hog_')]
    """

    o = 9
    ppc = 16
    cpb = 2
    
    #transform_sqrt=False
    for au, size, l in zip(AUs, sizes, labels):
        #print(np.shape(au))
        if len(au) == 1:
            fd, hog_image = hog(np.zeros(size), orientations= o , pixels_per_cell=(ppc, ppc),
                    cells_per_block=(cpb, cpb), visualize = True)
            all_HOG.append(fd)
        else:
            fd, hog_image = hog(au,  orientations= o, pixels_per_cell=(ppc,ppc),
                    cells_per_block=(cpb, cpb), visualize = True, multichannel=True)
            all_HOG.append(fd)
            hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))
            cv.imwrite(os.path.join(l + image_name), hog_image)
            
            #plt.figure()
            #plt.imshow(hog_image)
            #print('hog_image shape: ', len(np.hstack(hog_image)))
    return all_HOG, ratio_ears, ratio_nose