

#%%
###############################################################################
#                                IMPORTS
###############################################################################
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2 as cv
import os
import numpy as np

DATASET_PATH = '/' + os.path.join(*os.getcwd().split('/')[:-1],'dataset')  
TRACKER = 'kcf' # Discriminative Correlation Filter with Channel and Spatial Reliability (DCF-CSR)
#%%
###############################################################################
#                       CKECK OPENCV VERSION - INIT TRACKERS
###############################################################################

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv.TrackerCSRT_create,
	"kcf": cv.TrackerKCF_create,
	"boosting": cv.TrackerBoosting_create,
	"mil": cv.TrackerMIL_create,
	"tld": cv.TrackerTLD_create,
	"medianflow": cv.TrackerMedianFlow_create,
    "mosse": cv.TrackerMOSSE_create
	}
    
# initialize the bounding box coordinates of the object we are going
# to track
tracker = OPENCV_OBJECT_TRACKERS[TRACKER]()
    
clip_folder = os.path.join(DATASET_PATH, 'video_annotations', 'clips')
info_folder = os.path.join(DATASET_PATH, 'video_annotations', 'clips', 'info')
pain_folder = os.path.join(DATASET_PATH, 'video_annotations', 'clips', 'pain')
all_files = os.listdir(info_folder)
cheeks = 9
orbital = 9
ears = 9
lip = 9
nose = 9

#%%

def nothing(x):
    pass
    
for file in all_files:
    print(file[0])
    if len(file) != 4 :
        tracker = OPENCV_OBJECT_TRACKERS[TRACKER]()
        info = np.load(os.path.join(info_folder,file), allow_pickle=True)
        if len(info) > 0:
            vid_path, init_bb, lms, occs = info
            vid = cv.VideoCapture(os.path.join(clip_folder,vid_path))
            print('NEW VIDEO: ', vid_path)
            print('===============================')
            
            # (x,y), (x+w,y+h)  -> (x,y,w,h)
            x = int(init_bb[0][0])
            y = int(init_bb[0][1])
            w = int(init_bb[1][0]) - x
            h = int(init_bb[1][1]) - y
            init_bb = (x,y,w,h)
            print('init_bb: ', init_bb)
            
            success, frame = vid.read()
            
            if not success:
                print('Cannot read the video file')
                exit()
            
            h0, w0 = frame.shape[:2]
            frame = imutils.resize(frame, width=500)
            h1, w1 = frame.shape[:2]
            x = x * (w1/w0)
            y = y * (h1/h0)
            w = w * (w1/w0)
            h = h * (h1/h0)
            init_bb = (x,y,w,h)
            
            success = tracker.init(frame, init_bb)
            
            path_pain = os.path.join(pain_folder,file.split('.')[0] + '_pain.npy')
           
            if not os.path.exists(path_pain):
               print(path_pain, ' does not exist!!!')
              
            else:
                pain = np.load(path_pain)
                
                cheeks = int(pain[0,1])
                orbital = int(pain[1,1])
                ears = int(pain[2,1])
                lip = int(pain[3,1])
                nose = int(pain[4,1])
                
                window_name = 'Tracking ' + vid_path.split('/')[-1]
                cv.namedWindow(window_name)
                cv.createTrackbar('cheeks', window_name , cheeks, 3, nothing)
                cv.createTrackbar('orbital', window_name , orbital, 3, nothing)
                cv.createTrackbar('ears', window_name , ears, 3, nothing)
                cv.createTrackbar('lip', window_name , lip, 3, nothing)
                cv.createTrackbar('nose', window_name , nose, 3, nothing)
                    
                for i in [cheeks,orbital,ears, lip, nose]:
                    if i == 9:
                        i = 3
                                
                while True:
                    # read new frame
                    success, frame = vid.read()
                    if not success:
                        break
                    frame = imutils.resize(frame, width=500)
    
                    # timer
                    timer = cv.getTickCount()
                    # update tracker
                    sucess, bbox = tracker.update(frame)
                    # fps
                    fps = cv.getTickFrequency()/ (cv.getTickCount() - timer)
                    #print('fps: ', fps)
                    if success:
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]),  int(bbox[1] + bbox[3]))
                        #cv.rectangle(frame, p1, p2,(255,0,0),2,1)
                        
                    else:
                        cv.putText(frame, 'Tracking failure detected', (100,80),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                      
                    cheeks = cv.getTrackbarPos('cheeks', window_name)
                    orbital = cv.getTrackbarPos('orbital', window_name)
                    ears = cv.getTrackbarPos('ears', window_name)
                    lips = cv.getTrackbarPos('lips', window_name)
                    nose = cv.getTrackbarPos('nose', window_name)
                    
                    for i in [cheeks,orbital,ears, lips, nose]:
                        if i == 3:
                            i = 9
                    
                    cv.imshow(window_name, frame)
                    cv.waitKey(100) #30 frames per second - the time is in miliseconds
    
                    #print('cheeks: ', cheeks)
                    
                    
                    pain[0,1] = cheeks
                    pain[1,1] = orbital
                    pain[2,1] = ears
                    pain[3,1] = lip
                    pain[4,1] = nose
                    
                np.save(path_pain, pain)

        else:
            print('SEE WHATS GOING ON!!! - ', file)
        cv.waitKey(0)
        cv.destroyAllWindows()