#%%
###############################################################################
#                                   IMPORTS
###############################################################################
from imutils.video import VideoStream
from imutils.video import FPS

import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import cv2 as cv
import cv2
import time
from menpofit.dlib import DlibWrapper
import menpo.io as mio

import menpo
import pose.hopenet
import torchvision
import torch
from torchvision import transforms
from torch.autograd import Variable
from menpo.shape import PointCloud
import pose.datasets
import pickle

from pain.train_SVM import get_features, flip_sheep, update_lms


VIDEO_FOLDER = os.path.join(os.getcwd(), 'dataset', 'illness')
SCORED_FOLDER = os.path.join(os.getcwd(), 'dataset', 'scored', 'all_frames')
if not os.path.exists(os.path.join(SCORED_FOLDER)):
    os.makedirs(SCORED_FOLDER)
    
"""
NOTE: If access to a GPU uncomment all the lines with cuda(0)
=============================================================
"""

#%%
###############################################################################
#                            LOADING FOR OBJECT DETECTION
###############################################################################
print('Loading for object detection...')
sys.path.append("..")
from object_detection.utils import ops as utils_ops
FOLD = '0'
PATH_TO_CKPT = os.path.join(os.getcwd(),'0_detection', '0.05_data', 'final_data_' + FOLD, 'ssd_mobilenet_' + FOLD + '_10000.pb')
pb_fname = PATH_TO_CKPT
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(os.getcwd(),'0_detection', '0.05_data','final_data_' + FOLD, 'annotations','label_map.pbtxt')
num_classes = 1

assert os.path.isfile(pb_fname)
assert os.path.isfile(PATH_TO_LABELS)
# This is needed to display the images.
from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


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

MIN_FUSE = 0.6

def fuse_bboxes(pred_bboxes, method):
    #print('len pred_bboxes: ', len(pred_bboxes))
    fused_pred = []
    fused = False
    compare = np.copy(pred_bboxes).tolist()
    for box0 in pred_bboxes:
        if len(compare) > 1:
            xmin_0, ymin_0, xmax_0, ymax_0 = box0
        
            compare.remove(list(box0))
            for box1 in compare:
                xmin_1, ymin_1, xmax_1, ymax_1 = box1 
                iou = get_iou(xmin_0, ymin_0, xmax_0, ymax_0, xmin_1, ymin_1, xmax_1, ymax_1)
                    
                #fuse bounding boxes if they have a IoU > percentage defined by the user (defaut: 0.5) 
                if iou > MIN_FUSE:
                    if method == 'fuse':
                        x_left = min(xmin_0, xmin_1)
                        y_top = min(ymin_0, ymin_1)
                        x_right = max(xmax_0,xmax_1)
                        y_bottom = max(ymax_0, ymax_1)
                        fused = True
                        fused_pred.append([x_left, y_top, x_right,y_bottom])
    
                    elif method == 'mean':
                        x_left = int(np.mean([xmin_0, xmin_1]))
                        y_top = int(np.mean([ymin_0, ymin_1]))
                        x_right = int(np.mean([xmax_0,xmax_1]))
                        y_bottom = int(np.mean([ymax_0, ymax_1]))
                        fused = True
                        fused_pred.append([x_left, y_top, x_right,y_bottom])
                    break
                    
            if not fused:
                #print('not fused')
                fused_pred.append(box0)
        else:
            fused_pred.append(box0)        
    return fused_pred

print('Loaded')
#%%
###############################################################################
#                          LOADING FOR POSE DETECTION
###############################################################################
print('Loading posenet...', end='')
def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result


"""
==============================================================================
The model was train using the Fine-Grained Head Pose Estimation Without Keypoints,
proposed by Ruiz et al.
Repository: https://github.com/natanielruiz/deep-head-pose.git
==============================================================================
"""

FOLD = '1'
MODEL = os.path.join(os.getcwd(), '1_pose', 'models','hopenet_' + str(FOLD) + '_30_aug.pkl')
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
saved_state_dict = torch.load(MODEL)
model.load_state_dict(saved_state_dict)
transformations = transforms.Compose([transforms.Resize(224),
                                  transforms.CenterCrop(224), transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

pose_dataset = datasets.SFLW('', '1_pose/cross_val/pose_info_test_aug_' + str(FOLD) + '.npy', transformations)
test_loader = torch.utils.data.DataLoader(dataset=pose_dataset, batch_size =1, num_workers=2)    

#model.cuda(0) #Uncomment if the PC has a GPU
model.eval()

total = 0

idx_tensor = [idx for idx in range(66)]
#idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
idx_tensor = torch.FloatTensor(idx_tensor)

"""
Predict pose
==============================================================================

Arguments:
    * face: square Image object of a face; dimensions: (224,224)
    
Return:
    * pose prediction: yaw_predicted, pitch_predicted and roll_predicted
    
------------------------------------------------------------------------------
"""                         
def predict_pose(face):
    img = Image.fromarray(face)
    #t = time.time()
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
   
    #img = Variable(img).cuda(0) #Uncomment if the PC has a GPU
    img = Variable(img)
    yaw, pitch, roll = model(img)
    # Binned predictions
    _, yaw_bpred = torch.max(yaw.data, 1)
    _, pitch_bpred = torch.max(pitch.data, 1)
    _, roll_bpred = torch.max(roll.data, 1)

    # Continuous predictions
    yaw_predicted = softmax_temperature(yaw.data, 1)
    pitch_predicted = softmax_temperature(pitch.data, 1)
    roll_predicted = softmax_temperature(roll.data, 1)

    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
    roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99
    
    #print('Pose in', time.time() - t)
    return roll_predicted.item(), pitch_predicted.item(), yaw_predicted.item()

#%%
###############################################################################
#                           LOADING FITTERS
###############################################################################
print('Loading fitters...')
fitters = [DlibWrapper('models/%i_s.ert' % f) for f in range(3)]

print('Loaded %d fitters' % len(fitters))

def mirror_lms(lms, w):
    points = lms.as_vector().reshape((-1, 2))
    new_points = []
    for p in points:
        new_points.append((p[0], w - p[1]))
    new_points = np.array(new_points).reshape((-1, 2))
    return PointCloud(new_points)

#%%
###############################################################################
#                          SAVE ALL PREDICTIONS
###############################################################################

det_times = []
pose_times = []
lms_times = []
fps = []
for SHEEP in ['final']:
    SHEEP_FOLDER = os.path.join(VIDEO_FOLDER, SHEEP)
    paths_videos = [vid for vid in os.listdir(SHEEP_FOLDER)]

    count = 0

    for vid in paths_videos:
        if vid[0] != '.' and vid.split('.')[-1] != 'png' and vid.split('.')[-1] != 'txt':
            sheep_id = vid.split('_')[0]
            vid_frames = mio.import_video(os.path.join(SHEEP_FOLDER,vid))
            print('Starting playback')
            f_num = 0
            print('NEW VIDEO: %d/%d ' % (count, len(paths_videos)), vid)
            
            txt_file = open(os.path.join(SHEEP_FOLDER, vid.split('.')[0] + '.txt'), 'a')
            txt_file.write('frame, bbox, pain\n')
            
            while f_num < len(vid_frames): 

                start = time.time()
                # sort out image
                frame = vid_frames[f_num].rescale_to_diagonal(500)
                open_cv_frame = frame.as_PILImage().convert('RGB')
                open_cv_frame = np.array(open_cv_frame)
                
                # detect faces
                t = time.time()
                
                height,width = np.shape(open_cv_frame)[:2]
                image_np_expanded = np.expand_dims(open_cv_frame, axis=0)
        
                # Actual detection.
                output_dict = run_inference_for_single_image(open_cv_frame, detection_graph)
                # Visualization of the results of a detection.
               
                all_boxes = []
                
                for i in range(len(output_dict['detection_scores'])):
                    if output_dict['detection_scores'][i] > 0.70:
                        boxes = output_dict['detection_boxes'][i]
                        ymin = int(boxes[0]*height)
                        xmin = int(boxes[1]*width)
                        ymax = int(boxes[2]*height)
                        xmax = int(boxes[3]*width)
                        all_boxes.append((ymin, xmin, xmax, ymax))
                        #cv.rectangle(copy, pt1 = (xmin, ymin), pt2 = (xmax, ymax),color = (255,0,0), thickness = 2)
                all_boxes = fuse_bboxes(all_boxes, 'mean')
                det_times.append(time.time() - t)
                print('Detection time: ', time.time() - t)
               
                t = time.time()
                
                for i, box in enumerate(all_boxes):
                    copy = open_cv_frame.copy()
                    txt_file = open(os.path.join(SHEEP_FOLDER, vid.split('.')[0] + '.txt'), 'a')
                    ymin, xmin, xmax, ymax = box
                    erro = 0.10
                    w = xmax - xmin
                    h = ymax - ymin
              
                    ymin = int(max((ymin - h * erro),0))
                    ymax = int(min((ymax +  h * erro),height))
                    xmin = int(max((xmin - w * erro),0))
                    xmax = int(min((xmax +  w * erro),width))
                    cv.rectangle(copy, pt1 = (xmin, ymin), pt2 = (xmax, ymax),color = (0,255,0), thickness = 2)
                    
                    
                    img =open_cv_frame[ymin:ymax, xmin:xmax]


                    roll, pitch, yaw = predict_pose(img)
                    pose_times.append(time.time() - t)
                    print('Pose time: ', time.time() - t)
                    t = time.time()
                    
                    t = time.time()
                    
                    if abs(yaw) < 30:
                        fitter = fitters[0]
                    elif abs(yaw) < 60:
                        fitter = fitters[1]
                    else:
                        fitter = fitters[2]
                        
                    # if needed flip the image and fix bb
                    if yaw < 0:
                        frame = frame.mirror()
                        bb = menpo.shape.bounding_box((ymin, width - xmax), (ymax, width - xmin))
                    else:
                        bb = menpo.shape.bounding_box((ymin, xmin), (ymax, xmax))
                    # localise landmarks
                    shape = fitter.fit_from_bb(frame, bb).final_shape
                    # flip back if needed
                    if yaw < 0:
                        shape = mirror_lms(shape, frame.shape[1])
                    lms_times.append(time.time() - t)
                    
                
                    # draw lms
                    lms = []
                    for (y, x) in shape.as_vector().reshape((-1, 2)):
                        #cv2.circle(copy, (x, y), 3, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
                        lms.append((x,y))
                    for pt in lms:
                        cv2.circle(copy, pt , 3, (255, 9, 0), thickness=-1, lineType=cv2.LINE_AA)

                    lms = np.asarray(lms).astype(np.float64)
                                     
                        
                    #left: xmin top: ymin right: xmax bottom: ymax
                    img_h, img_w = img.shape[:2]                    
                    im_sq = open_cv_frame[ymin :ymax, xmin:xmax]
                    sq_h, sq_w = im_sq.shape[:2]
                    im_sq = cv.resize(im_sq, (224, 224))
                        
                    sq_lms = [] #landmarks of the cropped images
                    #image_o = im_sq.copy()
                        
                
                    for pt in lms:
                        # have into account the cropped and the resize!
                        new_pt = (int((pt[0] - xmin) * 224/sq_w), int((pt[1] - ymin) * 224/sq_h))
                        sq_lms.append(new_pt)
                        
                    
                    sq_lms = update_lms(sq_lms, yaw)
                    im_sq = cv.cvtColor(im_sq, cv.COLOR_BGR2RGB)
                     
                    if yaw < 0:
                        im_sq, sq_lms = flip_sheep(im_sq,sq_lms)
                        yaw = - yaw
                    
                    copy_sq = im_sq.copy()
                    
                    new_lms = []
                    for new_pt in sq_lms:
                        cv2.circle(copy_sq, tuple(new_pt) , 3, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
                        if new_pt[0] < 0:
                            new_lms.append((0, new_pt[1]))
                        elif new_pt[1] < 0:
                            new_lms.append((new_pt[0], 0))
                        elif new_pt[0] < 0 and new_pt[1] < 0:
                            new_lms.append((0, 0))
                        else:
                            new_lms.append((new_pt[0], new_pt[1]))
                    sq_lms = new_lms
                            
                    image_name =  vid.split('.')[0]  + '_'  + str(f_num) + '_' + str(i) +  '.png'
                    
                   
                    info = [(im_sq, sq_lms, (roll, pitch, yaw), -1, -1)]
                    
                    #cv.imwrite(os.path.join(SHEEP_FOLDER, 'sq_' + image_name), copy_sq)
                    
                    
                    HOG, geometry, pose = get_features(info, image_name)
                    features = np.concatenate((np.asarray(HOG), 
                                  np.asarray(pose), 
                                  np.asarray(geometry)), axis = 1)
                   
                    
                    
                    filename = os.path.join(os.getcwd(), 'models', 'final_pca_' + str(sheep_id) + '.pkl')
                    pca = pickle.load(open(os.path.join(os.getcwd(), filename), 'rb'))
                    x = pca.transform(np.vstack(features))
        
                    filename =  os.path.join(os.getcwd(), 'models', 'final_svm_' +  str(sheep_id) +'.pkl')
                    modelSVM = pickle.load(open(os.path.join(os.getcwd(), filename), 'rb'))
                    y_pred= modelSVM.predict(x)
                    
                    copy = cv.cvtColor(copy, cv.COLOR_BGR2RGB)
                    cv.imwrite(os.path.join(SHEEP_FOLDER, image_name), copy)
                    txt_file.write('%d, %d, %d\n' % (f_num, i, y_pred[0]))
                    txt_file.close()
                f_num += 10
                    
            
            end = time.time()
            txt_file.close()
            count += 1
            seconds = end - start
            fps.append(len(vid_frames) / seconds)
      
print('detection times: ', np.mean(det_times))
print('pose times: ', np.mean(pose_times))
print('lms times: ', np.mean(lms_times))
print('fps times: ', np.mean(fps))


cv.imshow('copy sq', copy_sq)
cv.waitKey(0)