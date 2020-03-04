
#%%
###############################################################################
#                                   IMPORTS
###############################################################################

import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import glob
import argparse

# Get predictions per fold
FOLD = '0'
folder = '0.05_data'

#%%
###############################################################################
#                          SAVE ALL PREDICTIONS
###############################################################################

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

PATH_TO_CKPT = os.path.join(os.getcwd(),folder, 'final_data_' + FOLD, 'ssd_mobilenet_' + FOLD + '_10000.pb')
pb_fname = PATH_TO_CKPT

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(os.getcwd(),folder,'final_data_' + FOLD, 'annotations','label_map.pbtxt')
num_classes = 1

# If you want to test the code with your images, just add images files to the PATH_TO_TEST_IMAGES_DIR.
PATH_TO_TEST_IMAGES_DIR = os.path.join(os.getcwd(),folder, 'final_data_' + FOLD, 'images', 'test')

assert os.path.isfile(pb_fname)
assert os.path.isfile(PATH_TO_LABELS)
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.png"))
assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)
#print(TEST_IMAGE_PATHS)


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
        with tf.Session() as sess:
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


detections_folder = os.path.join(os.getcwd(), folder, 'final_data_' + FOLD, 'predictions')

if not os.path.exists(detections_folder):
    os.mkdir(detections_folder)
    
for i, image_path in enumerate(TEST_IMAGE_PATHS,0):
    print(image_path)
    image = Image.open(image_path)
    width, height = image.size
    filename = image_path.split('/')[-1].split('.')[-2]
    
    txt_file = open(os.path.join(detections_folder, filename + '.txt'), 'w')
    
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
   
    for i in range(len(output_dict['detection_scores'])):
        #if output_dict['detection_scores'][i] > 0.75:
        boxes = output_dict['detection_boxes'][i]
        ymin = boxes[0]*height
        xmin = boxes[1]*width
        ymax = boxes[2]*height
        xmax = boxes[3]*width
        
        #pt1 = (int(xmin), int(ymin))
        #pt2 = (int(xmax), int(ymax))
        #cv.rectangle(image_np, pt1, pt2, (0,0,255),10)
        confidence = output_dict['detection_scores'][i]
        txt_file.write('sheep %0.5f %d %d %d %d \n' % (confidence,xmin,ymin,xmax,ymax))

