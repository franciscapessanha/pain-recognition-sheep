#%%
###############################################################################
#                                  IMPORTS
###############################################################################

import hopenet
import torchvision
import torch
from torchvision import transforms
from PIL import Image
import os
import cv2 as cv
import numpy as np
from eval import evaluate
from torch.autograd import Variable
import datasets
import utils

CV_FOLDER = os.path.join(os.getcwd(), 'cross_val')
#%%
###############################################################################
#                                  LOAD MODEL
###############################################################################

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
print('Loading posenet...', end='')
FOLD = '4'
MODEL = os.path.join(os.getcwd(), 'models','hopenet_' + str(FOLD) + '.pkl')
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
saved_state_dict = torch.load(MODEL)
model.load_state_dict(saved_state_dict)
transformations = transforms.Compose([transforms.Resize(224),
                                  transforms.CenterCrop(224), transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

pose_dataset = datasets.SFLW('', 'cross_val/pose_info_test_aug_' + str(FOLD) + '.npy', transformations)
test_loader = torch.utils.data.DataLoader(dataset=pose_dataset, batch_size =1, num_workers=2)    

#model.cuda(0) #Uncomment if the PC has a GPU
model.eval()

total = 0

idx_tensor = [idx for idx in range(66)]
#idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
idx_tensor = torch.FloatTensor(idx_tensor)

yaw_error = .0
pitch_error = .0
roll_error = .0


#%%
###############################################################################
#                                  PREDICT POSE
###############################################################################

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
   
    img = Variable(img).cuda(0) #Uncomment if the PC has a GPU
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
#                                   MAIN
###############################################################################
#def main():
    
data_out = []
 
for i, (images, labels, cont_labels, name) in enumerate(test_loader):
    #images = Variable(images).cuda(gpu)
    images = Variable(images)
    total += cont_labels.size(0)

    label_yaw = cont_labels[:, 0].float()
    label_pitch = cont_labels[:, 1].float()
    label_roll = cont_labels[:, 2].float()

    yaw, pitch, roll = model(images)

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

    # Mean absolute error
    yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
    pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
    roll_error += torch.sum(torch.abs(roll_predicted - label_roll))

    data_out.append([[yaw_predicted[0], pitch_predicted[0], roll_predicted[0]], [label_yaw[0], label_pitch[0], label_roll[0]]])
    
gt = np.vstack([a[0] for a in data_out])
pred = np.stack([a[1] for a in data_out])
evaluate(gt, pred)

print('Test error in degrees of the model on the ' + str(total) + ' test images')
print('Yaw: %.4f, Pitch: %.4f, Roll: %.4f   Ave: %.4f' % (yaw_error / total, pitch_error / total, roll_error / total, (yaw_error + pitch_error + roll_error) / (3 * total)))
