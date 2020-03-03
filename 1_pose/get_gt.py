#%%
###############################################################################
#                                  IMPORTS
###############################################################################

import numpy as np
import cv2
import os

DATASET_PATH = '/' + os.path.join(*os.getcwd().split('/')[:-1],'dataset')  
MNE = []
method_pose = []
#%%
###############################################################################
#                              BASE SHAPE (MEAN)
###############################################################################

FULL_THREED_SHAPE = [(-0.50, 0.12, 0.81),   # left ear end             0
                     (-0.23, 0.12, 0.84),   # left ear base            1
                     (-0.20, -0.10, 0.79),  # left eye                 2
                     (0.20, -0.10, 0.79),   # right eye                3
                     (0.23, 0.12, 0.84),    # right ear base           4
                     (0.50, 0.12, 0.81),    # right ear end            5
                     (0.00, -0.51, 0.48),   # nose                     6
                     (0.00, -0.46, 0.42),   # mouth                    7
                     (0.00, -0.07, 1.01),   # top of head centre       8
                     (-0.14, -0.07, 0.99),  # top of head left         9
                     (0.14, -0.07, 0.99),   # top of head right        10
                     (-0.23, 0.00, 0.67),   # left face high 1         11
                     (0.23, 0.00, 0.67),    # right face high 1        12
                     (-0.18, -0.09, 0.49),  # left face high 2         13
                     (0.18, -0.09, 0.49),   # right face high 2        14
                     (-0.08, -0.42, 0.59),  # left face low            15
                     (0.08, -0.42, 0.59),   # right face low           16
                     (-0.04, -0.42, 0.38),  # left chin                17
                     (0.04, -0.42, 0.38),   # right chin               18
                     (-0.05, -0.47, 0.54),  # left nose top            19
                     (0.05, -0.47, 0.54),   # right nose top           20
                     (-0.39, 0.10, 0.88),   # left ear middle top      21
                     (-0.39, 0.20, 0.77),   # left ear moddle bottom   22
                     (0.39, 0.10, 0.88),    # right ear middle top     23
                     (0.39, 0.20, 0.77)]    # right ear moddle bottom  24

 
#%%
###############################################################################
#                         GET POSE GROUND TRUTH
###############################################################################
    
"""
Pose Estimation
==============================================================================

Arguments:
    * im_ex: bool; True to save square face images
    * debug: bool; True to debug (show image faces with landmarks and axis)
    
Return:
    * pose prediction: yaw_predicted, pitch_predicted and roll_predicted
    
------------------------------------------------------------------------------
"""  

def pose_estimation(im_ex=False, debug=False):
    
    """
    Fix Angles
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Fix relative to 3D coordenates
    
    Arguments:
        * r: roll (x)
        * p: pitch (y)
        * y: yaw (z)
    Return:
        * r, p, y altered    
    ---------------------------------------------------------------------------
    """  
    def fix_angles(r, p, y):
        if abs(r) > 90:
            if r > 0:
                r -= 180
            else:
                r += 180
        if p > 0:
            p -= 100
        else:
            p += 100 
        return r, p, y
    """
    Draw axis
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Arguments:
        * img: image
        * yaw, pitch, roll: pose
        * tdx, tdy: application point; origin of the axis
        * size: length of each axis
    ---------------------------------------------------------------------------
    """  
    def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180
        if tdx is not None and tdy is not None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2
        # X-Axis pointing to right. drawn in red
        x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
        y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy
        # Y-Axis drawn in green
        x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
        y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy
        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (np.sin(yaw)) + tdx
        y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy
        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)
        return img
    
    """
    Euler angles to rotation vector
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Arguments:
        * yaw, pitch, roll: pose
    ---------------------------------------------------------------------------
    """    
    def euler_to_rotvec(yaw, pitch, roll):
        yaw *= np.pi / 180
        pitch += 100
        pitch *= np.pi / 180
        roll *= np.pi / 180
        rotmat = np.zeros((3, 3))
        x = pitch
        y = roll
        z = yaw
        ch = np.cos(z)
        sh = np.sin(z)
        ca = np.cos(y)
        sa = np.sin(y)
        cb = np.cos(x)
        sb = np.sin(x)
        m00 = ch * ca
        m01 = sh *sb - ch *sa *cb
        m02 = ch *sa *sb + sh *cb
        m10 = sa
        m11 = ca *cb
        m12 = -ca *sb
        m20 = -sh *ca
        m21 = sh *sa *cb + ch *sb
        m22 = -sh *sa *sb + ch *cb
        rotmat[0, 0] = m00
        rotmat[0, 1] = m01
        rotmat[0, 2] = m02
        rotmat[1, 0] = m10
        rotmat[1, 1] = m11
        rotmat[1, 1] = m12
        rotmat[2, 0] = m20
        rotmat[2, 1] = m21
        rotmat[2, 2] = m22
        rotvec, _ = cv2.Rodrigues(rotmat)
        return rotvec
    
    
    def load_data(dataset = 'all'):
        
        path_info = os.path.join(DATASET_PATH, 'info')
        
        if dataset == 'all':
            dataset = ['farm', 'mixed', 'flickr', 'wsid']
         
        if 'farm' in dataset:
            farm = np.load(os.path.join(path_info, 'farm.npy'), allow_pickle = True).item()
            farm_list = [farm[index] for index in farm]
        else:
            farm_list = []
        
        if 'mixed' in dataset:
            mixed = np.load(os.path.join(path_info, 'mixed.npy'), allow_pickle = True).item()
            mixed_list = [mixed[index] for index in mixed]
        else:
            mixed_list = []
        
        if 'flickr' in dataset:
            flickr = np.load(os.path.join(path_info, 'flickr.npy'), allow_pickle = True).item()
            flickr_list = [flickr[index] for index in flickr]
            flickr_list = [f for f in flickr_list if f[2] != []] # Remove the deleted bounding boxes (deleted manually because they don't follow the requirements)
            flickr_list = flickr_list[:384] #Last image manually annotated
        else:
            flickr_list = []
            
        if 'wsid' in dataset:
            wsid = np.load(os.path.join(path_info, 'wsid.npy'), allow_pickle = True).item()
            wsid_list = [wsid[index] for index in wsid]
            wsid_list = [w for w in wsid_list if w[2] != []] # Remove the deleted bounding boxes
            wsid_list = wsid_list[:59]
        else:
            wsid_list = []
           
        
        data_list = farm_list + mixed_list + flickr_list + wsid_list
        data = {i : data_list[i] for i in range(len(data_list))} # Convert to dictionary
    
        return data
    #==========================================================================
    
    for dataset in ['farm', 'mixed', 'flickr', 'wsid']:
        data = load_data(dataset)
        gt_data = {}
        count = 0 # number of success cases
        
        if im_ex and not os.path.exists(os.path.join(DATASET_PATH, 'pose_and_lms')):
            os.makedirs(os.path.join(DATASET_PATH, 'pose_and_lms'))
        
        for index in range(len(data)):
            print('Progress: %d / %d' %(index, len(data)))
            (im_path, _, lms, ocs) = data[index]
            cw_path = '/' + os.path.join(*DATASET_PATH.split('/')[:-1], im_path)
            im = cv2.imread(cw_path)
                     
            # Define bounding box has de minimum box that includes all landmarks 
            # Only the successfull images will be saved
            lms = np.asarray(lms).astype(np.float64)
            x, y, w, h = [np.min(lms[:, 0]),
                          np.min(lms[:, 1]),
                          np.max(lms[:, 0]) - np.min(lms[:, 0]),
                          np.max(lms[:, 1]) - np.min(lms[:, 1])]                    
            cx, cy = (x + w / 2, y + h /2)
            s = max(w, h)
            
            ew = 0.15
            eh = 0.15
            
            #left: xmin top: ymin right: xmax bottom: ymax
            img_h, img_w = im.shape[:2]
            l = int(max(int(cx - s / 2)  - w * ew, 0))
            t = int(max(int(cy - s / 2) - h * eh, 0))
            r = int(min(int(cx + s / 2) + h * eh, img_w))
            b = int(min(int(cy + s / 2)  + w * ew, img_h))
            
            im_sq = im[t :b, l:r]
            sq_h, sq_w = im_sq.shape[:2]
            im_sq = cv2.resize(im_sq, (224, 224))
            
            sq_lms = [] #landmarks of the cropped images
            #image_o = im_sq.copy()
            
            for pt in lms:
                # have into account the cropped and the resize!
                new_pt = (int((pt[0] - l) * 224/sq_w), int((pt[1] - t) * 224/sq_h))
                sq_lms.append(new_pt)
                r = 2
                #cv2.circle(image_o, new_pt, r, (255, 0, 0), -1)
           
    
            size = im.shape[:2]
            focal_length = size[1]
            center = (size[1] /2, size[0] /2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double")
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            
            """
            cv2.imshow('landmarks - test', image_o)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
              
            # ignore the ears as they move around too much
            lms = np.asarray(lms).astype(np.float64)
            im_shape = [lms[i] for i in (list(range(1, 5)) + list(range(6, 21)))]  #if i not in ocs]
            bs_shape = [FULL_THREED_SHAPE[i] for i in (list(range(1, 5)) + list(range(6, 21)))]  #if i not in ocs]
            
            """
            SOLVE PNP
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            """
            retval, rotation_vector, translation_vector =  cv2.solvePnP(np.array(bs_shape).reshape((-1, 1, 3)),
                                           np.array(im_shape).reshape((-1, 1, 2)),
                                           camera_matrix, dist_coeffs,  None, None, False,cv2.SOLVEPNP_ITERATIVE)
             
            #if success: # If it was possible to estimate the ground truth pose        
            rmat = cv2.Rodrigues(rotation_vector)[0]
            P = np.hstack((rmat, translation_vector))
            euler_angles_degrees = cv2.decomposeProjectionMatrix(P)[6].ravel()
            roll_PnP, pitch_PnP, yaw_PnP = fix_angles(euler_angles_degrees[2],
                                              euler_angles_degrees[0],
                                              euler_angles_degrees[1])
            
            pred_PnP, _ = cv2.projectPoints(np.float32(FULL_THREED_SHAPE),
                                          rotation_vector,
                                          translation_vector,
                                          camera_matrix,
                                          dist_coeffs)
            
            pred_PnP = np.vstack(np.asarray(pred_PnP).astype(np.float64))
            PnP_shape = [pred_PnP[i] for i in (list(range(1, 5)) + list(range(6, 21)))] 
            
            """
            RANSAC
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            """
            (success,
             rotation_vector,
             translation_vector,
             inliers) = cv2.solvePnPRansac(np.array(bs_shape).reshape((-1, 1, 3)),
                                       np.array(im_shape).reshape((-1, 1, 2)),
                                       camera_matrix,
                                       dist_coeffs)

            if success: # If it was possible to estimate the ground truth pose        
                rmat = cv2.Rodrigues(rotation_vector)[0]
                P = np.hstack((rmat, translation_vector))
                euler_angles_degrees = cv2.decomposeProjectionMatrix(P)[6].ravel()
                roll_Rsac, pitch_Rsac, yaw_Rsac = fix_angles(euler_angles_degrees[2],
                                                  euler_angles_degrees[0],
                                                  euler_angles_degrees[1])
                
                pred_Rsac, _ = cv2.projectPoints(np.float32(FULL_THREED_SHAPE),
                              rotation_vector,
                              translation_vector,
                              camera_matrix,
                              dist_coeffs)
                pred_Rsac = np.vstack(np.asarray(pred_Rsac).astype(np.float64))
                Rsac_shape = [pred_Rsac[i] for i in (list(range(1, 5)) + list(range(6, 21)))] 
            
            """
            Normalized euclidean error
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            """
            
            height, width = np.max(im_shape, axis=0) - np.min(im_shape, axis=0)
            avg_edge_length = 0.5 * (height + width)
            
            PnP_shape = np.vstack(PnP_shape)
            Rsac_shape = np.vstack(Rsac_shape)
            im_shape = np.vstack(im_shape)
            PnP_mne = np.mean(np.sqrt(np.sum((PnP_shape - im_shape) ** 2, axis=-1)))/avg_edge_length
            Rsac_mne = np.mean(np.sqrt(np.sum((Rsac_shape - im_shape) ** 2, axis=-1)))/avg_edge_length
            
            
            if PnP_mne < Rsac_mne:
                roll = roll_PnP
                pitch = pitch_PnP
                yaw = yaw_PnP
                MNE.append(PnP_mne)
                method_pose.append('PnP')
            else:
                roll = roll_Rsac
                pitch = pitch_Rsac
                yaw = yaw_Rsac
                MNE.append(Rsac_mne)
                method_pose.append('Ransac')
                
            print('Yaw, pitch, roll:', yaw, pitch, roll)
            
            """
            for p in lmspts:
                cv2.circle(im_sq, tuple(p.astype(int).ravel()), 2, (255, 0, 255), -1)
    
            for p in lms:
               cv2.circle(im_sq, tuple(p.astype(int).ravel()), 2, (255, 0, 0), -1)
             
            im = draw_axis(im_sq, yaw, pitch, roll, (lms[2][0] + lms[3][0]) / 2, (lms[2][1] + lms[3][1]) / 2)
            cv2.imshow(str(count) + '_' + im_path.split('/')[-1], im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

            # output using square image path for CNN
            if MNE[-1] < 0.15:
                cv2.imshow('cropped image', im_sq)
                key = cv2.waitKey(0)
                if key == 8:
                    print('%s was deleted' % im_path)
                elif key == ord(' '):
                    gt_data[count] = (im_path, (x, y, w, h), lms, ocs, (roll, pitch, yaw))
                    count += 1  

        print('Done.')
    
        npy_filename =  dataset + '.npy'
        np.save(os.path.join(DATASET_PATH, 'pose_lms_info', npy_filename), gt_data)
    

#%%
###############################################################################
#                                   MAIN
###############################################################################

def main():
    pose_estimation(im_ex = True)
    
if __name__== "__main__":
  main()
