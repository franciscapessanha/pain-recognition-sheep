#%%
###############################################################################
#                                  IMPORTS
###############################################################################
import os
import cv2 as cv
import numpy as np
from scipy import ndimage

DATASET_PATH = '/' + os.path.join(*os.getcwd().split('/')[:-1],'dataset') 

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


#############################################################################################
#        Adapted from Shape Matching and Object Recognition Using Shape Contexts            #
#                     Serge Belongie, Jitendra Malik and Jan Puzicha                        #
#  code at https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sc_digits.html  #
#############################################################################################


def dist2(x, y):
    o = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            o[i, j] = np.sqrt((x[i][0] - y[j][0])**2 + (x[i][1] - y[j][1])**2)
    return o


def bookstein(X, Y, beta_k):
    N = len(X)
    Nb = len(Y)
    if N != Nb:
        raise Exception('number of landmarks must be equal')
    r2 = dist2(X, X)
    K = r2 * np.log(r2 + np.identity(N))
    P = np.hstack((np.ones((N, 1)), X))
    L = np.vstack((np.hstack((K, P)),
                   np.hstack((P.conj().transpose(), np.zeros((3, 3))))))
    V = np.hstack((Y.conj().transpose(), np.zeros((2, 3))))
    L[0:N, 0:N] = L[0:N, 0:N] + beta_k * np.identity(N)
    invL = np.linalg.inv(L)
    c = np.matmul(invL, V.conj().transpose())
    cx = c[:, 0]
    cy = c[:, 1]
    Q = np.matmul(np.matmul((c[0:N, :]).conj().transpose(), K), c[0:N, :])
    E = np.mean(np.diag(Q))
    return cx, cy, E


def tps(src, dst, to_warp, itr=5, beta_k=0.005):
    Xk = src
    n_good = len(Xk)
    for k in range(itr):
        cx, cy, E = bookstein(Xk, dst, beta_k)
        fx_aff = np.matmul(cx[n_good:n_good + 3].conj().transpose(),
                           np.vstack((np.ones((1, len(to_warp))),
                                      to_warp.conj().transpose())))
        d2 = np.clip(dist2(Xk, to_warp), 0, None)
        U = d2 * np.log(d2 + np.finfo(float).eps)
        fx_wrp = np.matmul(cx[0:n_good].conj().transpose(), U)
        fx = fx_aff + fx_wrp
        fy_aff = np.matmul(cy[n_good:n_good + 3].conj().transpose(),
                           np.vstack((np.ones((1, len(to_warp))),
                                      to_warp.conj().transpose())))
        fy_wrp = np.matmul(cy[0:n_good].conj().transpose(), U)
        fy = fy_aff + fy_wrp
        Z = np.vstack((fx, fy)).conj().transpose()
        Xk = Z[0:n_good, :]
    return Z

#%%
###############################################################################
#                        DATA AUGMENTATION FUNCTIONS
############################################################################### 

def rotate_sheep(image, points, centre, angle_max):
    angle = np.random.random() * angle_max * np.random.choice([-1, 1])
    M = cv.getRotationMatrix2D(centre, angle, 1)
    h, w = image.shape[:2]
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    new_image = cv.warpAffine(image, M, (w, h))
    # new_points = cv2.transform(points_ones, M)
    new_points = M.dot(points_ones.T).T
    min_x = np.min(new_points[:, 0])
    max_x = np.max(new_points[:, 0])
    min_y = np.min(new_points[:, 1])
    max_y = np.max(new_points[:, 1])
    bw = max_x - min_x
    bh = max_y - min_y
    new_box = [int(min_x - 0.08 * bw), int(min_y - 0.1 * bh),
               int(1.16 * bw), int(1.16 * bh)]
    # draw('rotated', new_image, new_box, new_points, 0)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return new_image, new_box, new_points


def scale_sheep(image, box, points, scale):
    new_image = cv.resize(image, (0, 0), fx=scale, fy=scale)
    new_points = []
    for p in points:
        new_points.append((int(p[0] * scale), int(p[1] * scale)))
    new_box = []
    for p in box:
        new_box.append((int(p[0] * scale), int(p[1] * scale)))
    # draw('rotated', new_image, new_box, new_points, 0)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return new_image, new_box, new_points


def transform_image(image, warped_grid, n):
    # based on https://github.com/zpincus/celltool/
    # celltool/celltool/numerics/image_warp.py
    x_steps = n + 1
    y_steps = n + 1
    ih, iw = image.shape[:2]
    new_x, new_y = np.mgrid[0:iw, 0:ih]
    x_fracs, x_indices = np.modf((x_steps - 1) * new_x / float(iw))
    y_fracs, y_indices = np.modf((y_steps - 1) * new_y / float(ih))
    x_indices = x_indices.astype(int)
    y_indices = y_indices.astype(int)
    x1 = 1 - x_fracs
    y1 = 1 - y_fracs
    ix1 = (x_indices + 1).clip(0, x_steps - 1).astype(int)
    iy1 = (y_indices + 1).clip(0, y_steps - 1).astype(int)
    t00 = warped_grid[:, :, 0][(x_indices, y_indices)]
    t01 = warped_grid[:, :, 0][(x_indices, iy1)]
    t10 = warped_grid[:, :, 0][(ix1, y_indices)]
    t11 = warped_grid[:, :, 0][(ix1, iy1)]
    transform_x = t00 * x1 * y1 + t01 * x1 * y_fracs + \
        t10 * x_fracs * y1 + t11 * x_fracs * y_fracs
    t00 = warped_grid[:, :, 1][(x_indices, y_indices)]
    t01 = warped_grid[:, :, 1][(x_indices, iy1)]
    t10 = warped_grid[:, :, 1][(ix1, y_indices)]
    t11 = warped_grid[:, :, 1][(ix1, iy1)]
    transform_y = t00 * x1 * y1 + t01 * x1 * y_fracs + \
        t10 * x_fracs * y1 + t11 * x_fracs * y_fracs
    transform = [transform_y, transform_x]
    image_out = np.zeros((image.shape[1], image.shape[0], image.shape[2]), dtype=np.uint8)
    image_out[:, :, 0] = ndimage.map_coordinates(image[:, :, 0], transform, order=1)
    image_out[:, :, 1] = ndimage.map_coordinates(image[:, :, 1], transform, order=1)
    image_out[:, :, 2] = ndimage.map_coordinates(image[:, :, 2], transform, order=1)
    image_out = cv .flip(np.rot90(image_out, -1), 1)
    return image_out


def norm_distance_between_lms(lms1, lms2):
    def normalise_lms(lms):
        lms = lms.copy()
        xmin = np.min(lms[:, 0])
        xmax = np.max(lms[:, 0])
        ymin = np.min(lms[:, 1])
        ymax = np.max(lms[:, 1])
        lms[:, 0] -= xmin
        lms[:, 1] -= ymin
        lms[:, 0] /= xmax - xmin
        lms[:, 1] /= ymax - ymin
        return lms, (xmin, ymin, xmax - xmin, ymax - ymin)

    def project_into_bb(lms, bb):
        lms = lms.copy()
        lms[:, 0] *= bb[2]
        lms[:, 1] *= bb[3]
        lms[:, 0] += bb[0]
        lms[:, 1] += bb[1]
        return lms

    lms1, lms1_bb = normalise_lms(lms1)
    lms2, lms2_bb = normalise_lms(lms2)
    d = 0
    for i, (x1, y1) in enumerate(lms1):
        x2, y2 = lms2[i]
        d += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    d /= len(lms1)

    return d, project_into_bb(lms2, lms1_bb)


def warp_sheep(image, lms,bb, ocs = [],full_data=None):
    lms = np.array(lms)
    ih, iw = image.shape[:2]
    x, y, w, h = bb
    # add screen edge to lms for warping
    lms_w = np.append(lms, [[0, 0], [iw - 1, 0], [iw - 1, ih - 1], [0, ih - 1]])
    lms_w = lms_w.reshape((-1, 2))
    # warp the goal lms
    if full_data is None:
        # if no data given then randomly permute
        lms_out = lms_w.copy()
        # L EAR
        
        if not all(x in ocs for x in [0, 21, 22]):
            magx = np.random.random() * 0.07 * w * np.random.choice([-1, 1])
            magy = np.random.random() * 0.07 * w * np.random.choice([-1, 1])
            lms_out[0][0] += magx
            lms_out[0][1] += magy
            for i in [21, 22]:
                lms_out[i][0] += magx * 0.3
                lms_out[i][1] += magy * 0.3
        
        # R EAR
        if not all(x in ocs for x in [5, 23, 24]):
            magx = np.random.random() * 0.07 * w * np.random.choice([-1, 1])
            magy = np.random.random() * 0.07 * w * np.random.choice([-1, 1])
            lms_out[5][0] += magx
            lms_out[5][1] += magy
            for i in [23, 24]:
                lms_out[i][0] += magx * 0.3
                lms_out[i][1] += magy * 0.3
        # EYES
        magx = np.random.random() * 0.02 * w * np.random.choice([-1, 1])
        magy = np.random.random() * 0.02 * w * np.random.choice([-1, 1])
        
        if 2 not in ocs:
            lms_out[2][0] += magx
            lms_out[2][1] += magy
        if 3 not in ocs:
            lms_out[3][0] -= magx
            lms_out[3][1] += magy

        # NOSE
        mag = np.random.random() * 0.05 * w * np.random.choice([-1, 1])
        for i in [6, 7, 15, 16, 17, 18, 19, 20]:
            lms_out[i][0] += mag
        lms_out[13][0] += mag / 2
        lms_out[14][0] += mag / 2
        lms_out[11][0] += mag / 4
        lms_out[12][0] += mag / 4
    else:
        # if data given select random item and warp to that shape if the difference isnt too big
        diff = 999
        c = 0
        cmax = 100
        while diff > 0.05 and c < cmax:
            (_, lms_o, _) = full_data[np.random.randint(0, len(full_data) - 2)]
            diff, lms_out = norm_distance_between_lms(lms, lms_o)
            c += 1
        if c == cmax:
            print('Couldn\'t find good warp')
            return image, bb, lms
        lms_out = np.append(lms_out, [[0, 0], [iw - 1, 0], [iw - 1, ih - 1], [0, ih - 1]])
        lms_out = lms_out.reshape((-1, 2))
    # apply tps to grid
    n = int(12 * iw / w)
    to_warp = []
    for i in np.linspace(0, iw, n + 1):
        for j in np.linspace(0, ih, n + 1):
            to_warp.append((i, j))
    to_warp = np.array(to_warp)
    warped = tps(lms_w, lms_out, np.vstack((lms, to_warp)), 1, 0.0000000001)
    warped_lms = warped[:len(lms)]
    if True in np.isnan(warped_lms).ravel():
        print('Error warping')
        return image, bb, lms
    warped_grid = warped[len(lms):].reshape((n + 1, n + 1, 2))
    to_warp = to_warp.reshape((n + 1, n + 1, 2))
    # invert the difference to tps grid
    warped_grid = to_warp + (to_warp - warped_grid)
    # warp the image by interpolation of grid
    image_out = transform_image(image, warped_grid, n)
    return image_out, bb, warped_lms
    
def get_new_pose(img, lms):
        
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
    SOLVE PNP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
            
    # Calculate new pose
    size = img.shape[:2]
    focal_length = size[1]
    center = (size[1] /2, size[0] /2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    
    # ignore the ears as they move around too much
    lms = np.asarray(lms).astype(np.float64)
    im_shape = [lms[i] for i in (list(range(1, 5)) + list(range(6, 21)))]  #if i not in ocs]
    bs_shape = [FULL_THREED_SHAPE[i] for i in (list(range(1, 5)) + list(range(6, 21)))]  #if i not in ocs]    
    
    retval, rotation_vector, translation_vector =  cv.solvePnP(np.array(bs_shape).reshape((-1, 1, 3)),
                                   np.array(im_shape).reshape((-1, 1, 2)),
                                   camera_matrix, dist_coeffs,  None, None, False,cv.SOLVEPNP_ITERATIVE)
     
    #if success: # If it was possible to estimate the ground truth pose        
    rmat = cv.Rodrigues(rotation_vector)[0]
    P = np.hstack((rmat, translation_vector))
    euler_angles_degrees = cv.decomposeProjectionMatrix(P)[6].ravel()
    roll_PnP, pitch_PnP, yaw_PnP = fix_angles(euler_angles_degrees[2],
                                      euler_angles_degrees[0],
                                      euler_angles_degrees[1])
    
    pred_PnP, _ = cv.projectPoints(np.float32(FULL_THREED_SHAPE),
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
     inliers) = cv.solvePnPRansac(np.array(bs_shape).reshape((-1, 1, 3)),
                               np.array(im_shape).reshape((-1, 1, 2)),
                               camera_matrix,
                               dist_coeffs)

    if success: # If it was possible to estimate the ground truth pose        
        rmat = cv.Rodrigues(rotation_vector)[0]
        P = np.hstack((rmat, translation_vector))
        euler_angles_degrees = cv.decomposeProjectionMatrix(P)[6].ravel()
        roll_Rsac, pitch_Rsac, yaw_Rsac = fix_angles(euler_angles_degrees[2],
                                          euler_angles_degrees[0],
                                          euler_angles_degrees[1])
        
        pred_Rsac, _ = cv.projectPoints(np.float32(FULL_THREED_SHAPE),
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
        else:
            roll = roll_Rsac
            pitch = pitch_Rsac
            yaw = yaw_Rsac
    
    else:
        roll = roll_PnP
        pitch = pitch_PnP
        yaw = yaw_PnP
    
   
    return roll, pitch, yaw

def flip_sheep(image, box, points, ocs=[]):
    h, w = image.shape[:2]
    new_image = cv.flip(image, 1)
    new_points = []
    new_points.append(points[5])
    new_points.append(points[4])
    new_points.append(points[3])
    new_points.append(points[2])
    new_points.append(points[1])
    new_points.append(points[0])
    new_points.append(points[6])
    new_points.append(points[7])
    new_points.append(points[8])
    new_points.append(points[10])
    new_points.append(points[9])
    new_points.append(points[12])
    new_points.append(points[11])
    new_points.append(points[14])
    new_points.append(points[13])
    new_points.append(points[16])
    new_points.append(points[15])
    new_points.append(points[18])
    new_points.append(points[17])
    new_points.append(points[20])
    new_points.append(points[19])
    new_points.append(points[23])
    new_points.append(points[24])
    new_points.append(points[21])
    new_points.append(points[22])
    for i, p in enumerate(new_points):
        new_points[i] = [w - p[0], p[1]]
    new_box = [w - (box[0] + box[2]), box[1], box[2], box[3]]
    # draw('initial', image, points, 0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # draw('flipped', new_image, new_points, 0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    n_ocs = []
    n_ocs.append(5) if 0 in ocs else ()
    n_ocs.append(0) if 5 in ocs else ()
    n_ocs.append(4) if 1 in ocs else ()
    n_ocs.append(1) if 4 in ocs else ()
    n_ocs.append(3) if 2 in ocs else ()
    n_ocs.append(2) if 3 in ocs else ()
    n_ocs.append(10) if 9 in ocs else ()
    n_ocs.append(9) if 10 in ocs else ()
    n_ocs.append(12) if 11 in ocs else ()
    n_ocs.append(11) if 12 in ocs else ()
    n_ocs.append(14) if 13 in ocs else ()
    n_ocs.append(13) if 14 in ocs else ()
    n_ocs.append(16) if 15 in ocs else ()
    n_ocs.append(15) if 16 in ocs else ()
    n_ocs.append(18) if 17 in ocs else ()
    n_ocs.append(17) if 18 in ocs else ()
    n_ocs.append(20) if 19 in ocs else ()
    n_ocs.append(19) if 20 in ocs else ()
    n_ocs.append(22) if 21 in ocs else ()
    n_ocs.append(21) if 22 in ocs else ()
    n_ocs.append(24) if 23 in ocs else ()
    n_ocs.append(23) if 24 in ocs else ()
    n_ocs.append(6) if 6 in ocs else ()
    n_ocs.append(7) if 7 in ocs else ()
    n_ocs.append(8) if 8 in ocs else ()
    new_points = np.array(new_points)
    return new_image, new_box, new_points

#%%
###############################################################################
#                               MAIN FUNCTION
############################################################################### 

def aug_data(box, out_i, k, path, lms, out_path, aug_opts, save = False):
    lms = lms.copy()
    im = cv.imread(path)
    h, w, _ = np.shape(im)
    # x, y, w, h
    rot, warp, flip, crop, loose = aug_opts
    # random rotation
    if rot:
        im, box, lms = rotate_sheep(im,
                                    lms,
                                    (box[0] + box[2] / 2,
                                     box[1] + box[3] / 2), 15)



    # random warp
    if warp:
        # crop for better waping
        im, box, lms = warp_sheep(im, lms, box)
    # random flip
    if flip:
        if k % 2 == 0:
            im, box, lms = flip_sheep(im, box, lms)
    
    # save image
    if save:
        #print('out path: ',os.path.join(out_path,str(out_i) + '_' + path.split('/')[-1]))
        x,y,w,h = box
        cx, cy = (x + w / 2, y + h / 2)
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
        for lm in lms:
            lm[0] -= l
            lm[1] -= t
        x -= l
        y -= t
        box = (x,y,w,h)

        # resize to square
        ih, iw = im_sq.shape[:2]
        # resize output image (some are very big) and fix lms
        s = 500 / max(iw, ih)
        im_sq = cv.resize(im_sq, (int(s * iw), int(s * ih)))
        lms = [x * s for x in lms]
        box = [int(x * s) for x in box]

    """      
    #TO DEBUG
    #=========
    image_o = im_sq.copy()
    for pt in lms:
        print(pt)
        r = 2
        point= tuple(pt.astype(int))
        cv.circle(image_o, point, r, (255, 0, 0), -1)

    cv.imshow('image from the aug_data', image_o)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
    cv.imwrite(os.path.join(out_path,str(out_i) + '_' + str(k) + '_' + path.split('/')[-1]), im_sq)
    
    lms = [tuple(l)for l in lms]

    roll, pitch, yaw =  get_new_pose(im_sq, lms)
    return os.path.join(out_path,str(out_i) + '_' + str(k) + '_' + path.split('/')[-1]), box, lms, roll, pitch, yaw