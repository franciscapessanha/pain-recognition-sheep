Pose Estimation
=====================
* get_gt.py: Calculates the ground truth pose for each image and saves them in the folder "pose_lms_info" in the main dataset folder

* get_aug_sets.py: Performes the dataset augmentation and defines the cross-validation get_aug_sets

* train.sh: train the posenet model

* pred_pose.py: Given a model and a set of images it predicts the model and calculates the performance metrics

Requirements:
-------------
* opencv-python
* pytorch
* scikit-learn
* github.com/natanielruiz/deep-head-pose.git
