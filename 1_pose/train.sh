python3 code/train_hopenet.py --dataset SFLW --filename_list cross_val/pose_info_train_aug_$1.npy --output_string hopenet_$1_aug --snapshot hopenet_robust_alpha1.pkl --num_epoch 16
mv output/snapshots/hopenet_$1_aug_best.pkl models/hopenet_$1_aug.pkl
#rm output/snapshots/*
python3 code/test_hopenet.py --snapshot models/hopenet_$1_aug.pkl --dataset SFLW --filename_list cross_val/pose_info_test_aug_$1.npy
