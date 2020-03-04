Face Detection
===================================
* face_cross_val.py : 
    - will divide the dataset into 5-folds according to the source of the image (subset) and quanlitative head orientation
    - creates a folder "cross_val" with the test set image list for each fold, as well as a total image list

* get_fold.py:
    - generates a folder named "data_%f" (with f corresponding to the fold index)
    - inside "data_%f" a "images" folder is created with two subfolders "test" and "train"
    - the test and train images and annotations are copied to the respective folder

* 