Landmarking
========================
* create_subsets.py: creates subsets based in the absolute yaw and the pts files needed to train the ERT system
    - save them in a "absolute_yaw" folder with 3 folders inside "0_30", "30_60" and "60_90"

* lms_notebook.ipynb: notebook to train the landmark estimation (adapted from Hewitt previous work in the field)

* shapes.py: list of different mean shapes used 

Requirements:
-------------
* github.com/friggog/dlib.git@sheep
* menpo<0.9,>=0.8
* github.com/friggog/menpofit.git@sheep
* menpowidgets
* opencv-python
