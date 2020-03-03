import os
import numpy as np

#%%
###############################################################################
#                              GENERAL UTILS
###############################################################################

"""
Find images
==============================================================================
Finds the images on the directory.

Arguments:
    * directory: path in which we want to find a file
Returns:
    * full paths: full path to each of the files (*.jpg or *.png images)
------------------------------------------------------------------------------
"""

def find_images(directory):
    files = []
    full_path = []
    for file in os.listdir(directory):
        if file.endswith('.jpg') or file.endswith('.png'):
            files += [file]
            full_path += [os.path.join(directory, file)]            
    files.sort()
    full_path.sort()
    
    
    return np.array(full_path).reshape(-1,1)

"""
Find txt files
==============================================================================
Finds txt files on the directory.

Arguments:
    * directory: path in which we want to find a file
Returns:
    * info: information on each txt files after removing the ' and '[ ]' symbols
------------------------------------------------------------------------------
"""

def find_txt(directory):
    files = []
    full_path = []
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            files += [file]
            full_path += [os.path.join(directory, file)]     
    files.sort()
    full_path.sort()

    info = [open(file,'r').readlines() for file in full_path]
    
    
    split_info = [info[i][0].split(',') for i  in range(len(info))]
    info = []
    for i in range(len(split_info)):
        img_list = []
        for j in range(len(split_info[i])):
            img_list.append(split_info[i][j].replace('[','').replace(']','').replace("'",''))
        info.append(img_list)
    return np.array(info)

"""
Find XML
==============================================================================
Finds the xml files on the directory.

Arguments:
    * directory: path in which we want to find a file
Returns:
    * full paths: full path to each of the files (*.xml)
------------------------------------------------------------------------------
"""

def find_xml(directory):
    files = []
    full_path = []
    for file in os.listdir(directory):
        if file.endswith('.xml'):
            files += [file]
            full_path += [os.path.join(directory, file)]     
    files.sort()
    full_path.sort()
  
    return np.array(full_path).reshape(-1,1)
