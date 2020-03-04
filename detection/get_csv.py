"""
Usage:
# Create train data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

in our case:
python xml_to_csv.py -i data/images/train -o data/annotations/train_labels.csv

# Create test data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv

in our case:
python xml_to_csv.py -i data/images/test -o data/annotations/test_labels.csv
"""

import os
import glob
import pandas as pd
import argparse
import numpy as np

def npy_to_csv(path):
    
    classes_names = []
    info_list = []
    for npy_file in glob.glob(path + "/*.npy"):
        info = np.load(npy_file, allow_pickle = True)
        
        for box in info:
            filename,class_,width,height,(xmin,ymin,xmax,ymax) = box
            value = (
                filename,
                width,
                height,
                "sheep",
                xmin,
                ymin,
                xmax,
                ymax
            )
            info_list.append(value)

    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
        
    info_df = pd.DataFrame(info_list, columns=column_name)
    classes_names = ["sheep"]
    classes_names.sort()
    return info_df, classes_names


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Sample TensorFlow XML-to-CSV converter"
    )
    parser.add_argument(
        "-i",
        "--inputDir",
        help="Path to the folder where the input .xml files are stored",
        type=str,
    )
    parser.add_argument(
        "-o", "--outputFile", help="Name of output .csv file (including path)", type=str
    )

    parser.add_argument(
        "-l",
        "--labelMapDir",
        help="Directory path to save label_map.pbtxt file is specified.",
        type=str,
        default="",
    )

    args = parser.parse_args()

    if args.inputDir is None:
        args.inputDir = os.getcwd()
    if args.outputFile is None:
        args.outputFile = args.inputDir + "/labels.csv"

    assert os.path.isdir(args.inputDir)
    os.makedirs(os.path.dirname(args.outputFile), exist_ok=True)
    info_df, classes_names = npy_to_csv(args.inputDir)
    info_df.to_csv(args.outputFile, index=None)
    print("Successfully converted xml to csv.")
    if args.labelMapDir:
        os.makedirs(args.labelMapDir, exist_ok=True)
        label_map_path = os.path.join(args.labelMapDir, "label_map.pbtxt")
        print("Generate `{}`".format(label_map_path))

        # Create the `label_map.pbtxt` file
        pbtxt_content = ""
        for i, class_name in enumerate(classes_names):
            pbtxt_content = (
                pbtxt_content
                + "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(
                    i + 1, class_name
                )
            )
        pbtxt_content = pbtxt_content.strip()
        with open(label_map_path, "w") as f:
            f.write(pbtxt_content)


if __name__ == "__main__":
    main()
