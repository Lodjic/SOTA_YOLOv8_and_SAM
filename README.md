# Introduction

This repository concerns a personnal project. The objective was to experiment automatic object detection and segmentation with 2 state-of-the-art models (2023): YOLOv8 (You Only Look Once) and SAM (Segment Anything Model).


# File description

This repository is composed of 2 notebooks and 3 python files:
- `YOLO_&_SAM_exploration.ipynb` : a notebook just to understand what those 2 SOTA models are capable of
- `Automated_detection_&_segmentation.ipynb` : a notebook to experiment with public dataset how we can combine YOLOv8 and SAM to run object detection and segmentation tasks
- `preprocesing.py` : which contains functions to preprocess automatically downloaded datasets from the notebook
- `save_and_display_image.py` : which contains functions to display and save images overlayed with bounding boxes and/or segmentation masks
- `util.py` : which contains functions to run image augmentation and object segmentation on images

The notebooks have been made to be run on Google Colab with GPU enabled. They are ready-to-use notebooks. Run the `Automated_detection_&_segmentation` notebook and it will automatically download a dataset and run object detection and segmentation on it. You just need to upload the 3 python files at the root directory `/content` before running it.


# Manual Installation and run

The code was run in Python 3.10.11

## Installation

You should start with a clean virtual environment and install the requirements for the code to run. You may create a Python 3.10.11 and install the required packages `requirements.txt` (obtained from Google Colab). But you will need a GPU for quicker results visualization.
 

# Advices

## Running the notebooks

As stated above, I advise you to run the notebooks on a Google Colab with GPU enabled since I wrote and run them on Google Colab. If you run the notebooks locally, you might have to adjust them (comment out some lines and solve some minor bugs). 

## Running the automated detection and segmentation on other datasets

I ran the `Automated_detection_&_segmentation` notebook on the 2 datasets `pothole` and `soccer_players` available at : https://public.roboflow.com. You can try runnning it on other datasets by adding an dowload URL in the appropriate cell. But, for the notebook to automatically work with your dataset, you need to ensure that the dataset is downloaded in COCO json format (ie: with images and their corresponding annotations COCO json files seperated in 3 sub-directories named : 'train', 'val', 'test').

## Best performances

The best performances are usually achieved with some augmentations performed and greater number of epochs for YOLOv8 training.

## Increase the performance

To get increased results you might :
- perform more augmentations to get more data
- train the YOLOv8 model longer
- upgrade YOLOv8 and/or SAM model to a larger version. The notebook `Automated_detection_&_segmentation` only downloads weights of the smallest version of those models. You can try to download larger ones. They exist in at least 3 sizes.
