# Author : Loïc T

import os
import json
import numpy as np
import pybboxes as pbx
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.path import Path

from preprocessing import load_coco_json, COCOJsonUtility




###################### Functions to display images ######################

def display_random_images_from_dir(dir, nb_images_diplayed=9):
    """
    Displays some random images (mutiple of 3 above or equal to 6 to not have any bug) from a particular directory.
    """
    fig, ax = plt.subplots(int(nb_images_diplayed/3), 3, figsize=(12, 8))
    image_list = os.listdir(dir)
    image_list = [file_name for file_name in image_list if file_name[-3:] == "jpg"]
    for i, idx in enumerate(np.random.choice(len(image_list), size=nb_images_diplayed, replace=False)):
        image_rgb = plt.imread(f"{dir}/{image_list[idx]}")
        ax[i//3, i%3].imshow(image_rgb)
        ax[i//3, i%3].set_title(f"{image_list[idx][:12]}")
        ax[i//3, i%3].axis('off')
    plt.show()


def display_image_with_detections(image_path, ax=None, bboxes=None, bboxes_inferred=None, masks=None, 
                                  masks_inferred=None, ground_truth_polygon_list=None):
    """Displays an image with annotated bounding boxes in green (if provided), inferred bounding boxes in red (if provided), 
    annotated masks in yellow (if provided), inferred masks in pastel purple (if provided) and annoated polygons (segmenting line)
    in blue (if provided).

    Args:
        image_path (str): path of the image you want to display
        ax (plt.gca(), optional): an ax if you want to display the image aside to another. Defaults to None.
        bboxes (list or np.array, optional): annotated bounding boxes. Defaults to None.
        bboxes_inferred (list or np.array, optional): inferred bounding boxes. Defaults to None.
        masks (list or np.array, optional): annotated masks. Defaults to None.
        masks_inferred (list or np.array, optional): inferred masks. Defaults to None.
        ground_truth_polygon_list (list or np.array, optional): annotated polygons (segmenting line). Defaults to None.
    """
    # load image
    image = cv2.imread(image_path)

    # add the boxes from the dataset in green
    if bboxes is not None:
        image = overlay_image_with_bboxes(image, bboxes, format="voc", BGR_color=(0, 255, 0))
    # add the predicted boxes from the dataset in red
    if bboxes_inferred is not None:
        image = overlay_image_with_bboxes(image, bboxes_inferred, format="voc", BGR_color=(0, 0, 255))
    # add the ground truth with closed blue curves
    if ground_truth_polygon_list is not None:
        overlay_image_with_polygon(image, ground_truth_polygon_list, BGR_color=(255, 0, 0))   
    # add the masks with in pastel yellow and 40% transparency
    if masks is not None:
        overlay_image_with_masks(image, masks, BGR_color=(195, 247, 253), alpha=0.6)
    # add the predicted masks with in pastel purple and 40% transparency
    if masks_inferred is not None:
        overlay_image_with_masks(image, masks_inferred, BGR_color=(255, 164, 178), alpha=0.6)
    
    # Turn into RGB for matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # diplay the image
    if ax is None:
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(image)


def display_image_detections_comparision(data_sub_dir, image_name, annotations_file_name, inferred_masks, inferred_bboxes, bboxes_format="voc"):
    """
    Display the image (image_name) overlayed with its detections and compared them to the annotated bounding boxes.
    So, it displays the image overlayed with the annotated bounding boxes, inferred bounding boxes and the inferred masks.
    """
    coco_data = load_coco_json(f"{data_sub_dir}/{annotations_file_name}")
    coco_info_dict = COCOJsonUtility.get_dict_all_info_by_image(coco_data)

    image_height = coco_info_dict[image_name]['height']
    image_width = coco_info_dict[image_name]['width']

    # Get the dataset bboxes and segmentation line from the annoation file
    dataset_annotation_bboxes = [pbx.convert_bbox(list(annotation["bbox"]), from_type="coco", to_type="voc", image_size=(image_width, image_height)) for annotation in coco_info_dict[image_name]["annotations"]]
    dataset_annotation_segmentation_lines = [list(annotation["segmentation"]) for annotation in coco_info_dict[image_name]["annotations"]]
    infered_bboxes_only = [inferred_bbox[:4] for inferred_bbox in inferred_bboxes[image_name]]

    display_image_with_detections(f"{data_sub_dir}/images/{image_name}", bboxes=dataset_annotation_bboxes, bboxes_inferred=infered_bboxes_only, masks_inferred=inferred_masks[image_name], ground_truth_polygon_list=dataset_annotation_segmentation_lines)



###################### Functions to save images ######################

def overlay_image_with_bboxes(cv2_image, bboxes, format="voc", BGR_color=(0, 0, 255), image_size=None):
    """
    Takes a cv2 opened image (in BGR) and add on top all the bounding boxes located in the bboxes argument 
    and returns the cv2_image overlayed.
    """
    for bbox in bboxes:
        if format == "yolo":
            bbox_voc = pbx.convert_bbox(bbox=bbox, from_type="yolo", to_type="voc", image_size=image_size)
        elif format == "coco":
            bbox_voc = pbx.convert_bbox(bbox=bbox, from_type="coco", to_type="voc", image_size=image_size)
        elif format == "voc":
            bbox_voc = bbox
        cv2_image = cv2.rectangle(cv2_image, (round(bbox_voc[0]), round(bbox_voc[1])), 
                                  (round(bbox_voc[2]), round(bbox_voc[3])), color=BGR_color, thickness=2)
    return cv2_image

def overlay_image_with_masks(cv2_image, masks, BGR_color=(255, 164, 178), alpha=0.6):   # (255, 164, 178) = pastel purple in BGR
    """
    Takes a cv2 opened image (in BGR) and add on top with some transparent color all the (boolean) masks 
    located in the masks argument and returns the cv2_image overlayed.  
    """
    color = np.array(BGR_color).astype(np.uint8)  
    for mask in masks:
        mask_image = mask[:, :, np.newaxis] * color.reshape((1, 1, -1))
        cv2_image[mask] = cv2.addWeighted(cv2_image, 1-alpha, mask_image, alpha, 0)[mask]
    return cv2_image

def overlay_image_with_polygon(cv2_image, list_of_polygons_from_coco_json, BGR_color=(255, 0, 0)):
    """
    Takes a cv2 opened image (in BGR) and add on top all the polygons located in the list_of_polygons_from_coco_json
    argument and returns the cv2_image overlayed.  
    """
    for vertices_list in list_of_polygons_from_coco_json:
        if vertices_list:
            vertices_array = np.array([[vertices_list[i], vertices_list[i+1]] for i in range(0, len(vertices_list), 2)]).reshape((-1, 1, 2))
            cv2_image = cv2.polylines(cv2_image, [vertices_array], isClosed=True, color=BGR_color, thickness=1)
    return cv2_image

def save_overlayed_image(images_path, image_name, saving_dir, bboxes=None, masks=None, alpha=0.6, format="voc", 
                         BGR_color_bboxes=(0, 0, 255), BGR_color_masks=(255, 164, 178), image_size=None):
    """
    Saves into the directory 'saving_dir' the image overlayed with bounding boxes and masks if their are some.
    """
    image = cv2.imread(images_path)
    if bboxes:
        image = overlay_image_with_bboxes(image, bboxes, format=format, BGR_color=BGR_color_bboxes, image_size=image_size)
    if masks:
        image = overlay_image_with_masks(image, masks, BGR_color=BGR_color_masks, alpha=alpha)
    cv2.imwrite(f"{saving_dir}/{image_name[:-4]}-segmented.jpg", image)
   
def overlay_images_with_bboxes_from_txt_files(images_dir, labels_dir):
    """
    Save images from 'images_dir' to another sub-directory with their annoated bounding boxes contained in the YOLOv8 text files
    """
    for image_name in os.listdir(images_dir):
        if image_name[-3:] == "jpg":
            image = cv2.imread(f"{images_dir}/{image_name}")
            bboxes = []
            with open(f"{labels_dir}/{image_name[:-4]}.txt", 'r') as f:
                bbox_str_list = f.read().split('\n')
                for bbox_str in bbox_str_list:
                    bbox_coord_list = bbox_str.split(" ")
                    bboxes.append([float(bbox_coord_list[1]), float(bbox_coord_list[2]), float(bbox_coord_list[3]), float(bbox_coord_list[4])])
            cv2_image = overlay_image_with_bboxes(image, bboxes)
        cv2.imwrite(f"{images_dir}/overlayed/{image_name[:-4]}-overlayed.jpg", cv2_image)

