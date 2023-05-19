# Author : Loïc T

import os
import json
import numpy as np
import pybboxes as pbx
import cv2
import torch
from matplotlib.path import Path
from tqdm.auto import tqdm

from preprocessing import load_coco_json, COCOJsonUtility
from save_and_display_image import save_overlayed_image




###################### Functions to save YOLO predictions ######################

def save_yolo_bboxes_to_json(yolo_results_in_voc_format, json_file_dir="data/test", returns=True):
    """
    Saves yolo inferred bboxes to a json file and may also return the dict if 'returns' arg is at Tue
    """
    infered_bboxes = {}
    for result in yolo_results_in_voc_format:
        image_name = result.path.split('/')[-1]
        infered_bboxes[image_name] = result.boxes.data.cpu().numpy().astype(np.float32).tolist()
    
    # Saving json file
    json_object = json.dumps(infered_bboxes, indent=4)
    with open(f"{json_file_dir}/bboxes_predicted_by_yolo.json", "w") as f:
        f.write(json_object)
    
    if returns:
        return infered_bboxes


###################### Functions for segmentation with SAM model ######################

def segment_image(SAM_mask_predictor, image_path, coco_info_dict, bboxes_in_voc=None, plus_info=False):
    """Function which segements an image. Mainly returning the masks of things found in each bounding boxes by SAM.

    Args:
        SAM_mask_predictor (SamPredictor): SamPredictor model object
        image_path (str): path of the image which you want to segment
        coco_info_dict (dict): dictionnary with all COCO information of the directory where is your image
        bboxes_in_voc (list(list)), optional): external bounding boxes in voc format (eg. inferred by a YOLOv8). Defaults to None.
        plus_info (bool, optional): Whether you want more objects returned by the function. Defaults to False.

    Returns:
        masks (np.array): masks inferred by SAM
    """
    # load bboxes from annotations in the coco_info_dict
    image_name = image_path.split('/')[-1]
    annotations = coco_info_dict[image_name]["annotations"]
    # If some external bboxes are provided we remove the possible extra info (ie. after index 3 in each list) 
    if bboxes_in_voc is not None:
        bboxes = [bbox[:4] for bbox in bboxes_in_voc]
    # If no external bboxes we use the annotated bboxes provided in the COCO json
    else:
        bboxes = [pbx.convert_bbox(bbox=annotation["bbox"], from_type="coco", to_type="voc", image_size=(coco_info_dict[image_name]["width"], coco_info_dict[image_name]["height"])) for annotation in annotations]

    # load image
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_shape = image_rgb.shape[:2]

    # and transform bboxes in a torch tensor
    input_boxes = torch.tensor(bboxes, device=SAM_mask_predictor.device)

    # pass the image to the predicator instance
    SAM_mask_predictor.set_image(image_rgb)
    # transform the boxes to the right format for the SAM model
    transformed_boxes = SAM_mask_predictor.transform.apply_boxes_torch(input_boxes, image_rgb.shape[:2])
    # run the SAM model to get the masks of the different blobs
    masks, quality_predictions, _ = SAM_mask_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)
    masks = masks.cpu().numpy()[:, 0, :, :]  # Take only mask created (because of multimask_output=False) for each of the masks => delete dimension of index 1

    if plus_info:
        ground_truth_polygon_list = [annotation["segmentation"] for annotation in annotations]
        return image_rgb, masks, input_boxes, ground_truth_polygon_list
    else:
        return masks, quality_predictions.cpu().numpy(), image_shape


def segment_images_and_measure_results(mask_predictor, bboxes, data_sub_dir, annotations_file_name, segmentated_images_saving_dir=None):
    """Function to segment all images from a directory thanks to the mask_predictor (aka SAM)

    Args:
        mask_predictor (SamPredictor): SamPredictor object
        bboxes (dict(list(list))): dictionnary with images' name as keys and list of bounding boxes (each bbox is a list of len 4) as values
        data_sub_dir (str): data sub-directory (eg. '../data/test' or '../data/val')
        annotations_file_name (str): path of the json file annotations
        segmentated_images_saving_dir (str, optional): path where you want to save the segmented images with the masks and bboxes. 
                                                        If None images segmented are not saved. Defaults to None.

    Returns:
        insights (dict): dictionnary with some performance measures
    """
    coco_data = load_coco_json(f"{data_sub_dir}/{annotations_file_name}")
    coco_info_dict = COCOJsonUtility.get_dict_all_info_by_image(coco_data)

    images_names_list = list(coco_info_dict.keys())
    insights = {}  # too save some metrics of the results
    masks_dict = {}

    # Loop on all images in the dataset
    for image_name in tqdm(images_names_list):
        insights[image_name] = {}
        
        # If some bounding boxes have been placed by YOLO 
        if bboxes[image_name]:
            image_path = f"{data_sub_dir}/images/{image_name}"
            hand_segmentation_list = [annotation["segmentation"] for annotation in coco_info_dict[image_name]["annotations"]]
            
            masks, quality_predictions, image_shape = segment_image(mask_predictor, image_path, coco_info_dict, bboxes[image_name])
            masks_dict[image_name] = masks

            if segmentated_images_saving_dir is not None:
                save_overlayed_image(image_path, image_name, segmentated_images_saving_dir, bboxes[image_name], masks, alpha=0.6, format="voc")

            if [] not in hand_segmentation_list:
                # Extracting the hand segmentation grids for each void
                hand_segmentation_grid_list = []
                for hand_segmentation_polygon in hand_segmentation_list:
                    hand_segmentation_grid_list.append(ground_truth_grid(hand_segmentation_polygon, image_shape))

                # Comparing the hand segmentation grid with the one found by the SAM model and comparing the area sizes
                ground_truth_area = []
                model_area = []
                area_position_accuracy_rate = []
                for i, grid in enumerate(hand_segmentation_grid_list):
                    ground_truth_area.append(np.sum(grid))
                    model_area.append(np.sum(masks[i, :, :]))
                    area_position_accuracy_rate.append(1 - (np.sum(grid != masks[i, :, :]) / grid.sum()))

                area_accuracy_rate = list(1 - (np.abs(np.array(model_area) - np.array(ground_truth_area)) / np.array(model_area)))

                insights[image_name]["ground_truth_area"] = ground_truth_area
                insights[image_name]["model_area"] = model_area
                insights[image_name]["area_position_accuracy_rate"] = area_position_accuracy_rate
                insights[image_name]["area_accuracy_rate"] = area_accuracy_rate
            
            else:
                # Computing the area size (in pixels) found by the SAM model
                model_area = []
                for mask in masks:
                    model_area.append(np.sum(mask))

                insights[image_name]["model_area"] = model_area
            
            insights[image_name]["quality_predictions"] = list(quality_predictions.ravel())

        # If YOLO did not find any bounding boxes
        else:
            print(f"YOLO did not find any bounding boxes on image {image_name}.\nSo, the image will not be given for inference to SAM.")
    
    json_masks = {k: v.tolist() for k, v in masks_dict.items()}
    json_object = json.dumps(json_masks, indent=4)
    with open(f"{data_sub_dir}/masks_predicted_by_sam.json", "w") as f:
        f.write(json_object)
    
    return insights, masks_dict



###################### Functions related to masks and polygons grids and areas ######################

def ground_truth_grid(polygon_points_list, image_shape):
    """
    Function which returns the ground_truth_mask given the list of points of a polygon 
    (list of points given in the 'segementation' attribute of COCO JSON)
    """

    polygon = [(polygon_points_list[i], polygon_points_list[i+1]) for i in range(0, len(polygon_points_list), 2)]

    # meshgrid
    x, y = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    # constructs the mask of the ground truth
    path = Path(polygon)
    ground_truth_mask = path.contains_points(points, radius=-0.5)
    ground_truth_mask = ground_truth_mask.reshape((image_shape[0], image_shape[1]))

    return ground_truth_mask

def polygon_area(points):
    """
    Function which computes the area value given the list of points of a polygon
    (in the format of the list of points given in the 'segementation' attribute of COCO JSON)
    """
    x, y = [points[i] for i in range(0, len(points), 2)], [points[i] for i in range(1, len(points), 2)]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


