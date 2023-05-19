# Author : Loïc T

import os
import sys
import shutil
import json
import numpy as np
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from dataclasses_json import dataclass_json
import yaml
import albumentations as A
import pybboxes as pbx
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle




###################### Functions to modify directory structure ######################

def reorganize_dir_for_yolo(dir_path):
    """Reorganize the original COCO folder to fit the YOLO structure

    Args:
        dir_path (str): path of the directory to restructure
    """
    if len(os.listdir(dir_path)) == 0:
        print("Directory is empty !")
    # If a file is an image .jpg move it to the subfolder '/images'
    file_list = [file[-3:] for file in os.listdir(dir_path)]
    if "jpg" in file_list:
        if not os.path.isdir(f"{dir_path}/images"):
            os.mkdir(f"{dir_path}/images")
        for file in os.scandir(dir_path):
            if file.name[-3:] == "jpg":
                shutil.move(f"{dir_path}/{file.name}", f"{dir_path}/images/{file.name}")



###################### Functions to handdle COCO json ######################

@dataclass_json
@dataclass
class COCOCategory:
    id: int
    name: str
    supercategory: str


@dataclass_json
@dataclass
class COCOImage:
    id: int
    width: int
    height: int
    file_name: str
    license: int
    date_captured: str
    coco_url: Optional[str] = None
    flickr_url: Optional[str] = None


@dataclass_json
@dataclass
class COCOAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: List[List[float]]
    area: float
    bbox: Tuple[float, float, float, float]
    iscrowd: int


@dataclass_json
@dataclass
class COCOLicense:
    id: int
    name: str
    url: str


@dataclass_json
@dataclass
class COCOJson:
    images: List[COCOImage]
    annotations: List[COCOAnnotation]
    categories: List[COCOCategory]
    licenses: List[COCOLicense]


def load_coco_json(json_file: str) -> COCOJson:
    import json

    with open(json_file, "r") as f:
        json_data = json.load(f)

    return COCOJson.from_dict(json_data)


class COCOJsonUtility:
    """
    Class containing functions to extract information contained in the COCO json annoations file
    """

    @staticmethod
    def get_images_name(coco_data: COCOJson) -> List[str]:
        images_name_list = []
        for image in coco_data.images:
            images_name_list.append(image.file_name)
        return images_name_list

    @staticmethod
    def get_image_dataclass_by_path(coco_data: COCOJson, image_path: str) -> Optional[COCOImage]:
        for image in coco_data.images:
            if image.file_name in image_path:
                return image
        return None

    @staticmethod
    def get_annotations_by_image_id(coco_data: COCOJson, image_id: int) -> List[COCOAnnotation]:
        return [annotation for annotation in coco_data.annotations if annotation.image_id == image_id]

    @staticmethod
    def get_annotations_by_image_path(coco_data: COCOJson, image_path: str) -> Optional[List[COCOAnnotation]]:
        image = COCOJsonUtility.get_image_by_path(coco_data, image_path)
        if image:
            return COCOJsonUtility.get_annotations_by_image_id(coco_data, image.id)
        else:
            return None

    @staticmethod
    def get_dict_all_info_by_image(coco_data: COCOJson) -> dict:
        images_name_list = COCOJsonUtility.get_images_name(coco_data)
        all_info_dict = {}
        for image_name in images_name_list:
            all_info_dict[image_name] = {}
            # Copy relevant info from Image dataclass
            image_dataclass = COCOJsonUtility.get_image_dataclass_by_path(coco_data, image_path=image_name)
            all_info_dict[image_name]["id"] = image_dataclass.id
            all_info_dict[image_name]["width"] = image_dataclass.width
            all_info_dict[image_name]["height"] = image_dataclass.height
            # Copy relevant info from Annotation dataclass
            annotation_list = COCOJsonUtility.get_annotations_by_image_id(coco_data, image_id=image_dataclass.id)
            annotations_to_save = []
            for annotation in annotation_list:
                annotations_to_save.append({
                    "id" : annotation.id,
                    "category_id" : annotation.category_id,
                    "bbox" : annotation.bbox,
                    "segmentation" : annotation.segmentation,
                    "area" : annotation.area
                })
            all_info_dict[image_name]["annotations"] = annotations_to_save
        return all_info_dict


def coco_json_to_yolo_txt(dir, filename):
    """
    Genereates the text files containing the bounding boxes for each images in the right directory required for YOLOv8 training
    """
    
    coco_data = load_coco_json(f"{dir}/{filename}")
    coco_info_dict = COCOJsonUtility.get_dict_all_info_by_image(coco_data)
    
    if not os.path.isdir(f"{dir}/labels"):
        os.mkdir(f"{dir}/labels")

    # For each image, it writes a txt file with the related bounding boxes in YOLO format
    for image_name in coco_info_dict.keys():
        with open(f"{dir}/labels/{image_name[:-3]}txt", 'w') as f:
            image_id = coco_info_dict[image_name]['id']
            image_width = coco_info_dict[image_name]['width']
            image_height = coco_info_dict[image_name]['height']
            for annotation in coco_info_dict[image_name]["annotations"]:
                try:
                    bbox_yolo = pbx.convert_bbox(annotation["bbox"], from_type="coco", to_type="yolo", image_size=((image_width, image_height)))
                    f.write(f"{annotation['category_id'] - 1} {bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]}\n")
                except ValueError:
                    _, error, traceback = sys.exc_info()
                    print(f"A ValueError was raised : '{error}'\nIt was ignored because it should be due to a fake or an error on a bounding box in the COCO json, on image {image_name} with id = {image_id}\n")
                except Exception:
                    _, error, traceback = sys.exc_info()
                    raise error.with_traceback(traceback)
            f.close()
        
        # To remove last '\n' in the text files
        remove_last_backslash_n_from_text_file(f"{dir}/labels/{image_name[:-3]}txt")


def remove_last_backslash_n_from_text_file(text_file_path):
    """
    Fucntion to remove last '\n' in the text files, assuming that there is at least 1 line in the file
    """
    f = open(text_file_path, 'r')
    lines = f.readlines()
    f.close()
    if len(lines) > 1:
        f = open(text_file_path, 'w')
        f.writelines(lines[:-1])
        f.close()
        f = open(text_file_path, 'a')
        f.write(lines[-1][:-1])
        f.close()
    else:
        f = open(text_file_path, 'w')
        f.write(lines[-1][:-1])
        f.close()


###################### Functions to create YAML file for YOLO ######################

def create_yaml(which_dataset="road_holes"):
    if which_dataset == "road_holes":
        yaml_str = """
            train: data/train/images
            val: data/valid/images
            test: data/test/images

            nc: 1
            names: ['hole']
        """
    elif which_dataset == "football_players":
        yaml_str = """
            train: data/train/images
            val: data/valid/images
            test: data/test/images

            nc: 3
            names: ['ball', 'player', 'referee']
        """

    yaml_config = yaml.load(yaml_str, Loader=yaml.SafeLoader)
    with open('custom.yaml', 'w') as f:
        yaml.dump(yaml_config, f)



###################### Functions to create new augmentated image ######################

def image_augmentation(data_sub_dir, annotations_filename, augmentation_nb_per_image=4):
    """Function which performs a data augmentation for YOLOv8

    Args:
        data_sub_dir (str): sub-directory on which you want to perform the augmentation (eg. '../data/train')
        annotations_filename (str): name of the json annotations file
        augmentation_nb_per_image (int, optional): number times an image is augmented. Defaults to 4.

    Raises:
        error.with_traceback: to handdle possible errors on the bounding boxes
    """
    coco_data = load_coco_json(f"{data_sub_dir}/{annotations_filename}")
    coco_info_dict = COCOJsonUtility.get_dict_all_info_by_image(coco_data)
    
    for image_name in tqdm(coco_info_dict.keys()):
        image = cv2.imread(f"{data_sub_dir}/images/{image_name}")
        image_width = coco_info_dict[image_name]['width']
        image_height = coco_info_dict[image_name]['height']

        transform = A.Compose([
            A.Rotate(limit=90, p=0.5),
            A.RandomResizedCrop(height=image_height, width=image_width, scale=(0.6, 0.9), ratio=(0.2, 2)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            ], bbox_params=A.BboxParams(format='coco', min_area=30, min_visibility=0.1)
            )

        # Get all the bboxes as a list of list
        bboxes = [list(annotation["bbox"]) + [annotation["category_id"] - 1] for annotation in coco_info_dict[image_name]["annotations"]]
    
        for i in range(augmentation_nb_per_image):
            # Transform the image according to Albumentation composition defined above
            transformed = transform(image=image, bboxes=bboxes)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']

            # Save the image in the correct directory
            transformed_image_name = f"{data_sub_dir}/images/{image_name[:-4]}-augm{i+1}.jpg"
            cv2.imwrite(transformed_image_name, transformed_image)

            # Write and save the YOLO annotation txt file
            with open(f"{data_sub_dir}/labels/{image_name[:-4]}-augm{i+1}.txt", 'w') as f:
                # Loop through all the bboxes except the last one
                for bbox in transformed_bboxes:
                    try:
                        bbox_yolo = pbx.convert_bbox(bbox[:-1], from_type="coco", to_type="yolo", image_size=(image_width, image_height))
                        f.write(f"{bbox[4]} {bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]}\n")
                    except ValueError:
                        _, error, traceback = sys.exc_info()
                        print(f"A ValueError was raised : '{error}'\nIt was ignored because it should be due to a fake or an error on a bounding box in the COCO json, on image {image_name} with id = {coco_info_dict[image_name]['id']}\n")
                    except Exception:
                        _, error, traceback = sys.exc_info()
                        raise error.with_traceback(traceback)

            # To remove last '\n' in the text files
            remove_last_backslash_n_from_text_file(f"{data_sub_dir}/labels/{image_name[:-4]}-augm{i+1}.txt")

              

