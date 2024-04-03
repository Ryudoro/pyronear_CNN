import os
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd

def get_subdirectories(directory):
    subdirectories = sorted([os.path.join(directory, name) for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])
    return subdirectories
    
def flatten(list_of_lists):
    flattened_list = [y for x in list_of_lists for y in x]
    return flattened_list
    
def get_all_img_paths(input_dir):
    class_subdirs = get_subdirectories(input_dir)
    img_split = "images" 
    
    
    scenes_list = []
    for class_subdir in class_subdirs:
        scenes_list.append(get_subdirectories(os.path.join(class_subdir, img_split)))
    
    all_img_list = []
    for scene_dir in flatten(scenes_list):
        img_list = sorted(glob.glob(os.path.join(scene_dir, "*jpg")))
        all_img_list.append(img_list)
    flattened_all_img_list = flatten(all_img_list)
    return flattened_all_img_list



def dataset2csv(input_dir, base_path, csv_output_path):
    flattened_all_img_list = get_all_img_paths(input_dir)

    df = pd.DataFrame()
    scene_list = [] 
    transf_id_list = []
    transf_type_list = []
    transf_list = []
    img_name_id_list = []
    dataset_list = []
    img_width_list = []
    img_height_list = []
    img_path_list = []
    img_rel_path_list = []
    
    img_name_list = []
    label_path_list = []
    label_rel_path_list = []
    label_list = []  

    for img_path in flattened_all_img_list:
        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape
        img_basename = img_path.split('/')[-1]
        img_name, img_ext = os.path.splitext(img_basename)
        img_id = img_name.split("_xtrans")[0]
        ref_img_name = f"{img_id}{img_ext}"

        # tranformation info
        transf_id = img_name.split("_")[-1]
        transf_type = img_name.split("_")[-4]
        transf = img_name.split("_")[-3]
        
        # source dataset info 
        dataset_set_dirname = img_path.split('/')[-4]
        dataset_name = dataset_set_dirname.split('--')[-1]
        
        # label has_wild_fire/no_wildfire
        label = dataset_set_dirname.split('--')[0]
        
        scene_name = img_path.split('/')[-2]
        scene_index = scene_name.split("_")[1]
        scene_transformation_step = scene_name.split("_")[-1]

        # label paths
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        
        # relative path
        img_rel_path = os.path.relpath(img_path, base_path)
        label_rel_path = os.path.relpath(label_path, base_path)

        scene_list.append(scene_index) 
        transf_id_list.append(transf_id)
        transf_type_list.append(transf_type)
        transf_list.append(int(transf))
        dataset_list.append(dataset_name)
        img_name_id_list.append(ref_img_name)
        img_name_list.append(img_basename)
        img_path_list.append(img_path)
        # image shape
        img_width_list.append(img_width)
        img_height_list.append(img_height)
        img_rel_path_list.append(img_rel_path) 
        label_rel_path_list.append(label_rel_path) 
        label_path_list.append(label_path)
        label_list.append(label)

    # Dataframe
    df["scene"] = scene_list
    df["transf_id"] = transf_id_list
    df["transf_type"] = transf_type_list
    df["transf_dx"] = transf_list
    df["dataset"] = dataset_list
    df["ref_img"] = img_name_id_list
    df["img_name"] = img_name_list
    df["img_height"] = img_height_list
    df["img_width"] = img_width_list
    df["img_path"] = img_path_list 
    df["img_rel_path"] = img_rel_path_list 
    df["label_path"] = label_path_list
    df["label_rel_path"] = label_rel_path_list 
    df["label_list"] = label_list

    # Save dataframe
    df.to_csv(csv_output_path, index=False)

def get_yolo_bbox_info(csv_path):
    df_original = pd.read_csv(csv_path)
    df = df_original.copy()    
    label_paths_list = df.label_path.to_list()
    
    class_label_list = []
    yolo_bbox_list = []
    yolo_bbox_xcenter_list = []
    yolo_bbox_ycenter_list = []
    yolo_bbox_bboxw_list = []
    yolo_bbox_bboxh_list = []
    for label_path in label_paths_list:    
      with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_label, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                yolo_bbox = [x_center, y_center, bbox_width, bbox_height]
                yolo_bbox_xcenter_list.append(x_center)
                yolo_bbox_ycenter_list.append(y_center)
                yolo_bbox_bboxw_list.append(bbox_width)
                yolo_bbox_bboxh_list.append(bbox_height)
                yolo_bbox_list.append(yolo_bbox)
                class_label_list.append(class_label)

    df["class_label"] = class_label_list
    df["yolo_bbox"] = yolo_bbox_list
    df["yolo_bbox_xcenter"] = yolo_bbox_xcenter_list
    df["yolo_bbox_ycenter"] = yolo_bbox_ycenter_list
    df["yolo_bbox_bbox_width"] = yolo_bbox_bboxw_list
    df["yolo_bbox_bbox_height"] = yolo_bbox_bboxh_list
    df.to_csv(csv_path, index=False)


def main(input_dir, base_path, csv_output_path, light_csv_output_path):
    # Get all paths of images
    flattened_all_img_list = get_all_img_paths(input_dir)

    # Save info to a csv
    dataset2csv(input_dir, base_path, csv_output_path)
    
    # Add yolo bounding boxes coordinates to csv
    get_yolo_bbox_info(csv_output_path)
    
    
    # Lighten csv + save 
    df_all = pd.read_csv(csv_output_path)
    df_light = df_all.copy()
    columns_to_remove = ['img_name', 'img_path', 'label_path']
    df_light = df_light.drop(columns=columns_to_remove)
    df_light.to_csv(light_csv_output_path)


if __name__ == "__main__":
    input_dir = '/Users/marguerite/workspace_DS/pyronear_CNN/mini_dataset_translations'
    base_path = '/Users/marguerite/workspace_DS/pyronear_CNN/'
    csv_output_path = "mini_dataset_all_df.csv"
    light_csv_output_path = "mini_dataset_translation_df.csv"
    main(input_dir, base_path, csv_output_path, light_csv_output_path)


# DESCRIPTION OF THE FINAL CSV 
# scene: index of the scene (from 0 to 21)
    # scene 00 to 05 : has_wildfire, translation step : +50
    # scene 06 to 10 : has_wildfire, translation step : -50
    # scene 11 to 16 : no_wildfire, translation step : +50
    # scene 17 to 22 : no_wildfire, translation step : -50
# transf_id : index of the transform (from 0 to 9)	
# transf_type = "xtrans" (translation on horizontal x-axis)
# transf_dx : nb of pixels for the translation (values: -450, -400, ... -50, 0, 50, ....400, 450)	
# dataset: name of the dataset where the image is taken
# ref_img : name of the reference image used for translations
# img_height : image height in pixels
# img_width : image width in pixels
# img_rel_path : relative path of the image 	
# label_rel_path : relative path of the label
# label_list : label (2 values "has_wildfire", "no_wildfire")
# class_label : label (2 values 1.0 = "has_wildfire", 0.0 = "no_wildfire")	
# yolo_bbox : list of YOLO format bounding boxes [x_center, y_center, bbox_width, bbox_height]	
# yolo_bbox_xcenter: normalised x-coordinate of the center of the bounding box, YOLO format (x_center range [0, 1])
# yolo_bbox_ycenter: normalised y-coordinate of the center of the bounding box, YOLO format (y_center range [0, 1])
# yolo_bbox_bbox_width: normalised width of the bounding box, YOLO format (bbox_width range [0, 1])
# yolo_bbox_bbox_height: normalised height of the bounding box, YOLO format (bbox_height range [0, 1])	
