import glob
from datetime import datetime
import re
import pandas as pd 
import os
from tqdm import tqdm
import cv2


def get_group_time_series(input_directory, output_csv_path): 
    imgs = glob.glob(os.path.join(input_directory, "*.jpg"))
    imgs.sort()
    
    fires = {}
    fire_idx = -1
    t0 = datetime.now()

    for file in imgs:
        match = re.search(r"(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})", file)
        t = datetime.strptime(match.group(), "%Y_%m_%dT%H_%M_%S")
        if abs((t-t0).total_seconds()) > 59:
            fire_idx += 1
    
        t0 = t
    
        if fire_idx in fires.keys():
            fires[fire_idx].append(file)
        else:
            fires[fire_idx] = [file]
    
    # Extract keys and image paths
    keys = [key for key in fires.keys() for _ in range(len(fires[key]))]
    image_paths = [path for paths in fires.values() for path in paths]
    
    # Create DataFrame
    df = pd.DataFrame({'Key': keys, 'Image_Path': image_paths})
    df.to_csv(output_csv_path, index=False)

def extract_data_from_DS_fp_listoflists_multiple_bbox(dataset_dir, base_dir, input_csv_path, output_csv_path):
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    df_group = pd.read_csv(input_csv_path)
    group_list = df_group.Key.tolist()
    paths_list = df_group.Image_Path.tolist()
    
    # Initialize lists
    extracted_datetimes = []
    img_paths_list = []
    label_paths_list = []
    rel_img_paths_list = []
    rel_label_paths_list = []
    origin_dataset_name_list = []
    img_basename_list = []
    label_basename_list = []
    prefix_list = []
    datetime_str_list = []
    ext_list = []
    days_list = []
    times_list = []
    years = []
    months = []
    days = []
    hours = []
    minutes = []
    seconds = []
    img_height_list = []
    img_width_list = []
    label_list = []
    bbox_xcenter_list = []
    bbox_ycenter_list = []
    bbox_width_list = []
    bbox_height_list = []
    pred_list = []
    label_yolobbox_pred_list = []
    has_label_booleans = []
    nb_detections_list = []
    bbox_xmin_abs_list = []
    bbox_ymin_abs_list = []
    bbox_xmax_abs_list = []
    bbox_ymax_abs_list = []
    label_absyolobbox_pred_list = []
    prefix_group_id_list = []
    
    for img_path, group_id in tqdm(zip(paths_list, group_list)):
        img_name  = os.path.basename(img_path)

        label_name = img_name.split('.')[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        origin_dataset_name = os.path.dirname(os.path.dirname(label_path)).split("/")[-1]
    
        parts = img_name.split('_')
        rel_img_path = img_path.split(base_dir)[-1]
        
        # Extract the datetime part and join it to form the datetime string
        datetime_str_ext = '_'.join(parts[-5:])
        datetime_str =  datetime_str_ext.split(".")[0]
        ext = datetime_str_ext.split(".")[-1]
        prefix = img_name.split(f"_{datetime_str_ext}")[0]
        datetime_obj = datetime.strptime(datetime_str, "%Y_%m_%dT%H_%M_%S")
        year = datetime_obj.year
        month = datetime_obj.month
        day = datetime_obj.day
        hour = datetime_obj.hour
        minute = datetime_obj.minute
        second = datetime_obj.second
        prefix_group_id = f"{prefix}_group_{group_id}"
        
        # Append the values to their respective lists
        extracted_datetimes.append(datetime_obj)
        prefix_list.append(prefix)
        prefix_group_id_list.append(prefix_group_id)
        datetime_str_list.append(datetime_str)
        ext_list.append(ext)
        days_list.append(datetime_obj.date())
        times_list.append(datetime_obj.time())
        years.append(year)
        months.append(month)
        days.append(day)
        hours.append(hour)
        minutes.append(minute)
        seconds.append(second)
    
        # Extract img shape
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        label_name = img_name.split('.')[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        rel_label_path = label_path.split(base_dir)[-1]
    
        if os.path.exists(label_path):
            has_label = True
            with open(label_path, "r") as f:    
                label_multiple_detection_list = []
                x_center_multiple_detection_list = []
                y_center_multiple_detection_list = []
                bbox_width_multiple_detection_list = []
                bbox_height_multiple_detection_list = []
                pred_multiple_detection_list = []
                bboxes_multiple_detection_list = []
                x_min_multiple_detection_list = []
                y_min_multiple_detection_list = []
                x_max_multiple_detection_list = []
                y_max_multiple_detection_list = []
                bboxes_abs_multiple_detection_list = []
    
                nb_lines = 0
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        label, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                        pred = None
    
                    if len(parts) == 6:
                        label, x_center, y_center, bbox_width, bbox_height, pred = map(float, parts)
                        bbox = [label, x_center, y_center, bbox_width, bbox_height, pred]
                        
                    # Absolute bbox coords
                    x_min = x_center - bbox_width / 2
                    y_min = y_center - bbox_height / 2
                    x_max = x_center + bbox_width / 2
                    y_max = y_center + bbox_height / 2
                    x_min_real = int(x_min * width)
                    y_min_real = int(y_min * height)
                    x_max_real = int(x_max * width)
                    y_max_real = int(y_max * height)
    
                    bbox_abs = [label, x_min_real, y_min_real, x_max_real, y_max_real, pred]
                    
                    label_multiple_detection_list.append(label)
                    x_center_multiple_detection_list.append(x_center)
                    y_center_multiple_detection_list.append(y_center)
                    bbox_width_multiple_detection_list.append(bbox_width)
                    bbox_height_multiple_detection_list.append(bbox_height)
                    pred_multiple_detection_list.append(pred)
                    bboxes_multiple_detection_list.append(bbox)
                
                    x_min_multiple_detection_list.append(x_min_real)
                    y_min_multiple_detection_list.append(y_min_real)
                    x_max_multiple_detection_list.append(x_max_real)
                    y_max_multiple_detection_list.append(y_max_real)
                    bboxes_abs_multiple_detection_list.append(bbox_abs)
            
                    nb_lines +=1
    
        else:
            # print(f"Aucun fichier de label trouvé pour {img_name}, ajout d'une image sans annotations.")
            has_label = False    
            label_multiple_detection_list = [None]
            bboxes_multiple_detection_list = [None]
            x_center_multiple_detection_list = [None]
            y_center_multiple_detection_list = [None]
            bbox_width_multiple_detection_list = [None]
            bbox_height_multiple_detection_list = [None]
            pred_multiple_detection_list = [None]
            bboxes_multiple_detection_list = [None]
            nb_lines = 0

            x_min_multiple_detection_list = [None]
            y_min_multiple_detection_list = [None]
            x_max_multiple_detection_list = [None]
            y_max_multiple_detection_list = [None]
            bboxes_abs_multiple_detection_list = [None]
   
        has_label_booleans.append(has_label)
        img_paths_list.append(img_path)
        label_paths_list.append(label_path)
        rel_img_paths_list.append(rel_img_path)
        rel_label_paths_list.append(rel_label_path)
        origin_dataset_name_list.append(origin_dataset_name)
        img_basename_list.append(img_name)
        label_basename_list.append(label_name)
        img_height_list.append(height)
        img_width_list.append(width)
        label_list.append(label_multiple_detection_list)
        bbox_xcenter_list.append(x_center_multiple_detection_list)
        bbox_ycenter_list.append(y_center_multiple_detection_list)
        bbox_width_list.append(bbox_width_multiple_detection_list)
        bbox_height_list.append(bbox_height_multiple_detection_list)
        pred_list.append(pred_multiple_detection_list)
        label_yolobbox_pred_list.append(bboxes_multiple_detection_list)
        nb_detections_list.append(nb_lines)
        bbox_xmin_abs_list.append(x_min_multiple_detection_list)
        bbox_ymin_abs_list.append(y_min_multiple_detection_list)
        bbox_xmax_abs_list.append(x_max_multiple_detection_list)
        bbox_ymax_abs_list.append(y_max_multiple_detection_list)
        label_absyolobbox_pred_list.append(bboxes_abs_multiple_detection_list)
    
       
    data = {'Extracted_Datetime': extracted_datetimes,
        'Image_Path': img_paths_list,
        'Label_Path': label_paths_list,
        'Rel_Image_Path': rel_img_paths_list,
        'Rel_Label_Path': rel_label_paths_list,
        'Origin_dataset_name': origin_dataset_name_list,
        'Image_basename': img_basename_list,
        'Label_basename': label_basename_list,
        'has_label': has_label_booleans,
        'Group': group_list,
        'Prefix': prefix_list,
        'Prefix_group_id': prefix_group_id_list,
        'Datetime_Str': datetime_str_list,
        'Extension': ext_list,
        'Date': days_list,
        'Time': times_list,
        'Year': years,
        'Month': months,
        'Day': days,
        'Hour': hours,
        'Minute': minutes,
        'Second': seconds,  
        'img_height':img_height_list,
        'img_width': img_width_list,
        'label': label_list,
        'yolo_bbox_xcenter': bbox_xcenter_list,
        'yolo_bbox_ycenter': bbox_ycenter_list,
        'yolo_bbox_width': bbox_width_list,
        'yolo_bbox_height': bbox_height_list,
        'bbox_xmin_abs': bbox_xmin_abs_list,
        'bbox_ymin_abs': bbox_ymin_abs_list,
        'bbox_xmax_abs': bbox_xmax_abs_list,
        'bbox_ymax_abs': bbox_ymax_abs_list,
        'pred': pred_list,
        'label_yolobbox_pred': label_yolobbox_pred_list,
        'label_absyolobbox_pred': label_absyolobbox_pred_list,
        'nb_detections': nb_detections_list
    }
    
    df = pd.DataFrame(data)   
    df.to_csv(output_csv_path, index=False)


def extract_data_from_DS_fp_newlines_multiple_bbox(dataset_dir, base_dir, input_csv_path, output_csv_path):
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    df_group = pd.read_csv(input_csv_path)
    group_list = df_group.Key.tolist()
    paths_list = df_group.Image_Path.tolist()

    extracted_datetimes = []
    img_paths_list = []
    label_paths_list = []
    rel_img_paths_list = []
    rel_label_paths_list = []
    origin_dataset_name_list = []
    img_basename_list = []
    label_basename_list = []
    prefix_list = []
    datetime_str_list = []
    ext_list = []
    days_list = []
    times_list = []
    years = []
    months = []
    days = []
    hours = []
    minutes = []
    seconds = []
    img_height_list = []
    img_width_list = []
    label_list = []
    bbox_xcenter_list = []
    bbox_ycenter_list = []
    bbox_width_list = []
    bbox_height_list = []
    pred_list = []
    label_yolobbox_pred_list = []
    has_label_booleans = []
    nb_detections_list = []
    bbox_xmin_abs_list = []
    bbox_ymin_abs_list = []
    bbox_xmax_abs_list = []
    bbox_ymax_abs_list = []
    label_absyolobbox_pred_list = []
    prefix_group_id_list = []
    group_list_update = []
    
    for i, (img_path, group_id) in tqdm(enumerate(zip(paths_list, group_list))):
        img_name  = os.path.basename(img_path)

        label_name = img_name.split('.')[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        origin_dataset_name = os.path.dirname(os.path.dirname(label_path)).split("/")[-1]
    
        parts = img_name.split('_')
        rel_img_path = img_path.split(base_dir)[-1]
        
        # Extract the datetime part and join it to form the datetime string
        datetime_str_ext = '_'.join(parts[-5:])
        datetime_str =  datetime_str_ext.split(".")[0]
        ext = datetime_str_ext.split(".")[-1]
        prefix = img_name.split(f"_{datetime_str_ext}")[0]
        datetime_obj = datetime.strptime(datetime_str, "%Y_%m_%dT%H_%M_%S")
        year = datetime_obj.year
        month = datetime_obj.month
        day = datetime_obj.day
        hour = datetime_obj.hour
        minute = datetime_obj.minute
        second = datetime_obj.second
        prefix_group_id = f"{prefix}_group_{group_id}"

        # Extract img shape
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        label_name = img_name.split('.')[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        rel_label_path = label_path.split(base_dir)[-1]
    
        if os.path.exists(label_path):
            has_label = True
            with open(label_path, 'r') as file:
                line_count = sum(1 for line in file)
            nb_lines = line_count

            with open(label_path, "r") as f:    
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        label, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                        pred = None
    
                    if len(parts) == 6:
                        label, x_center, y_center, bbox_width, bbox_height, pred = map(float, parts)
                        bbox = [label, x_center, y_center, bbox_width, bbox_height, pred]
                        
                    # Absolute bbox coords
                    x_min = x_center - bbox_width / 2
                    y_min = y_center - bbox_height / 2
                    x_max = x_center + bbox_width / 2
                    y_max = y_center + bbox_height / 2
                    x_min_real = int(x_min * width)
                    y_min_real = int(y_min * height)
                    x_max_real = int(x_max * width)
                    y_max_real = int(y_max * height)
                    bbox_abs = [label, x_min_real, y_min_real, x_max_real, y_max_real, pred]
                    # Append the values to their respective lists
                    extracted_datetimes.append(datetime_obj)
                    prefix_list.append(prefix)
                    prefix_group_id_list.append(prefix_group_id)
                    group_list_update.append(group_id)
                    datetime_str_list.append(datetime_str)
                    ext_list.append(ext)
                    days_list.append(datetime_obj.date())
                    times_list.append(datetime_obj.time())
                    years.append(year)
                    months.append(month)
                    days.append(day)
                    hours.append(hour)
                    minutes.append(minute)
                    seconds.append(second)
                    has_label_booleans.append(has_label)
                    img_paths_list.append(img_path)
                    label_paths_list.append(label_path)
                    rel_img_paths_list.append(rel_img_path)
                    rel_label_paths_list.append(rel_label_path)
                    origin_dataset_name_list.append(origin_dataset_name)
                    img_basename_list.append(img_name)
                    label_basename_list.append(label_name)
                    img_height_list.append(height)
                    img_width_list.append(width)
                    nb_detections_list.append(nb_lines)
                    label_list.append(label)
                    bbox_xcenter_list.append(x_center)
                    bbox_ycenter_list.append(y_center)
                    bbox_width_list.append(bbox_width)
                    bbox_height_list.append(bbox_height)
                    pred_list.append(pred)
                    label_yolobbox_pred_list.append(bbox)
                    bbox_xmin_abs_list.append(x_min_real)
                    bbox_ymin_abs_list.append(y_min_real)
                    bbox_xmax_abs_list.append(x_max_real)
                    bbox_ymax_abs_list.append(y_max_real)
                    label_absyolobbox_pred_list.append(bbox_abs)

    
        else:
            # print(f"Aucun fichier de label trouvé pour {img_name}, ajout d'une image sans annotations.")
            has_label = False    
            label = None
            bbox = None
            x_center = None
            y_center = None
            bbox_width = None
            bbox_height = None
            pred = None
            nb_lines = 0
            x_min = None
            y_min = None
            x_max = None
            y_max = None
            x_min_real = None
            y_min_real = None
            x_max_real = None
            y_max_real = None
            bbox_abs = None
            
            # Append the values to their respective lists
            extracted_datetimes.append(datetime_obj)
            prefix_list.append(prefix)
            prefix_group_id_list.append(prefix_group_id)
            group_list_update.append(group_id)
            datetime_str_list.append(datetime_str)
            ext_list.append(ext)
            days_list.append(datetime_obj.date())
            times_list.append(datetime_obj.time())
            years.append(year)
            months.append(month)
            days.append(day)
            hours.append(hour)
            minutes.append(minute)
            seconds.append(second)

            has_label_booleans.append(has_label)
            img_paths_list.append(img_path)
            label_paths_list.append(label_path)
            rel_img_paths_list.append(rel_img_path)
            rel_label_paths_list.append(rel_label_path)
            origin_dataset_name_list.append(origin_dataset_name)
            img_basename_list.append(img_name)
            label_basename_list.append(label_name)
            img_height_list.append(height)
            img_width_list.append(width)
            
            nb_detections_list.append(nb_lines)
            
            label_list.append(label)
            bbox_xcenter_list.append(x_center)
            bbox_ycenter_list.append(y_center)
            bbox_width_list.append(bbox_width)
            bbox_height_list.append(bbox_height)
            pred_list.append(pred)
            label_yolobbox_pred_list.append(bbox)
         
            bbox_xmin_abs_list.append(x_min_real)
            bbox_ymin_abs_list.append(y_min_real)
            bbox_xmax_abs_list.append(x_max_real)
            bbox_ymax_abs_list.append(y_max_real)
            label_absyolobbox_pred_list.append(bbox_abs)

    data = {'Extracted_Datetime': extracted_datetimes,
        'Image_Path': img_paths_list,
        'Label_Path': label_paths_list,
        'Rel_Image_Path': rel_img_paths_list,
        'Rel_Label_Path': rel_label_paths_list,
        'Origin_dataset_name': origin_dataset_name_list,
        'Image_basename': img_basename_list,
        'Label_basename': label_basename_list,
        'has_label': has_label_booleans,
        'Group': group_list_update,
        'Prefix': prefix_list,
        'Prefix_group_id': prefix_group_id_list,
        'Datetime_Str': datetime_str_list,
        'Extension': ext_list,
        'Date': days_list,
        'Time': times_list,
        'Year': years,
        'Month': months,
        'Day': days,
        'Hour': hours,
        'Minute': minutes,
        'Second': seconds,  
        'img_height':img_height_list,
        'img_width': img_width_list,
        'label': label_list,
        'yolo_bbox_xcenter': bbox_xcenter_list,
        'yolo_bbox_ycenter': bbox_ycenter_list,
        'yolo_bbox_width': bbox_width_list,
        'yolo_bbox_height': bbox_height_list,
        'bbox_xmin_abs': bbox_xmin_abs_list,
        'bbox_ymin_abs': bbox_ymin_abs_list,
        'bbox_xmax_abs': bbox_xmax_abs_list,
        'bbox_ymax_abs': bbox_ymax_abs_list,
        'pred': pred_list,
        'label_yolobbox_pred': label_yolobbox_pred_list,
        'label_absyolobbox_pred': label_absyolobbox_pred_list,
        'nb_detections': nb_detections_list
    }

        
    df = pd.DataFrame(data)   
    df.to_csv(output_csv_path, index=False)

def extract_data(dataset_dir, base_dir, split, input_csv_path, output_csv_path): 
    images_dir = os.path.join(dataset_dir, "images", split)
    labels_dir = os.path.join(dataset_dir, "labels", split)
    
    df_group = pd.read_csv(input_csv_path)
    group_list = df_group.Key.tolist()
    paths_list = df_group.Image_Path.tolist()
    
    extracted_datetimes = []
    img_paths_list = []
    label_paths_list = []
    rel_img_paths_list = []
    rel_label_paths_list = []
    img_basename_list = []
    label_basename_list = []
    prefix_list = []
    datetime_str_list = []
    ext_list = []
    days_list = []
    times_list = []
    years = []
    months = []
    days = []
    hours = []
    minutes = []
    seconds = []
    img_height_list = []
    img_width_list = []
    label_list = []
    bbox_xcenter_list = []
    bbox_ycenter_list = []
    bbox_width_list = []
    bbox_height_list = []
    bbox_xmin_abs_list = []
    bbox_ymin_abs_list = []
    bbox_xmax_abs_list = []
    bbox_ymax_abs_list = []
    has_label_booleans = []
    prefix_group_id_list = []

    for i, (img_path, group_id) in tqdm(enumerate(zip(paths_list, group_list))):
        img_name  = os.path.basename(img_path)

        # Split the string by "_"
        parts = img_name.split('_')
        rel_img_path = img_path.split(base_dir)[-1]
        
        # Extract the datetime part and join it to form the datetime string
        datetime_str_ext = '_'.join(parts[-5:])
        datetime_str =  datetime_str_ext.split(".")[0]
        ext = datetime_str_ext.split(".")[-1]
        prefix = img_name.split(f"_{datetime_str_ext}")[0]
        datetime_obj = datetime.strptime(datetime_str, "%Y_%m_%dT%H_%M_%S")
        year = datetime_obj.year
        month = datetime_obj.month
        day = datetime_obj.day
        hour = datetime_obj.hour
        minute = datetime_obj.minute
        second = datetime_obj.second
        prefix_group_id = f"{prefix}_group_{group_id}"
        
        # Append the values to their respective lists
        extracted_datetimes.append(datetime_obj)
        prefix_list.append(prefix)
        prefix_group_id_list.append(prefix_group_id)
        datetime_str_list.append(datetime_str)
        ext_list.append(ext)
        days_list.append(datetime_obj.date())
        times_list.append(datetime_obj.time())
        years.append(year)
        months.append(month)
        days.append(day)
        hours.append(hour)
        minutes.append(minute)
        seconds.append(second)

        # Extract img shape
        img = cv2.imread(img_path)
        height, width, _ = img.shape
    
        # Extract labels  
        label_name = img_name.split('.')[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        rel_label_path = label_path.split(base_dir)[-1]
       
        if os.path.exists(label_path):
            has_label = True
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        label, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                        x_min = x_center - bbox_width / 2
                        y_min = y_center - bbox_height / 2
                        x_max = x_center + bbox_width / 2
                        y_max = y_center + bbox_height / 2
                        x_min_real = int(x_min * width)
                        y_min_real = int(y_min * height)
                        x_max_real = int(x_max * width)
                        y_max_real = int(y_max * height)

    
    
    
        else:
            # print(f"Aucun fichier de label trouvé pour {img_name}, ajout d'une image sans annotations.")
            has_label = False
            label = None
            x_center = None
            y_center = None
            bbox_width = None
            bbox_height = None
            x_min = None
            y_min = None
            x_max = None
            y_max = None
            x_min_real = None
            y_min_real = None
            x_max_real = None
            y_max_real = None

        has_label_booleans.append(has_label)
        img_paths_list.append(img_path)
        label_paths_list.append(label_path)
        rel_img_paths_list.append(rel_img_path)
        rel_label_paths_list.append(rel_label_path)
        img_basename_list.append(img_name)
        label_basename_list.append(label_name)
        img_height_list.append(height)
        img_width_list.append(width)
        label_list.append(label)
        bbox_xcenter_list.append(x_center)
        bbox_ycenter_list.append(y_center)
        bbox_width_list.append(bbox_width)
        bbox_height_list.append(bbox_height)
        bbox_xmin_abs_list.append(x_min_real)
        bbox_ymin_abs_list.append(y_min_real)
        bbox_xmax_abs_list.append(x_max_real)
        bbox_ymax_abs_list.append(y_max_real)
    
    data = {'Extracted_Datetime': extracted_datetimes,
            'Image_Path': img_paths_list,
            'Label_Path': label_paths_list,
            'Rel_Image_Path': rel_img_paths_list,
            'Rel_Label_Path': rel_label_paths_list,
            'Image_basename': img_basename_list,
            'Label_basename': label_basename_list,
            'has_label': has_label_booleans,
            'Group': group_list,
            'Prefix': prefix_list,
            'Prefix_group_id': prefix_group_id_list,
            'Datetime_Str': datetime_str_list,
            'Extension': ext_list,
            'Date': days_list,
            'Time': times_list,
            'Year': years,
            'Month': months,
            'Day': days,
            'Hour': hours,
            'Minute': minutes,
            'Second': seconds,  
            'img_height':img_height_list,
            'img_width': img_width_list,
            'yolo_bbox_xcenter': bbox_xcenter_list,
            'yolo_bbox_ycenter': bbox_ycenter_list,
            'yolo_bbox_width': bbox_width_list,
            'yolo_bbox_height': bbox_height_list,
            'bbox_xmin_abs': bbox_xmin_abs_list,
            'bbox_ymin_abs': bbox_ymin_abs_list,
            'bbox_xmax_abs': bbox_xmax_abs_list,
            'bbox_ymax_abs': bbox_ymax_abs_list,
            'label': label_list
    }

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)



def main(): 
    # Get groups 
    print("Group time series to CSV")
    input_directory = "/Users/marguerite/workspace_DS/DS_fp/images/"
    output_csv_path = "/Users/marguerite/workspace_DS/pyronear_CNN/df_group_DS_fp.csv"
    get_group_time_series(input_directory, output_csv_path)

    input_directory = "/Users/marguerite/workspace_DS/pyronear_ds_03_2024/images/train/"
    output_csv_path = "/Users/marguerite/workspace_DS/pyronear_CNN/df_group_pyronear_ds_03_2024_train.csv" 
    get_group_time_series(input_directory, output_csv_path)

    input_directory = "/Users/marguerite/workspace_DS/pyronear_ds_03_2024/images/val/"
    output_csv_path = "/Users/marguerite/workspace_DS/pyronear_CNN/df_group_pyronear_ds_03_2024_val.csv"    
    get_group_time_series(input_directory, output_csv_path)


    # Dataset DS_fp 
    print("--- Extract data from dataset DS_fp : one line per image ---")
    dataset_dir =  "/Users/marguerite/workspace_DS/DS_fp/"
    base_dir = "/Users/marguerite/workspace_DS/"
    dataset_name_DS_fp = "DS_fp"
    input_csv_path = "/Users/marguerite/workspace_DS/pyronear_CNN/df_group_DS_fp.csv"
    output_csv_path = f"df_{dataset_name_DS_fp}_listoflists_multiple_bbox.csv"
    extract_data_from_DS_fp_listoflists_multiple_bbox(dataset_dir, base_dir, input_csv_path, output_csv_path)
    
    print("--- Extract data from dataset DS_fp : new line for each multiple detection ---")
    output_csv_path = f"df_{dataset_name_DS_fp}_newlines_multiple_bbox.csv"
    extract_data_from_DS_fp_newlines_multiple_bbox(dataset_dir, base_dir, input_csv_path, output_csv_path)
   

    # Extract data for pyronear_ds_03_2024 TRAIN
    print("---  Extract data for pyronear_ds_03_2024 TRAIN ---")
    dataset_dir =  "/Users/marguerite/workspace_DS/pyronear_ds_03_2024/"
    base_dir = "/Users/marguerite/workspace_DS/"
    split = "train"
    dataset_name = "pyronear_ds_03_2024"
    input_csv_path_train = "/Users/marguerite/workspace_DS/pyronear_CNN/df_group_pyronear_ds_03_2024_train.csv"   
    output_csv_path_train = f"df_{dataset_name}_train_w_datetime_groups.csv"
    extract_data(dataset_dir, base_dir, split, input_csv_path_train, output_csv_path_train)

    # Extract data for pyronear_ds_03_2024 VAL
    print("--- Extract data for pyronear_ds_03_2024 VAL ---")
    dataset_dir =  "/Users/marguerite/workspace_DS/pyronear_ds_03_2024/"
    base_dir = "/Users/marguerite/workspace_DS/"
    split = "val"
    dataset_name = "pyronear_ds_03_2024"
    input_csv_path_val = "/Users/marguerite/workspace_DS/pyronear_CNN/df_group_pyronear_ds_03_2024_val.csv"   
    output_csv_path_val = f"df_{dataset_name}_val_w_datetime_groups.csv"
    extract_data(dataset_dir, base_dir, split, input_csv_path_val, output_csv_path_val)


if __name__=="__main__":
    main()