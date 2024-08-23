import glob
from datetime import datetime
import re
import pandas as pd 
import os
from tqdm import tqdm
import cv2
import logging
import numpy as np
import argparse

class GetTimeSeriesGroups(): 
    def __init__(self, all_data_dir:str, csv_output_dir:str, nb_seconds_to_separate_groups:str):
        """Class to group images into time series based on this condition. 
        Create new group if 2 consecutive images have datetime > time_condition_sec in seconds.

        Args:
            all_data_dir (str): Path to the main directory with all the data.
            csv_output_dir (str): Path to output directory to save the csv containing different time series separated by the time condition.
            nb_seconds_to_separate_groups (str): Time gap in seconds to separate time series. Defaults to 59.
        """
        self.all_data_dir = all_data_dir
        self.csv_output_dir = csv_output_dir
        self.nb_seconds_to_separate_groups = nb_seconds_to_separate_groups

    def get_group_time_series(self, input_directory:str, output_csv_path:str):
        """Group images into time series based on this condition.

        Args:
            input_directory (str): Path to input directory with images to separate into time series.
            output_csv_path (str): Path to save csv with time series.
        """
        imgs = glob.glob(os.path.join(input_directory, "*.jpg"))
        imgs.sort()
        
        fires = {}
        fire_idx = -1
        t0 = datetime.now()

        for file in imgs:
            match = re.search(r"(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})", file)
            t = datetime.strptime(match.group(), "%Y_%m_%dT%H_%M_%S")
            if abs((t-t0).total_seconds()) > self.nb_seconds_to_separate_groups:
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

    def process(self) -> tuple: 
        """Process 3 datasets: DS_fp, pyronear_ds_03_2024 train and validation

        Returns:
            tuple: 3 csv paths
                - output_csv_path_DS_fp : Path to csv with time series for DS_fp dataset.
                - output_csv_path_pyronear_ds_03_2024_train : Path to csv with time series for pyronear_ds_03_2024 train.
                - output_csv_path_pyronear_ds_03_2024_val : Path to csv with time series for pyronear_ds_03_2024 validation.
        """
        # Get groups 
        logging.info("  Group time series to CSV - Dataset: DS_fp.")
        input_dir_DS_fp = os.path.join(self.all_data_dir, "DS_fp/images/")
        output_csv_path_DS_fp = os.path.join(self.csv_output_dir, "df_group_DS_fp.csv")  
        self.get_group_time_series(input_dir_DS_fp, output_csv_path_DS_fp)

        logging.info("  Group time series to CSV - Dataset: pyronear_ds_03_2024 train.")
        input_dir_pyronear_ds_03_2024_train = os.path.join(self.all_data_dir, "pyronear_ds_03_2024/images/train/")
        output_csv_path_pyronear_ds_03_2024_train = os.path.join(self.csv_output_dir, "df_group_pyronear_ds_03_2024_train.csv")
        self.get_group_time_series(input_dir_pyronear_ds_03_2024_train, output_csv_path_pyronear_ds_03_2024_train)

        logging.info("  Group time series to CSV - Dataset: pyronear_ds_03_2024 val.")
        input_dir_pyronear_ds_03_2024_val = os.path.join(self.all_data_dir, "pyronear_ds_03_2024/images/val/")
        output_csv_path_pyronear_ds_03_2024_val = os.path.join(self.csv_output_dir, "df_group_pyronear_ds_03_2024_val.csv")
        self.get_group_time_series(input_dir_pyronear_ds_03_2024_val, output_csv_path_pyronear_ds_03_2024_val)

        return output_csv_path_DS_fp, output_csv_path_pyronear_ds_03_2024_train, output_csv_path_pyronear_ds_03_2024_val


class ExtractRawDataFromDataset(): 
    def __init__(self, all_data_dir: str, csv_output_dir: str, csv_path_DS_fp_groups: str, csv_path_pyronear_ds_03_2024_train_groups: str, csv_path_pyronear_ds_03_2024_val_groups: str, get_abs_bbox_coords: bool):
        """Extract raw data from datasets : read images and labels to get related paths, bbox coordinates and other info.

        Args:
            all_data_dir (str): Path to directory with all data.
            csv_output_dir (str): Path of output directory to save CSV results.
            csv_path_DS_fp_groups (str): Path to CSV with DS_fp time series, grouped using class GetTimeSeriesGroups().
            csv_path_pyronear_ds_03_2024_train_groups (str): Path to CSV with pyronear_ds_03_2024_train time series, grouped using class GetTimeSeriesGroups().
            csv_path_pyronear_ds_03_2024_val_groups (str): Path to CSV with pyronear_ds_03_2024_val time series, grouped using class GetTimeSeriesGroups().
            get_abs_bbox_coords (bool): boolean to convert YOLO bbox to absolute coordinates in the image shape.
        """
        self.all_data_dir = all_data_dir
        self.csv_output_dir = csv_output_dir
        self.csv_path_DS_fp_groups = csv_path_DS_fp_groups
        self.csv_path_pyronear_ds_03_2024_train_groups = csv_path_pyronear_ds_03_2024_train_groups 
        self.csv_path_pyronear_ds_03_2024_val_groups = csv_path_pyronear_ds_03_2024_val_groups
        self.get_abs_bbox_coords = get_abs_bbox_coords

    def extract_data(self, dataset_name: str) -> str:
        """Extract raw data from images/labels for a given dataset_name: related paths to images/labels, YOLO bbox coordinates.

        Args:
            dataset_name (str): Name of the dataset to extract raw data.

        Returns:
            str: output_csv_path, Output path of the CSV with extracted raw data.
        """
        if dataset_name == "DS_fp":
            images_dir = os.path.join(self.all_data_dir, "DS_fp/images/")
            labels_dir = os.path.join(self.all_data_dir, "DS_fp/labels")
            dataset_dir = os.path.join(self.all_data_dir, "DS_fp")
            input_csv_path = self.csv_path_DS_fp_groups
            output_csv_path = os.path.join(self.csv_output_dir, f"df_{dataset_name}_newlines_multiple_bbox.csv")

        if dataset_name == "pyronear_ds_03_2024_train": 
            images_dir = os.path.join(self.all_data_dir, "pyronear_ds_03_2024/images", "train")
            labels_dir = os.path.join(self.all_data_dir, "pyronear_ds_03_2024/labels", "train")
            dataset_dir = os.path.join(self.all_data_dir, "pyronear_ds_03_2024")
            input_csv_path = self.csv_path_pyronear_ds_03_2024_train_groups
            output_csv_path = os.path.join(self.csv_output_dir, f"df_{dataset_name}_w_datetime_groups.csv")

        if dataset_name == "pyronear_ds_03_2024_val": 
            images_dir = os.path.join(self.all_data_dir, "pyronear_ds_03_2024/images", "val")
            labels_dir = os.path.join(self.all_data_dir, "pyronear_ds_03_2024/labels", "val")
            dataset_dir = os.path.join(self.all_data_dir, "pyronear_ds_03_2024")
            input_csv_path = self.csv_path_pyronear_ds_03_2024_val_groups
            output_csv_path = os.path.join(self.csv_output_dir, f"df_{dataset_name}_w_datetime_groups.csv")
       
        df_group = pd.read_csv(input_csv_path)
        group_list = df_group.Key.tolist()
        paths_list = df_group.Image_Path.tolist()

        extracted_datetimes = []
        dataset_prefix_group_id_list = []
        
        rel_img_paths_list = []
        rel_label_paths_list = []
        
        img_basename_list = []
        label_basename_list = []
        
        origin_dataset_name_list = []
        prefix_list = []
        prefix_group_id_list = []
        group_list_update = []

        datetime_str_list = []
        ext_list = []
        dates_list = []
        times_list = []

        img_height_list = []
        img_width_list = []

        has_label_booleans = []
        label_list = []
        bbox_xcenter_list = []
        bbox_ycenter_list = []
        bbox_width_list = []
        bbox_height_list = []
        pred_list = []
        nb_detections_list = []

        if self.get_abs_bbox_coords: 
            bbox_xmin_abs_list = []
            bbox_ymin_abs_list = []
            bbox_xmax_abs_list = []
            bbox_ymax_abs_list = []

        for i, (img_path, group_id) in tqdm(enumerate(zip(paths_list, group_list))):
            # Extract image basename and related image path
            img_name  = os.path.basename(img_path)
            rel_img_path = img_path.split(self.all_data_dir)[-1]

            # Extract label name and related path
            label_name = img_name.split('.')[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            rel_label_path = label_path.split(self.all_data_dir)[-1]
            
            # Extract the datetime part and join it to form the datetime string
            parts = img_name.split('_')
            datetime_str_ext = '_'.join(parts[-5:])
            datetime_str =  datetime_str_ext.split(".")[0]
            datetime_obj = datetime.strptime(datetime_str, "%Y_%m_%dT%H_%M_%S")

            # Get dataset_prefix_group_id to filter temporal series
            prefix = img_name.split(f"_{datetime_str_ext}")[0]
            prefix_group_id = f"{prefix}_group_{group_id:04}"
            dataset_prefix_group_id = f"{dataset_name}_{prefix_group_id}"

            # Get image extension
            ext = datetime_str_ext.split(".")[-1]

            # Extract img shape
            img = cv2.imread(img_path)
            height, width, _ = img.shape

            # Extract bbox coordinates if label_path exists
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
                            
                        # Absolute bbox coords
                        if self.get_abs_bbox_coords: 
                            x_min = x_center - bbox_width / 2
                            y_min = y_center - bbox_height / 2
                            x_max = x_center + bbox_width / 2
                            y_max = y_center + bbox_height / 2
                            x_min_real = int(x_min * width)
                            y_min_real = int(y_min * height)
                            x_max_real = int(x_max * width)
                            y_max_real = int(y_max * height)
                        
                        # Append the values to their respective lists
                        extracted_datetimes.append(datetime_obj)
                        dataset_prefix_group_id_list.append(dataset_prefix_group_id)

                        rel_img_paths_list.append(rel_img_path)
                        rel_label_paths_list.append(rel_label_path)
                        img_basename_list.append(img_name)
                        label_basename_list.append(label_name)

                        origin_dataset_name_list.append(dataset_name)
                        prefix_list.append(prefix)
                        prefix_group_id_list.append(prefix_group_id)
                        group_list_update.append(group_id)

                        datetime_str_list.append(datetime_str)
                        ext_list.append(ext)
                        dates_list.append(datetime_obj.date())
                        times_list.append(datetime_obj.time())
                        
                        img_height_list.append(height)
                        img_width_list.append(width)
                        has_label_booleans.append(has_label)
                        label_list.append(label)
                        bbox_xcenter_list.append(x_center)
                        bbox_ycenter_list.append(y_center)
                        bbox_width_list.append(bbox_width)
                        bbox_height_list.append(bbox_height)
                        pred_list.append(pred)
                        nb_detections_list.append(nb_lines)

                        if self.get_abs_bbox_coords: 
                            bbox_xmin_abs_list.append(x_min_real)
                            bbox_ymin_abs_list.append(y_min_real)
                            bbox_xmax_abs_list.append(x_max_real)
                            bbox_ymax_abs_list.append(y_max_real)
            else:
                # print(f"Aucun fichier de label trouvé pour {img_name}, ajout d'une image sans annotations.")
                has_label = False    
                label = None
                x_center = None
                y_center = None
                bbox_width = None
                bbox_height = None
                pred = None
                nb_lines = 0
                
                if self.get_abs_bbox_coords:
                    x_min = None
                    y_min = None
                    x_max = None
                    y_max = None
                    x_min_real = None
                    y_min_real = None
                    x_max_real = None
                    y_max_real = None
                
                # Append the values to their respective lists
                extracted_datetimes.append(datetime_obj)
                dataset_prefix_group_id_list.append(dataset_prefix_group_id)

                rel_img_paths_list.append(rel_img_path)
                rel_label_paths_list.append(rel_label_path)
                img_basename_list.append(img_name)
                label_basename_list.append(label_name)

                origin_dataset_name_list.append(dataset_name)
                prefix_list.append(prefix)
                prefix_group_id_list.append(prefix_group_id)
                group_list_update.append(group_id)

                datetime_str_list.append(datetime_str)
                ext_list.append(ext)
                dates_list.append(datetime_obj.date())
                times_list.append(datetime_obj.time())

                img_height_list.append(height)
                img_width_list.append(width)
                has_label_booleans.append(has_label)
                label_list.append(label)
                bbox_xcenter_list.append(x_center)
                bbox_ycenter_list.append(y_center)
                bbox_width_list.append(bbox_width)
                bbox_height_list.append(bbox_height)
                pred_list.append(pred)
                nb_detections_list.append(nb_lines)
            
                if self.get_abs_bbox_coords: 
                    bbox_xmin_abs_list.append(x_min_real)
                    bbox_ymin_abs_list.append(y_min_real)
                    bbox_xmax_abs_list.append(x_max_real)
                    bbox_ymax_abs_list.append(y_max_real)

        data = {'Extracted_Datetime': extracted_datetimes,
            'Dataset_prefix_group_id': dataset_prefix_group_id_list,
            'Rel_Image_Path': rel_img_paths_list,
            'Rel_Label_Path': rel_label_paths_list,
            'Image_basename': img_basename_list,
            'Label_basename': label_basename_list,
            'Origin_dataset_name': origin_dataset_name_list,
            'Prefix_group_id': prefix_group_id_list,
            'Prefix': prefix_list,
            'Group': group_list_update,    
            'Datetime_Str': datetime_str_list,
            'Extension': ext_list,
            'Date': dates_list,
            'Time': times_list,
            'img_height':img_height_list,
            'img_width': img_width_list,
            'has_label': has_label_booleans,
            'raw_label': label_list,
            'yolo_bbox_xcenter': bbox_xcenter_list,
            'yolo_bbox_ycenter': bbox_ycenter_list,
            'yolo_bbox_width': bbox_width_list,
            'yolo_bbox_height': bbox_height_list,
            'pred': pred_list,
            'nb_detections': nb_detections_list
        }

        if self.get_abs_bbox_coords:
            data.update({
                'bbox_xmin_abs': bbox_xmin_abs_list,
                'bbox_ymin_abs': bbox_ymin_abs_list,
                'bbox_xmax_abs': bbox_xmax_abs_list,
                'bbox_ymax_abs': bbox_ymax_abs_list
            })
            
        df = pd.DataFrame(data)   
        df.to_csv(output_csv_path, index=False)

        return output_csv_path

    def process_data(self) -> tuple:
        """Process the 3 datasets to get their raw data.

        Returns:
            tuple: 3 paths to CSV with extracted raw data from the 3 datasets.
                - csv_path_data_ds_fp: Path to csv with extracted raw data from dataset DS_fp.
                - csv_path_data_train: Path to csv with extracted raw data from dataset pyronear_ds_03_2024_train.
                - csv_path_data_val: Path to csv with extracted raw data from dataset pyronear_ds_03_2024_val.
        """
        logging.info(" Extract data from dataset DS_fp : multiple labels per image are saved in a new row.")
        csv_path_data_ds_fp = self.extract_data("DS_fp")

        logging.info(" Extract data for pyronear_ds_03_2024 TRAIN")
        csv_path_data_train = self.extract_data("pyronear_ds_03_2024_train")

        logging.info("  Extract data for pyronear_ds_03_2024 VAL")
        csv_path_data_val = self.extract_data("pyronear_ds_03_2024_val")

        return csv_path_data_ds_fp, csv_path_data_train, csv_path_data_val


class MergeCSV(): 
    def __init__(self, all_data_dir: str, csv_output_dir: str, csv_path_data_ds_fp: str, csv_path_data_train: str, csv_path_data_val: str):
        """_summary_

        Args:
            all_data_dir (str): Path to directory with all data.
            csv_output_dir (str): Path of output directory to save final CSV with merging the data from the 3 datasets.
            csv_path_data_ds_fp (str): Path to csv with extracted raw data from dataset DS_fp.
            csv_path_data_train (str): Path to csv with extracted raw data from dataset pyronear_ds_03_2024_train.
            csv_path_data_val (str): Path to csv with extracted raw data from dataset pyronear_ds_03_2024_val.
        """
        self.all_data_dir = all_data_dir
        self.csv_output_dir = csv_output_dir
        self.csv_path_data_ds_fp = csv_path_data_ds_fp
        self.csv_path_data_train = csv_path_data_train 
        self.csv_path_data_val = csv_path_data_val
        self.merged_csv_path = os.path.join(self.csv_output_dir, "0_raw_data_pyronear_ds_03_2024_train_val_DS_fp_temporal_dataset.csv")
        self.df_train = pd.read_csv(self.csv_path_data_train)
        self.df_val = pd.read_csv(self.csv_path_data_val)
        self.df_ds_fp = pd.read_csv(self.csv_path_data_ds_fp)

    def create_new_label(self):
        # In the dataset "df_pyronear_ds_03_2024", when an image is labeled with a bbox, the label is 0 instead of 1
        # We don't fill NaN when there is missing labels
        self.df_train['new_label'] = self.df_train['raw_label'].replace(0.0, 1.0)
        self.df_val['new_label'] = self.df_val['raw_label'].replace(0.0, 1.0)
        self.df_ds_fp['new_label'] = self.df_ds_fp['raw_label']

    def update_columns(self):
        self.df_train['pred'] = np.nan
        self.df_val['pred'] = np.nan

        self.df_train['nb_detections'] = self.df_train['raw_label'].replace(0.0, 1.0).fillna(0)
        self.df_val['nb_detections'] = self.df_val['raw_label'].replace(0.0, 1.0).fillna(0)
        
    def order_columns(self): 
        desired_order = ['Extracted_Datetime', 'Dataset_prefix_group_id', 'Rel_Image_Path',
        'Rel_Label_Path', 'Image_basename',
        'Label_basename', 'Origin_dataset_name', 'Prefix_group_id', 'Prefix', 'Group', 
        'Datetime_Str', 'Extension', 'Date', 'Time', 'img_height', 'img_width', 'has_label', 'raw_label', 'new_label', 
        'yolo_bbox_xcenter', 'yolo_bbox_ycenter', 'yolo_bbox_width',
        'yolo_bbox_height', 'pred', 'nb_detections']

        self.df_train = self.df_train[desired_order]
        self.df_val = self.df_val[desired_order]
        self.df_ds_fp = self.df_ds_fp[desired_order]


    def process_merge(self):
        logging.info(" Create new labels from raw labels.")
        self.create_new_label()

        logging.info(" Update columns with pred and nb_detections for pyronear_ds_03_2024 data.")
        self.update_columns()

        logging.info(" Order columns for the 3 dataframes.")
        self.order_columns()

        logging.info(" Merge dataframes.")
        merged_df = pd.concat([self.df_train, self.df_val, self.df_ds_fp], ignore_index=True)
        merged_df.to_csv(self.merged_csv_path, index=False)
        logging.info(f" Merged dataframe is saved: {self.merged_csv_path}")


def main(): 
    parser = argparse.ArgumentParser(description="Get raw data from datasets DS_fp and pyronear_ds_03_2024 train / val.")
    parser.add_argument("-i", "--data_dir", type=str, default="/Users/marguerite/workspace_DS/",
                        help="Path to directory with datasets.")
    parser.add_argument("-o", "--csv_output_dir", type=str, default="/Users/marguerite/workspace_DS/pyronear_CNN/csv/",
                        help="Output directory to save csv files with extracted data.")
    parser.add_argument("-n", "--nb_seconds_to_separate_groups", type=int, default=59,
                        help="Number of seconds to separate temporal groups.")
    parser.add_argument("--log_file", type=str, default="",
                        help="Path to the log file. Logs to console if not specified.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level.")
    
    args = parser.parse_args()

    if args.log_file:
        logging.basicConfig(filename=args.log_file, level=getattr(logging, args.log_level),
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=getattr(logging, args.log_level),
                            format='%(asctime)s - %(levelname)s - %(message)s')

    
    # Split time series with a condition on nb_seconds_to_separate_groups 
    # Outputs 3 CSV with 2 columns : group_index | filepath
    logging.info("- 1/3 - Split time series with condition on datetime.")
    data_dir =  args.data_dir
    csv_output_dir = args.csv_output_dir
    nb_seconds_to_separate_groups = args.nb_seconds_to_separate_groups

    if not os.path.exists(csv_output_dir): 
        logging.info(f"Creating output directory to save csv files: {csv_output_dir}")
        os.makedirs(csv_output_dir, exist_ok=True)

    get_time_series_groups = GetTimeSeriesGroups(data_dir, csv_output_dir, nb_seconds_to_separate_groups)
    csv_path_DS_fp_groups, csv_path_train_groups, csv_path_val_groups = get_time_series_groups.process()
    
    # Extract raw data for each CSV file to get bbox
    logging.info("- 2/3 - Extract raw data for each CSV file to get yolo bbox.")
    get_abs_bbox_coords = False 
    extract_raw_data = ExtractRawDataFromDataset(data_dir, csv_output_dir, csv_path_DS_fp_groups, csv_path_train_groups, csv_path_val_groups, get_abs_bbox_coords)
    csv_path_data_ds_fp, csv_path_data_train, csv_path_data_val = extract_raw_data.process_data()

    # Update label, order columsn and merge the 3 csv files 
    logging.info("- 3/3 - Update label, order columns and merge the 3 csv files.")
    merge = MergeCSV(data_dir, csv_output_dir, csv_path_data_ds_fp, csv_path_data_train, csv_path_data_val)
    merge.process_merge()


if __name__=="__main__":
    main()