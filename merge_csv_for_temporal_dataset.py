import os
import pandas as pd 
import numpy as np
import cv2
import matplotlib.pyplot as plt

def check_prefix_group_id(df_train, df_val, df_ds_fp): 
    train_prefix_group_id_unique = df_train.Prefix_group_id.unique()
    val_prefix_group_id_unique = df_val.Prefix_group_id.unique()
    fp_prefix_group_id_unique = df_ds_fp.Prefix_group_id.unique()
    
    set_train_prefix_group_id_unique = set(train_prefix_group_id_unique)
    set_val_prefix_group_id_unique = set(val_prefix_group_id_unique)
    set_fp_prefix_group_id_unique = set(fp_prefix_group_id_unique)
    
    return set_train_prefix_group_id_unique, set_val_prefix_group_id_unique, set_fp_prefix_group_id_unique
    
def check_sets(a_set, b_set):
    if len(a_set.intersection(b_set)) > 0:
        has_common_elements = True
        print(a_set.intersection(b_set))  
        has_common_elements
    else:
        has_common_elements = False
        print("no common elements")
    return has_common_elements
    
def check_if_common_prefix_group_id(df_train, df_val, df_ds_fp): 
    set_train_prefix_group_id_unique, set_val_prefix_group_id_unique, set_fp_prefix_group_id_unique = check_prefix_group_id(df_train, df_val, df_ds_fp)
    result_train_val = check_sets(set_train_prefix_group_id_unique, set_val_prefix_group_id_unique)
    result_train_fp = check_sets(set_train_prefix_group_id_unique, set_fp_prefix_group_id_unique)
    result_val_fp = check_sets(set_val_prefix_group_id_unique, set_fp_prefix_group_id_unique)
    if result_train_val or result_train_fp or result_val_fp:
        print("Need to create a prefix with  f'{Origin_dataset_name}_{prefix_group_id}'")

    
def add_dataset_prefix_group_id(df_train, df_val, df_ds_fp):
    df_train["Origin_dataset_name"]="df_pyronear_ds_03_2024_train"
    df_train['Dataset_prefix_group_id'] = df_train['Origin_dataset_name'] + '_' + df_train['Prefix_group_id']
    
    df_val["Origin_dataset_name"]="df_pyronear_ds_03_2024_val"
    df_val['Dataset_prefix_group_id'] = df_val['Origin_dataset_name'] + '_' + df_val['Prefix_group_id']

    df_ds_fp['Dataset_prefix_group_id'] = df_ds_fp['Origin_dataset_name'] + '_' + df_ds_fp['Prefix_group_id']


def create_new_label(df_train, df_val, df_ds_fp):
    # In the dataset "df_pyronear_ds_03_2024", when an image is labeled with a bbox, the label is 0 instead of 1
    # We don't fill NaN when there is missing labels
    df_train['new_label'] = df_train['label'].replace(0.0, 1.0)
    df_val['new_label'] = df_val['label'].replace(0.0, 1.0)
    df_ds_fp['new_label'] = df_ds_fp['label']


def update_columns(df_train, df_val, df_ds_fp):
    df_train['pred'] = np.nan
    df_val['pred'] = np.nan

    
    df_train['label_yolobbox_pred'] = df_train.apply(lambda row: [row['new_label'], row['yolo_bbox_xcenter'], row['yolo_bbox_ycenter'], row['yolo_bbox_width'], row['yolo_bbox_height'], row['pred']], axis=1)
    df_val['label_yolobbox_pred'] = df_val.apply(lambda row: [row['new_label'], row['yolo_bbox_xcenter'], row['yolo_bbox_ycenter'], row['yolo_bbox_width'], row['yolo_bbox_height'], row['pred']], axis=1)
    df_ds_fp['label_yolobbox_pred'] = df_ds_fp.apply(lambda row: [row['label'], row['yolo_bbox_xcenter'], row['yolo_bbox_ycenter'], row['yolo_bbox_width'], row['yolo_bbox_height'], row['pred']], axis=1)
    
    
    df_train['label_absyolobbox_pred'] = df_train.apply(lambda row: [row['new_label'], row['bbox_xmin_abs'], row['bbox_ymin_abs'], row['bbox_xmax_abs'], row['bbox_ymax_abs'], row['pred']], axis=1)
    df_val['label_absyolobbox_pred'] = df_val.apply(lambda row: [row['new_label'], row['bbox_xmin_abs'], row['bbox_ymin_abs'], row['bbox_xmax_abs'], row['bbox_ymax_abs'], row['pred']], axis=1)
    df_ds_fp['label_absyolobbox_pred'] = df_ds_fp.apply(lambda row: [row['label'], row['bbox_xmin_abs'], row['bbox_ymin_abs'], row['bbox_xmax_abs'], row['bbox_ymax_abs'], row['pred']], axis=1)
    
    
    df_train['nb_detections'] = df_train['label'].replace(0.0, 1.0).fillna(0)
    df_val['nb_detections'] = df_val['label'].replace(0.0, 1.0).fillna(0)
    


def order_columns(df_train, df_val, df_ds_fp): 
    desired_order = ['Extracted_Datetime', 'Image_Path', 'Label_Path', 'Rel_Image_Path',
       'Rel_Label_Path', 'Origin_dataset_name', 'Image_basename',
       'Label_basename', 'has_label', 'Group', 'Prefix', 'Prefix_group_id',
       'Dataset_prefix_group_id', 'Datetime_Str', 'Extension', 'Date', 'Time', 'Year', 'Month', 'Day',
       'Hour', 'Minute', 'Second', 'img_height', 'img_width', 'label', 'new_label', 
       'yolo_bbox_xcenter', 'yolo_bbox_ycenter', 'yolo_bbox_width',
       'yolo_bbox_height', 'bbox_xmin_abs', 'bbox_ymin_abs', 'bbox_xmax_abs',
       'bbox_ymax_abs', 'pred', 'label_yolobbox_pred',
       'label_absyolobbox_pred', 'nb_detections']


    df_train = df_train[desired_order]
    df_val = df_val[desired_order]
    df_ds_fp = df_ds_fp[desired_order]


def plot_yolo_bbox(merged_df, image_path):
    image = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rows_with_image_path = merged_df[merged_df['Image_Path'] == image_path]
    index = rows_with_image_path.index[0]
    bbox = merged_df.iloc[index]["label_absyolobbox_pred"]
    
    print("plot_yolo_bbox")
    print(bbox)
    try:
        print(len(bbox))
    except Exception:
        print("no length")
    print("-----")

    if len(bbox)==6:
        label, x_min_real, y_min_real, x_max_real, y_max_real, pred = bbox
        print(bbox)
    else:
        label = np.nan
        x_min_real = 0
        y_min_real = 0
        x_max_real = 0
        y_max_real = 0
        pred = np.nan
    
    color = (0,255,0)
    thickness = 2
    img_bb = cv2.rectangle(rgb_img, (x_min_real, y_min_real), (x_max_real, y_max_real), color, thickness)
    plt.imshow(img_bb)
    plt.show()

def visual_check(merged_df, n=5):
    # n random samples to check
    random_rows = merged_df.sample(n)
    image_paths = random_rows['Image_Path']
    label_paths = random_rows['Label_Path']

    for img_path, label_path in zip(image_paths, label_paths):
        print(img_path)
        print(label_path)
        print(os.path.exists(label_path))
        plot_yolo_bbox(merged_df, img_path)


def main(csv_file):
    data_dir = "/Users/marguerite/workspace_DS/"
    csv_file_ds_fp = os.path.join(data_dir, "df_DS_fp_newlines_multiple_bbox.csv")
    csv_file_train = os.path.join(data_dir, "df_pyronear_ds_03_2024_train_w_datetime_groups.csv")
    csv_file_val = os.path.join(data_dir, "df_pyronear_ds_03_2024_val_w_datetime_groups.csv")
    
    df_train = pd.read_csv(csv_file_train)
    df_val = pd.read_csv(csv_file_val)
    df_ds_fp = pd.read_csv(csv_file_ds_fp)
    
    print("check_if_common_prefix_group_id")
    check_if_common_prefix_group_id(df_train, df_val, df_ds_fp)
    print("add_dataset_prefix_group_id")
    add_dataset_prefix_group_id(df_train, df_val, df_ds_fp)
    print("create_new_label")
    create_new_label(df_train, df_val, df_ds_fp)
    print("update_columns")
    update_columns(df_train, df_val, df_ds_fp)
    print("order_columns")
    order_columns(df_train, df_val, df_ds_fp)
    merged_df = pd.concat([df_train, df_val, df_ds_fp], ignore_index=True)
    merged_df.to_csv(csv_file, index=False)
    print(f"Merged dataframe saved to : {csv_file}")


if __name__=="__main__":
    csv_file = "/Users/marguerite/workspace_DS/pyronear_ds_03_2024_train_val_DS_fp_temporal_dataset.csv"
    main(csv_file)
