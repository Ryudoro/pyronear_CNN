import os 
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import random
from multiprocessing import Pool
from tqdm import tqdm
import time

def load_and_process_image(image_path, bbox, target_size=(224, 224)):
    """
    Charge et traite une image : recadre selon la bbox et redimensionne.
    """

    image = cv2.imread(image_path)

    x_center, y_center, width, height = bbox
    x_center *= image.shape[1]
    y_center *= image.shape[0]
    width *= image.shape[1]
    height *= image.shape[0]
    x = int(x_center - width / 2)
    y = int(y_center - height / 2)
    
    cropped_image = image[y:y+int(height), x:x+int(width)]
    cropped_height, cropped_width = cropped_image.shape[:2]

    resized_image = cv2.resize(cropped_image, target_size)
    
    image_array = img_to_array(resized_image)
    preprocessed_image = preprocess_input(image_array)
    
    return preprocessed_image

def load_and_process_image2(image_path, bbox, target_size=(224, 224), ratio=0.1):
         """
         Charge et traite une image : recadre selon la bbox et redimensionne.
         """
         image = cv2.imread(image_path)
 
         x_center, y_center, width, height = bbox
         x_center *= image.shape[1]
         y_center *= image.shape[0]
         width *= image.shape[1]
         height *= image.shape[0]
 
         x = int(x_center - width / 2 - ratio*width/2)
         y = int(y_center - height / 2 - ratio*height/2)
         width = int(width *(1+ratio))
         height = int(height *(1+ratio))
 
         if x < 0:
                 width -= int(x)
                 x = 0
         if x+width > image.shape[1]: 
                 x -= int(abs(image.shape[1] - (x+width)))
                 if x<0:
                        x=0
         if y < 0:
                 height -= int(y)
                 y = 0
         if y+height > image.shape[0]: 
                 y -= int(abs(image.shape[0] - (y+height)))
                 if y<0:
                         y=0
 
         cropped_image = image[y:y+int(height), x:x+int(width)]
         cropped_height, cropped_width = cropped_image.shape[:2]
         
         resized_image = cv2.resize(cropped_image, target_size)
 
         image_array = img_to_array(resized_image)
         preprocessed_image = preprocess_input(image_array)
 
         return preprocessed_image 

def prepare_sequences(df, data_dir ="dataset_pyronear_yolo_lstm", sequence_length=5):
    """
    Prépare les séquences d'images pour l'entrée du modèle.
    """
    X = []
    y = []
    
    grouped = df.sort_values(by=['Dataset_prefix_group_id']).groupby('Dataset_prefix_group_id')
    
    for name, group in grouped:
        
        images = []
        y_temp = []
        for _, row in group.iterrows():
            image_path = os.path.join(data_dir, row['Rel_Image_Path'])
            
            bbox = row[['yolo_bbox_xcenter', 'yolo_bbox_ycenter', 'yolo_bbox_width', 'yolo_bbox_height']]
            processed_image = load_and_process_image2(image_path, bbox)
            images.append(processed_image)
        
            y_temp.append(row['new_label'])
            
            if len(images) == sequence_length:
                X.append(np.array(images))
                y.append(np.array(y_temp))
                images = []
                y_temp = []
            
    y = transform_list(y)
    return np.array(X), np.array(y)



def process_group(group, data_dir, sequence_length):
    X = []
    y = []

    images = []
    y_temp = []

    for _, row in group.iterrows():
        image_path = os.path.join(data_dir, row['Rel_Image_Path'])

        bbox = row[['yolo_bbox_xcenter', 'yolo_bbox_ycenter', 'yolo_bbox_width', 'yolo_bbox_height']]
        processed_image = load_and_process_image_for_resnet(image_path, bbox)
        images.append(processed_image)

        y_temp.append(row['new_label'])

        if len(images) == sequence_length:
            X.append(np.array(images))
            y.append(np.array(y_temp))
            images = []
            y_temp = []

    return X, y

def prepare_sequences2(df, data_dir="dataset_pyronear_yolo_lstm", sequence_length=5):
    """
    Prépare les séquences d'images pour l'entrée du modèle.
    """
    grouped = df.sort_values(by=['Dataset_prefix_group_id']).groupby('Dataset_prefix_group_id')
    
    pool = Pool()
    
    results = []
    total_groups = len(grouped)
    start_time = time.time()
    
    for name, group in tqdm(grouped, total=total_groups, desc="Processing groups"):
        result = pool.apply_async(process_group, args=(group, data_dir, sequence_length))
        results.append(result)

    X = []
    y = []

    for result in tqdm(results, total=len(results), desc="Collecting results"):
        X_group, y_group = result.get()
        X.extend(X_group)
        y.extend(y_group)
    
    pool.close()
    pool.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    
    y = transform_list(y)
    return np.array(X), np.array(y)

def transform_list(lists):
    result = [1 if any(sublist) else 0 for sublist in lists]
    return result

def calculate_proportions(lst):
    count_0 = lst.count(0)
    count_1 = lst.count(1)
    total = len(lst)
    proportion_0 = count_0 / total
    proportion_1 = count_1 / total

    return proportion_0, proportion_1


def create_tensorflow_dataset(X_train, y_train, X_test, y_test, batch_size, shuffle_buffer_size=100 ):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    test_dataset = test_dataset.shuffle(shuffle_buffer_size).batch(batch_size)

    return train_dataset, test_dataset


def create_classical_dataset(df, total_train_val_size=2000, sequence_length=5, save=False, test_size=None):
    df_sorted = df.sort_values(by='Dataset_prefix_group_id')
    df_sorted['group'] = df_sorted.groupby('Dataset_prefix_group_id')['new_label'].transform(lambda x: 1 if 1 in x.values else 0)

    group1 = df_sorted[df_sorted['group'] == 1]
    group2 = df_sorted[df_sorted['group'] == 0]

    set_group1 = group1['Dataset_prefix_group_id'].unique().tolist()
    set_group2 = group2['Dataset_prefix_group_id'].unique().tolist()

    sample_size_train_val = total_train_val_size // 2

    num_groups_needed = sample_size_train_val // sequence_length

    if len(set_group1) < num_groups_needed:
        print(f"Warning: Not enough groups in group1. Using all {len(set_group1)} available groups.")
        random_group1 = set_group1
        remaining_needed = num_groups_needed - len(set_group1)
        random_group2 = random.sample(set_group2, min(len(set_group2), remaining_needed))
    else:
        random_group1 = random.sample(set_group1, num_groups_needed)
        random_group2 = []

    if len(set_group2) < num_groups_needed:
        print(f"Warning: Not enough groups in group2. Using all {len(set_group2)} available groups.")
        random_group2 += set_group2
        remaining_needed = num_groups_needed - len(set_group2)
        random_group1 += random.sample(set_group1, min(len(set_group1), remaining_needed))
    else:
        if not random_group2:
            random_group2 = random.sample(set_group2, num_groups_needed)

    selected_rows_group1 = df_sorted[df_sorted['Dataset_prefix_group_id'].isin(random_group1)]
    selected_rows_group2 = df_sorted[df_sorted['Dataset_prefix_group_id'].isin(random_group2)]

    rows_selected_train_val = pd.concat([selected_rows_group1, selected_rows_group2])
    result_df_train_val = rows_selected_train_val

    if save:
        result_df_train_val.to_csv(f"teacher_model_5epochs_{total_train_val_size}_random_v0_train_val.csv", index=False)

    if test_size:
        sample_size_test = test_size // 2
        num_groups_needed_test = sample_size_test // sequence_length

        all_possible_test_group1 = list(set(set_group1) - set(random_group1))
        all_possible_test_group2 = list(set(set_group2) - set(random_group2))

        if len(all_possible_test_group1) < num_groups_needed_test:
            print(f"Warning: Not enough groups in group1 for test. Using all {len(all_possible_test_group1)} available groups.")
            test_random_group1 = all_possible_test_group1
            remaining_needed = num_groups_needed_test - len(all_possible_test_group1)
            test_random_group2 = random.sample(all_possible_test_group2, min(len(all_possible_test_group2), remaining_needed))
        else:
            test_random_group1 = random.sample(all_possible_test_group1, num_groups_needed_test)
            test_random_group2 = []

        if len(all_possible_test_group2) < num_groups_needed_test:
            print(f"Warning: Not enough groups in group2 for test. Using all {len(all_possible_test_group2)} available groups.")
            test_random_group2 += all_possible_test_group2
            remaining_needed = num_groups_needed_test - len(all_possible_test_group2)
            test_random_group1 += random.sample(all_possible_test_group1, min(len(all_possible_test_group1), remaining_needed))
        else:
            if not test_random_group2:
                test_random_group2 = random.sample(all_possible_test_group2, num_groups_needed_test)

        test_selected_rows_group1 = df_sorted[df_sorted['Dataset_prefix_group_id'].isin(test_random_group1)]
        test_selected_rows_group2 = df_sorted[df_sorted['Dataset_prefix_group_id'].isin(test_random_group2)]

        rows_selected_real_test = pd.concat([test_selected_rows_group1, test_selected_rows_group2])
        result_df_real_test = rows_selected_real_test

        if save:
            result_df_real_test.to_csv(f"teacher_model_5epochs_{test_size}_random_v0_test.csv", index=False)

        return result_df_train_val, result_df_real_test

    return result_df_train_val


def yolo_bbox2abs_pix_coords(yolo_bbox_center_x: int, yolo_bbox_center_y: int, yolo_bbox_width: int, yolo_bbox_height: int, image_width: int, image_height: int) -> tuple:
    """Convert bounding box in YOLO format to absolute pixel coordinates.

    Args:
        yolo_bbox_center_x (int): X-center of bbox in YOLO format.
        yolo_bbox_center_y (int): Y-center of bbox in YOLO format.
        yolo_bbox_width (int): Width of bbox in YOLO format.
        yolo_bbox_height (int): Height of bbox in YOLO format.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.

    Returns:
        tuple: bbox in absolute pixel coordinates.
        - bbox_x (int): X-coordinate of the top-left corner of the bounding box, in absolute pixel coordinates.
        - bbox_y (int): Y-coordinate of the top-left corner of the bounding box, in absolute pixel coordinates.
        - bbox_width (int): Width of the bounding box, in absolute pixel coordinates.
        - bbox_height (int): Height ot the bounding box, in absolute pixel coordinates.
    """
    bbox_x = int((yolo_bbox_center_x - yolo_bbox_width/2)*image_width)
    bbox_y = int((yolo_bbox_center_y - yolo_bbox_height/2)*image_height)
    bbox_width = int(yolo_bbox_width*image_width)
    bbox_height =  int(yolo_bbox_height*image_height)

    return bbox_x, bbox_y, bbox_width, bbox_height


def load_and_process_image_for_resnet(image_path: str, bbox:tuple, target_size=(224, 224), coeff_crop=1.2) -> np.ndarray:
    """Load and process image for ResNet. The output image shape should be (224, 224, 3). 
        1. Find the center of the bounding box.
        2. Determine the size of the crop. If the bounding box size is (w, h), the crop_size is max(w, h, 224). 
        We can take it a bit larger by adding a coefficient like crop_size = crop_size * coeff_crop.
        3. Then, crop an image of size (crop_size, crop_size) around the center of the bounding box.
        4. If crop_size > 224, resize the crop to (224, 224).

    Args:
        image_path (str): Path to image to process.
        bbox (tuple): YOLO bbox.
        target_size (tuple, optional): Target size for ResNet input. Defaults to (224, 224).
        coeff_crop (float, optional): Coefficient to enlarge the crop. Defaults to 1.2.

    Returns:
        np.ndarray: Processed image. 
    """
    assert target_size[0] == target_size[1]
    target_bbox_resnet = target_size[0]
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # YOLO bbox
    yolo_bbox_center_x, yolo_bbox_center_y, yolo_bbox_width, yolo_bbox_height = bbox

    # Convert YOLO bbox to absolute pixels coordinates
    bbox_x, bbox_y, bbox_width, bbox_height = yolo_bbox2abs_pix_coords(yolo_bbox_center_x, yolo_bbox_center_y, yolo_bbox_width, yolo_bbox_height, image_width, image_height)

    # Crop size for final bbox
    crop_size = max(bbox_width, bbox_height, target_bbox_resnet)
    if coeff_crop is not None:
        crop_size *= coeff_crop

    # Center of the bbox in absolute coordinates
    center_x = bbox_x + bbox_width // 2
    center_y = bbox_y + bbox_height // 2

    # Coordinates for cropping around the center
    left = int(center_x - crop_size // 2)
    top = int(center_y - crop_size // 2)
    right = int(center_x + crop_size // 2)
    bottom = int(center_y + crop_size // 2)

    # Deal with border limit-cases 
    # If the left border is negative, put it to 0
    # And translate the bbox to the right
    if left < 0:
        right += abs(left)
        left = 0
    # If the top border is negative, put it to 0
    # And translate the bbox to the bottom
    if top < 0:
        bottom += abs(top)
        top = 0
    # If the right border is > image_width, put it to image_width
    # And translate the bbox to the left
    if right > image_width:
        left -= (right - image_width)
        right = image_width
    # If the bottom border is > image_height, put it to image_height
    # And translate the bbox to the top
    if bottom > image_height:
        top -= (bottom - image_height)
        bottom = image_height

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)

    # Crop the original image 
    cropped_image = image[top:bottom, left:right]

    # If the crop size is larger than the target size for ResNet, resize the image
    if crop_size > target_bbox_resnet:
        cropped_image = cv2.resize(cropped_image, (target_bbox_resnet, target_bbox_resnet), interpolation=cv2.INTER_AREA)
    
    image_array = img_to_array(cropped_image)
    preprocessed_image = preprocess_input(image_array)
        
    return preprocessed_image
        