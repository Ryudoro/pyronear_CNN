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
        processed_image = load_and_process_image(image_path, bbox)
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

