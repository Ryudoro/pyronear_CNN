import pandas as pd
import random
from utils_dataset import prepare_sequences2
import cv2
import os 
import numpy as np
from matplotlib import pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
import re
import tensorflow as tf 


def get_test_set(all_data_csv_path: str, train_val_csv_path: str, test_size: int, sequence_length: int) -> pd.DataFrame:
    """Get test set to check model performances.

    Args:
        all_data_csv_path (str): Path to csv path with all raw data after padding/truncating/interpolation.
        train_val_csv_path (str): Path to csv with data for train/validation.
        test_size (int): Size for test set.
        sequence_length (int): Number of images in a sequence.

    Returns:
        result_df_real_test (pd.DataFrame): Pandas DataFrame with real test set.
    """

    # Dataframe with all data
    df = pd.read_csv(all_data_csv_path)

    # Datafraome with train / validation set
    result_df_train_val = pd.read_csv(train_val_csv_path)
 
    df_sorted = df.sort_values(by='Dataset_prefix_group_id')
    df_sorted['group'] = df_sorted.groupby('Dataset_prefix_group_id')['new_label'].transform(lambda x: 1 if 1 in x.values else 0)

    group1 = df_sorted[df_sorted['group'] == 1]
    group2 = df_sorted[df_sorted['group'] == 0]

    set_group1 = group1['Dataset_prefix_group_id'].unique().tolist()
    set_group2 = group2['Dataset_prefix_group_id'].unique().tolist()

    train_val_group1 = result_df_train_val[result_df_train_val['group']== 1]
    train_val_group2 = result_df_train_val[result_df_train_val['group']== 0]

    all_possible_test_group1 = list(set(set_group1) - set(train_val_group1))
    all_possible_test_group2 = list(set(set_group2) - set(train_val_group2))
    print(len(all_possible_test_group1), len(all_possible_test_group2))

    # TEST set
    sample_size_test = test_size // 2
    num_groups_needed_test = sample_size_test // sequence_length

    all_possible_test_group1 = list(set(set_group1) - set(train_val_group1))
    all_possible_test_group2 = list(set(set_group2) - set(train_val_group2))

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
    return result_df_real_test

def create_model_info_dict(data_dir: str, model_dir_list: list, train_val_csv_paths_list: list, result_df_real_test_csv_paths_list: list, df_gt_pred_csv_paths_list: list, df_metrics_paths_list: list) -> dict:
    """Create dictionary with models info.

    Args:
        data_dir (str): Path to data directory.
        model_dir_list (list): List of directories of models.
        train_val_csv_paths_list (list): List of csv paths to train/validation sets.
        result_df_real_test_csv_paths_list (list): List of csv paths to real test sets.
        df_gt_pred_csv_paths_list (list): List of csv paths to save gt and predictions.
        df_metrics_paths_list (list): List of csv paths with metrics.

    Returns:
        model_dict (dict): Dictionary with csv paths for different models 
    """

    model_dict = {}
    for i, model_dir in enumerate(model_dir_list): 
        train_val_csv_path = os.path.join(data_dir, train_val_csv_paths_list[i])
        result_df_real_test_csv_path = os.path.join(data_dir, result_df_real_test_csv_paths_list[i])
        df_gt_pred_csv_path = os.path.join(data_dir, df_gt_pred_csv_paths_list[i])
        df_metrics_path = os.path.join(data_dir, df_metrics_paths_list[i])

        model_dict[os.path.basename(model_dir)]={"model_dir": model_dir, 
                    "train_val_csv_path": train_val_csv_path, 
                    "result_df_real_test_csv_path": result_df_real_test_csv_path, 
                    "df_gt_pred_csv_path": df_gt_pred_csv_path, 
                    "df_metrics_path": df_metrics_path}     
    
    return model_dict
        

def select_model_and_paths(model_name: str, model_dict: dict) -> tuple:
    """Select model and paths from model_dict.

    Args:
        model_name (str): Name of the model.
        model_dict (dict): Dictionnary with csv paths corresponding to a given model.

    Returns:
        tuple: Information of csv paths for a given model_name
        - model_dir (str): Directory corresponding to the model_name.
        - train_val_csv_path (str): CSV path for train/val set for the model_name.
        - result_df_real_test_csv_path (str): CSV path for test set for the model_name.
        - df_gt_pred_csv_path (str): CSV path for the gt, prediction of the model_name
        - df_metrics_path (str): CSV path of the metrics from the model_name.
    """
    model_dir = model_dict[model_name]["model_dir"]
    train_val_csv_path = model_dict[model_name]["train_val_csv_path"]
    result_df_real_test_csv_path = model_dict[model_name]["result_df_real_test_csv_path"]
    df_gt_pred_csv_path = model_dict[model_name]["df_gt_pred_csv_path"]
    df_metrics_path = model_dict[model_name]["df_metrics_path"]

    return model_dir, train_val_csv_path, result_df_real_test_csv_path, df_gt_pred_csv_path, df_metrics_path



def make_inference(model_dir: str, X_test: np.ndarray, batch_size=32) -> np.ndarray:
    """Make inference for model. 

    Args:
        model_dir (str): Directory of the model with subfolders assets/variables and .pb files
        X_test (np.ndarray): Numpy array of sequences of test images, each sequence has sequence_length images.
        batch_size (int, optional): Size of the batch. Defaults to 32.

    Returns:
        predictions (np.ndarray): Predictions of the model.
    """

    model = tf.saved_model.load(model_dir)

    # Print model's signature to identify input and output tensor names
    for key, value in model.signatures.items():
        print(f"Signature Key: {key}")
        print("Inputs:")
        for input_key, input_value in value.structured_input_signature[1].items():
            print(f"\t{input_key}: {input_value}")

        print("Outputs:")
        for output_key, output_value in value.structured_outputs.items():
            print(f"\t{output_key}: {output_value}")

    # Assuming the model has 'serving_default' signature
    infer = model.signatures['serving_default']

    # Extract the correct input tensor name from the signature inspection
    input_tensor_name = input_value.name
    
    # Extract the correct output tensor name from the signature inspection
    output_tensor_name = output_value.name
    
    # Placeholder for predictions
    predictions = []

    # Process data in batches
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i + batch_size]
        input_tensor = tf.convert_to_tensor(batch)
        print(input_tensor.shape)
        input_dict = {input_tensor_name: input_tensor}
        batch_predictions = infer(**input_dict)[output_tensor_name].numpy()
        predictions.append(batch_predictions)

    # Concatenate all batch predictions
    predictions = np.concatenate(predictions, axis=0)

    return predictions


def get_class(gt_list: list, pred_list: list) -> list:
    """From lists of ground truths and predictions, give a TP/TN/FP/FN class.

    Args:
        gt_list (list): List of ground truths.
        pred_list (list): List of predictions.

    Returns:
        class_list (list): List containing TP/TN/FP/FN.
    """

    class_list = []

    for gt, pred in zip(gt_list, pred_list):
        if (pred == 1) & (gt == 1):
            class_val = "tp"
        if (pred == 0) & (gt == 0):
            class_val = "tn"
        if (pred == 1) & (gt == 0): 
            class_val = "fp"
        if (pred == 0) & (gt == 1):
            class_val = "fn"
        class_list.append(class_val)
    return class_list


def get_tp_fp_tn_fn(df_gt_pred: pd.DataFrame, sequence_length = 5) -> tuple:
    """Get TP/FP/TN/FN indexes

    Args:
        df_gt_pred (pd.DataFrame): Pandas datagrame with ground truths and predictions, and class_label (TP/FP/FN/TN).
        sequence_length (int, optional): Length of a sequence. Defaults to 5.

    Returns:
        tuple: 4 lists of indexes for TP, FP, FN, TN
            - tp_ind
            - fp_ind
            - fn_ind 
            - tn_ind
    """
    tp_index_list = df_gt_pred[df_gt_pred.class_label=="tp"].index.tolist()
    tp_ind = [index*sequence_length for index in tp_index_list]
   
    fp_index_list = df_gt_pred[df_gt_pred.class_label=="fp"].index.to_list()
    fp_ind = [index*sequence_length for index in fp_index_list]

    fn_index_list = df_gt_pred[df_gt_pred.class_label=="fn"].index.tolist()
    fn_ind = [index*sequence_length for index in fn_index_list]

    tn_index_list = df_gt_pred[df_gt_pred.class_label=="tn"].index.tolist()
    tn_ind = [index*sequence_length for index in tn_index_list]

    return tp_ind, fp_ind, fn_ind, tn_ind


def save_class_label(y_test: list, y_pred: list, csv_path: str, threshold=0.5) -> pd.DataFrame:
    """Save class label (TP/TN/FP/FN) into a csv.

    Args:
        y_test (list): List of ground truth (0: no fire / 1: fire) for each sequence.
        y_pred (list): List of predictions.
        csv_path (str): CSV path to save results.
        threshold (float, optional): Threshold to make y_pred binary. Defaults to 0.5.

    Returns:
        df_gt_pred (pd.DataFrame): Dataframe with gt, pred, pred_binary and class_label columns.
    """

    df_gt_pred = pd.DataFrame()
    df_gt_pred["gt"] = y_test
    df_gt_pred["pred"] = y_pred
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_pred_binary = y_pred_binary.reshape(-1)
    
    df_gt_pred["pred_binary"] = y_pred_binary
    class_list = get_class(y_test, y_pred_binary)
    df_gt_pred["class_label"] = class_list
    df_gt_pred.to_csv(csv_path, index=False)

    return df_gt_pred

def get_index_for_series(tp_ind: list, sequence_length: int) -> list:
    """Extend the indexes in tp_ind according to sequence_length, as the gt/pred are 1 output for a whole sequence (not sequence_length outputs).

    Args:
        tp_ind (list): List of indexes for a given class (named with TP but can be any class).
        sequence_length (int): Length of a sequence.

    Returns:
        tp_ind_series (list): List
    """
    tp_ind_series = []

    for val in tp_ind:
        for i in range(sequence_length):
            tp_ind_series.append(val+i)

    return tp_ind_series


def update_df_with_gt_pred_class_label(df_gt_pred: pd.DataFrame, result_df_test: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
    """Updata dataframe with class label (TP/TN/FP/FN).

    Args:
        df_gt_pred (pd.DataFrame): Pandas Dataframe with ground truths and predictions.
        result_df_test (pd.DataFrame): Pandas Dataframe to update adding TP/TN/FP/FN in column gt_pred_class_label.  
        sequence_length (int): Length of a sequence.

    Returns:
        pd.DataFrame: Pandas Dataframe with TP/FP/TN/FN on each rows.
    """
    # Get indexes for TP/FP/TN/TN
    tp_ind, fp_ind, fn_ind, tn_ind = get_tp_fp_tn_fn(df_gt_pred, sequence_length)

    # Compute the extended indexes for each sequence as a prediction is made for a whole sequence.
    tp_ind_series = get_index_for_series(tp_ind, sequence_length)
    fp_ind_series = get_index_for_series(fp_ind, sequence_length)
    fn_ind_series = get_index_for_series(fn_ind, sequence_length)
    tn_ind_series = get_index_for_series(tn_ind, sequence_length)

    result_df_test['gt_pred_class_label'] = None
    result_df_test.loc[tp_ind_series, 'gt_pred_class_label'] = 'tp'
    result_df_test.loc[fp_ind_series, 'gt_pred_class_label'] = 'fp'
    result_df_test.loc[fn_ind_series, 'gt_pred_class_label'] = 'fn'
    result_df_test.loc[tn_ind_series, 'gt_pred_class_label'] = 'tn'

    return result_df_test

def add_tp_fp_tn_fn_class(y_test: list, y_pred: list, csv_path: str, result_df_test: pd.DataFrame, sequence_length=5, threshold=0.5) -> pd.DataFrame:
    """Add TP/FP/TN/FN class.

    Args:
        y_test (list): List of ground truth (0: no fire / 1: fire) for each sequence.
        y_pred (list): List of predictions.
        csv_path (str): CSV path to save results.
        result_df_test (pd.DataFrame): _description_
        sequence_length (int, optional): Length of a sequence. Defaults to 5.
        threshold (float, optional): Threshold to make y_pred binary. Defaults to 0.5.

    Returns:
        result_df_test (pd.DataFrame): Dataframe with gt, pred, class label.
    """
    df_gt_pred = save_class_label(y_test, y_pred, csv_path, threshold)
    result_df_test = result_df_test.reset_index()
    result_df_test = update_df_with_gt_pred_class_label(df_gt_pred, result_df_test, sequence_length)

    return result_df_test


def add_gt_pred_pred_binary(result_df_real_test: pd.DataFrame, y_test: list, y_pred: list, sequence_length=5, threshold=0.5) -> pd.DataFrame: 
    """_summary_

    Args:
        result_df_real_test (pd.DataFrame): Dataframe with real test set data.
        y_test (list): List of ground truth (0: no fire / 1: fire) for each sequence.
        y_pred (list): List of predictions.
        sequence_length (int, optional): Length of a sequence. Defaults to 5.
        threshold (float, optional): Threshold to make y_pred binary. Defaults to 0.5.

    Returns:
        result_df_real_test (pd.DataFrame): Dataframe with columns gt, y_pred, y_pred_binary.
    """

    y_pred_binary = (y_pred >= threshold).astype(int)
    if len(y_pred.shape)==2:
        y_pred = y_pred.reshape(-1)
    y_pred_binary = y_pred_binary.reshape(-1)

    y_test_replicated_for_seq = [y_test_value for y_test_value in y_test for i in range(sequence_length)]
    y_pred_replicated_for_seq = [y_pred_value for y_pred_value in y_pred for i in range(sequence_length)]
    y_pred_binary_replicated_for_seq = [y_pred_bin_value for y_pred_bin_value in y_pred_binary for i in range(sequence_length)]

    result_df_real_test["gt"] = y_test_replicated_for_seq
    result_df_real_test["y_pred"] = y_pred_replicated_for_seq
    result_df_real_test["y_pred_binary"] = y_pred_binary_replicated_for_seq
    return result_df_real_test


def prepare_visual_sequences(df: pd.DataFrame, data_dir: str, gt_pred_class_label: str, sequence_length=5) -> tuple:
    """Prepare sequences for visualization depending on the gt_pred_class_label (TP/TN/FP/FN).

    Args:
        df (pd.DataFrame): Dataframe with gt_pred_class_label.
        data_dir (str): Directory of the data.
        gt_pred_class_label (str): Class label 'tp', 'tn', 'fn', 'fp'.
        sequence_length (int, optional): Length of a sequence. Defaults to 5.

    Returns:
        tuple: Filtered sequences corresponding to the given gt_pred_class_label.
            - np.array(X): Array of sequences of images. Shape :(number_of_filtered_sequences, sequence_length, image_height, image_width, channels)
            - np.array(y): Array of sequences labels (0: no fire, 1: fire). Shape :(number_of_filtered_sequences, sequence_length).
            - np.array(X_groups): Array with names of the sequences.  Shape :(number_of_filtered_sequences, sequence_length).
            - np.array(X_names): Array with names of the basenames of images.  Shape :(number_of_filtered_sequences, sequence_length).
    """
    X = []
    y = []
    X_groups = []
    X_names = []
    grouped = df.sort_values(by=['Dataset_prefix_group_id', 'Extracted_Datetime']).groupby('Dataset_prefix_group_id')
    
    tp_examples = grouped.filter(lambda x: (x['gt_pred_class_label'] == 'tp').any())
    tn_examples = grouped.filter(lambda x: (x['gt_pred_class_label'] == 'tn').any())
    fn_examples = grouped.filter(lambda x: (x['gt_pred_class_label'] == 'fn').any())
    fp_examples = grouped.filter(lambda x: (x['gt_pred_class_label'] == 'fp').any())
    
    if gt_pred_class_label == "tp": 
        grouped_filter = tp_examples
    if gt_pred_class_label == "tn": 
        grouped_filter = tn_examples
    if gt_pred_class_label == "fn": 
        grouped_filter = fn_examples
    if gt_pred_class_label == "fp": 
        grouped_filter = fp_examples

    grouped_filter = grouped_filter.sort_values(by=['Dataset_prefix_group_id', 'Extracted_Datetime']).groupby('Dataset_prefix_group_id')
    
    for i, (group_name, group) in enumerate(grouped_filter):
        images = []
        y_temp = []
        groups = []
        img_names = []
        for _, row in group.iterrows():
            
            image_path = os.path.join(data_dir, row['Rel_Image_Path'])
            image = cv2.imread(image_path)
            images.append(image)
            
            y_temp.append(row['new_label'])
            groups.append(group_name)
            img_names.append(os.path.basename(image_path))

            if len(images) == sequence_length:
                X.append(np.array(images))
                y.append(np.array(y_temp))
                X_groups.append(np.array(groups))
                X_names.append(np.array(img_names))
    
    return np.array(X), np.array(y), np.array(X_groups), np.array(X_names)


def imshow_series(seq_images: np.ndarray, seq_filenames: np.ndarray, sequence_length=5) -> plt.figure:
    """Plot series.
    Args:
        seq_images (np.ndarray): Array of sequences of images.
        seq_filenames (ndarray): Array of sequences of corresponding filenames.
        sequence_length (int, optional): Length of a sequence. Defaults to 5.

    Returns:
        plt.figure: Figure with subplots for each image in the serie.
    """
    pattern = r"(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})"

    group_name = re.split(pattern, seq_filenames[0])[0][:-1]
    

    fig, axes = plt.subplots(1, sequence_length, figsize=(sequence_length * 5, 5))
    plt.suptitle(group_name)

    for i in range(sequence_length):
        filename = re.split(pattern, seq_filenames[i])[1]
        axes[i].set_title(filename, wrap=True)
        axes[i].imshow(seq_images[i][:,:,::-1])
        axes[i].axis('off') 
    
    return fig


def imshow_all_series(all_seq_images: np.ndarray, all_seq_filenames: np.ndarray, sequence_length=5, pdf_path="output_test_pdf.pdf"):
    """Save all series in a pdf.

    Args:
        all_seq_images (np.ndarray): Array with sequences of images.
        all_seq_filenames (np.ndarray): Array of sequences of corresponding filenames.
        sequence_length (int, optional): Length of a sequence.. Defaults to 5.
        pdf_path (str, optional): Path to save the figures into a pdf. Defaults to "output_test_pdf.pdf".
    """
    with PdfPages(pdf_path) as pdf:
        for i in range(10):
            fig = imshow_series(all_seq_images[i], all_seq_filenames[i], sequence_length)
            pdf.savefig(fig)