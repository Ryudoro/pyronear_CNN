import os 
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from multiprocessing import Pool
import time
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
import re
import matplotlib.patches as patches


def resize(img: np.ndarray, long_side_out=128, enlarge=True) -> np.ndarray: 
    """Resize image.

    Args:
        img (np.ndarray): Image to resize.
        long_side_out (int, optional): Longest size of the image after resizing. Defaults to 128.
        enlarge (bool, optional): Enlarge or not. Defaults to True.

    Returns:
        np.ndarray: Resized image.
    """

    long_side = max(img.shape[:2]) 
    resize_factor = long_side_out/long_side 

    if resize_factor < 1: 
        img = cv2.resize( 
            img, 
            (int(resize_factor*img.shape[1]), int(resize_factor*img.shape[0])),  
            interpolation=cv2.INTER_AREA) 

    elif resize_factor > 1: 
        if enlarge: 
            img = cv2.resize( 
                img, 
                (int(resize_factor*img.shape[1]), int(resize_factor*img.shape[0])),  
                interpolation=cv2.INTER_CUBIC) 

        else: 
            img = img 

    return img 


def resize_by_longest_side(img: np.ndarray, output_longest_size=128, is_enlarge=True) -> np.ndarray: 
    """Resize image.

    Args:
        img (np.ndarray): Image to resize.
        long_side_out (int, optional): Longest size of the image after resizing. Defaults to 128.
        enlarge (bool, optional): Enlarge or not. Defaults to True.

    Returns:
        np.ndarray: Resized image.
    """
    input_longest_side = max(img.shape[:2]) 
    resize_factor = output_longest_size/input_longest_side 
    new_dimensions = (int(img.shape[1] * resize_factor), int(img.shape[0] * resize_factor))

    if resize_factor < 1: 
        interpolation = cv2.INTER_AREA
        img = cv2.resize(img, new_dimensions, interpolation=interpolation)
    elif resize_factor > 1 : 
        if is_enlarge: 
            interpolation=cv2.INTER_CUBIC
            img = cv2.resize(img, new_dimensions, interpolation=interpolation)
        else: 
            img = img 

    return img 


def resize_images_to_common_shape(images: list, target_height: int, target_width: int) -> list:
    """Resize list of images to a common shape.

    Args:
        images (list): List of images to resize.
        target_height (int): Target height for resizing.
        target_width (int): Target width for resizing.

    Returns:
        list: List of resized images.
    """
    resized_images = []

    for image in images:
        # Resize the image to the target dimensions
        resized_image = cv2.resize(image, (target_width, target_height))
        resized_images.append(resized_image)

    return np.array(resized_images)





def get_largest_bbox_from_sequence(images_in_sequence: list, bboxes_in_sequence: list, in_yolo_format=True) -> np.ndarray:
    """Get the largest bounding box that encompasses all the bounding boxes in a sequence.

    Args:
        images_in_sequence (list): List of images in a sequence.
        bboxes_in_sequence (list): List of bounding boxes in a sequence.
        in_yolo_format (bool, optional): Return bounding box coordinates in YOLO format or not. Defaults to True.

    Returns:
        largest_bbox_in_sequence (np.ndarray): Largest bounding box of a sequence. 
    """

    bboxes_in_sequence = np.array(bboxes_in_sequence)
    
    images_heights = [image.shape[0] for image in images_in_sequence]
    images_widths = [image.shape[1] for image in images_in_sequence]
    
    # Convert YOLO coordinates to image coordinates
    bbox_seq_x = bboxes_in_sequence[:, 0] * np.array(images_widths)
    bbox_seq_y = bboxes_in_sequence[:, 1] * np.array(images_heights)
    bbox_seq_w = bboxes_in_sequence[:, 2] * np.array(images_widths)
    bbox_seq_h = bboxes_in_sequence[:, 3] * np.array(images_heights)

    # Compute x_min, x_max, y_min, y_max for the sequence
    x_min_seq = (bbox_seq_x - bbox_seq_w / 2)
    y_min_seq = (bbox_seq_y - bbox_seq_h / 2)
    x_max_seq = (bbox_seq_x + bbox_seq_w / 2)
    y_max_seq = (bbox_seq_y + bbox_seq_h / 2)

    # Calculate the largest bounding box (min of mins, max of maxs)
    min_x = int(np.min(x_min_seq))
    min_y = int(np.min(y_min_seq))
    max_x = int(np.max(x_max_seq))
    max_y = int(np.max(y_max_seq))

    largest_bbox_x = min_x
    largest_bbox_y = min_y
    largest_bbox_width = max_x - min_x
    largest_bbox_height = max_y - min_y

    largest_bbox = [largest_bbox_x, largest_bbox_y, largest_bbox_width, largest_bbox_height]
    largest_bbox_in_sequence = largest_bbox * len(images_in_sequence)
    
    if in_yolo_format:
        if (len(set(images_heights)) == 1) & (len(set(images_widths)) == 1):
            # SAME HEIGHTS and WIDTHS
            image_height = images_heights[0]
            image_width = images_widths[0]
            yolo_largest_bbox_x = (largest_bbox_x + largest_bbox_width / 2) / image_width
            yolo_largest_bbox_y = (largest_bbox_y + largest_bbox_height / 2) / image_height
            yolo_largest_bbox_width = largest_bbox_width / image_width
            yolo_largest_bbox_height = largest_bbox_height / image_height

            yolo_largest_bbox = [yolo_largest_bbox_x, yolo_largest_bbox_y, yolo_largest_bbox_width, yolo_largest_bbox_height]
            yolo_largest_bbox_in_sequence = [yolo_largest_bbox] * len(images_in_sequence)
        else: 
            yolo_largest_bbox_in_sequence = []
            for i in range(len(images_in_sequence)): 
                yolo_largest_bbox_x = (largest_bbox_x + largest_bbox_width / 2) / images_widths[i]
                yolo_largest_bbox_y = (largest_bbox_y + largest_bbox_height / 2) / images_heights[i]
                yolo_largest_bbox_width = largest_bbox_width / images_widths[i]
                yolo_largest_bbox_height = largest_bbox_height / images_heights[i]

                yolo_largest_bbox = [yolo_largest_bbox_x, yolo_largest_bbox_y, yolo_largest_bbox_width, yolo_largest_bbox_height]
                yolo_largest_bbox_in_sequence.append(yolo_largest_bbox)
        return np.array(yolo_largest_bbox_in_sequence)
    else:
        return np.array(largest_bbox_in_sequence)


def flatten(list_of_lists: list) -> list:
    """Flatten a list of lists.

    Args:
        list_of_list (list): List of lists to flatten.

    Returns:
        list: Flattened list.
    """
    return [element for sublist in list_of_lists for element in sublist]


def convert_yolo_bbox_to_abs_pixel_coords(yolo_bbox_center_x: int, yolo_bbox_center_y: int, yolo_bbox_width: int, yolo_bbox_height: int, image_width: int, image_height: int) -> tuple:
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
    bbox_x = (yolo_bbox_center_x - yolo_bbox_width/2)*image_width
    bbox_y = (yolo_bbox_center_y - yolo_bbox_height/2)*image_height
    bbox_width = yolo_bbox_width*image_width
    bbox_height =  yolo_bbox_height*image_height

    return bbox_x, bbox_y, bbox_width, bbox_height


def convert_abs_pixel_coords_bbox_to_yolo(bbox_x: int, bbox_y: int, bbox_width: int, bbox_height: int, image_width: int, image_height: int) -> tuple:
    """Convert bbox from absolute pixel coordinates to YOLO format.

    Args:
        bbox_x (int): X-coordinate of the top-left corner of the bounding box, in absolute pixel coordinates.
        bbox_y (int): Y-coordinate of the top-left corner of the bounding box, in absolute pixel coordinates.
        bbox_width (int): Width of the bounding box, in absolute pixel coordinates.
        bbox_height (int): Height ot the bounding box, in absolute pixel coordinates.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.

    Returns:
        tuple: 
        - yolo_bbox_center_x (int): X-center of bbox in YOLO format.
        - yolo_bbox_center_y (int): Y-center of bbox in YOLO format.
        - yolo_bbox_width (int): Width of bbox in YOLO format.
        - yolo_bbox_height (int): Height of bbox in YOLO format.
    """
    yolo_bbox_center_x = (bbox_x + bbox_width / 2) / image_width
    yolo_bbox_center_y = (bbox_y + bbox_height / 2) / image_height
    yolo_bbox_width = bbox_width/image_width
    yolo_bbox_height = bbox_height/image_height

    return yolo_bbox_center_x, yolo_bbox_center_y, yolo_bbox_width, yolo_bbox_height


def convert_yolo_to_other_image_abs_pixel_coords(yolo_bbox_center_x: int, yolo_bbox_center_y: int, yolo_bbox_width: int, yolo_bbox_height: int,
                                       original_image_width: int, original_image_height: int,
                                       new_image_width: int, new_image_height: int) -> tuple:
    """Converts YOLO bbox coordinates from the original image dimension to another image dimension, in absolute pixel coordinates.

    Args:
        yolo_bbox_center_x (int): X-center of bbox in YOLO format.
        yolo_bbox_center_y (int): Y-center of bbox in YOLO format.
        yolo_bbox_width (int): Width of bbox in YOLO format.
        yolo_bbox_height (int): Height of bbox in YOLO format.
        original_image_width (int): Original image width in pixels.
        original_image_height (int): Original image height in pixels.
        new_image_width (int): New image width in pixels, for final scaling.
        new_image_height (int): New image height in pixels, for final scalinig.

    Returns:
        tuple: 
        - new_bbox_x (int): Scaled X-coordinate of the top-left corner of the bounding box, in absolute pixel coordinates.
        - new_bbox_y (int): Scaled Y-coordinate of the top-left corner of the bounding box, in absolute pixel coordinates.
        - new_bbox_width (int): Scaled width of the bounding box, in absolute pixel coordinates.
        - new_bbox_height (int): Scaled height ot the bounding box, in absolute pixel coordinates.
    """
    # Step 1: Convert YOLO bbox to original image coordinates
    bbox_x, bbox_y, bbox_width, bbox_height = convert_yolo_bbox_to_abs_pixel_coords(
        yolo_bbox_center_x, yolo_bbox_center_y, yolo_bbox_width, yolo_bbox_height,
        original_image_width, original_image_height
    )

    # Step 2: Scale the coordinates to the new image dimensions
    new_bbox_x = bbox_x * (new_image_width / original_image_width)
    new_bbox_y = bbox_y * (new_image_height / original_image_height)
    new_bbox_width = bbox_width * (new_image_width / original_image_width)
    new_bbox_height = bbox_height * (new_image_height / original_image_height)

    return new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height


def convert_yolo_with_image_reshape(yolo_bbox_center_x: int, yolo_bbox_center_y: int, yolo_bbox_width: int, yolo_bbox_height: int,
                                       original_image_width: int, original_image_height: int,
                                       new_image_width: int, new_image_height: int) -> tuple: 
    """_summary_

    Args:
        yolo_bbox_center_x (int): X-center of bbox in YOLO format.
        yolo_bbox_center_y (int): Y-center of bbox in YOLO format.
        yolo_bbox_width (int): Width of bbox in YOLO format.
        yolo_bbox_height (int): Height of bbox in YOLO format.
        original_image_width (int): Original image width in pixels.
        original_image_height (int): Original image height in pixels.
        new_image_width (int): New image width in pixels, for final scaling.
        new_image_height (int): New image height in pixels, for final scalinig.

    Returns:
        tuple: 
        - new_yolo_bbox_center_x (int): Scaled X-coordinate of the top-left corner of the bounding box, in YOLO format.
        - new_yolo_bbox_center_y (int): Scaled Y-coordinate of the top-left corner of the bounding box, in YOLO format.
        - new_yolo_bbox_width (int): Scaled width of the bounding box, in YOLO format.
        - new_yolo_bbox_height (int): Scaled height ot the bounding box, in YOLO format.
    """

    # Step 1: Convert YOLO bbox to original image coordinates
    bbox_x, bbox_y, bbox_width, bbox_height = convert_yolo_bbox_to_abs_pixel_coords(
        yolo_bbox_center_x, yolo_bbox_center_y, yolo_bbox_width, yolo_bbox_height,
        original_image_width, original_image_height
    )
    
    # Step 2: Scale the coordinates to the new image dimensions
    new_bbox_x = bbox_x * (new_image_width / original_image_width)
    new_bbox_y = bbox_y * (new_image_height / original_image_height)
    new_bbox_width = bbox_width * (new_image_width / original_image_width)
    new_bbox_height = bbox_height * (new_image_height / original_image_height)

    # Back to YOLO
    new_yolo_bbox_center_x, new_yolo_bbox_center_y, new_yolo_bbox_width, new_yolo_bbox_height = convert_abs_pixel_coords_bbox_to_yolo(new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height, new_image_width, new_image_height)
    
    return new_yolo_bbox_center_x, new_yolo_bbox_center_y, new_yolo_bbox_width, new_yolo_bbox_height


def convert_yolo_bbox_to_abs_pixel_coords(yolo_bbox_center_x: int, yolo_bbox_center_y: int, yolo_bbox_width: int, yolo_bbox_height: int, image_width: int, image_height: int) -> tuple:
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
    bbox_x = (yolo_bbox_center_x - yolo_bbox_width/2)*image_width
    bbox_y = (yolo_bbox_center_y - yolo_bbox_height/2)*image_height
    bbox_width = yolo_bbox_width*image_width
    bbox_height =  yolo_bbox_height*image_height

    return bbox_x, bbox_y, bbox_width, bbox_height


def convert_abs_pixel_coords_bbox_to_yolo(bbox_x: int, bbox_y: int, bbox_width: int, bbox_height: int, image_width: int, image_height: int) -> tuple:
    """Convert bbox from absolute pixel coordinates to YOLO format.

    Args:
        bbox_x (int): X-coordinate of the top-left corner of the bounding box, in absolute pixel coordinates.
        bbox_y (int): Y-coordinate of the top-left corner of the bounding box, in absolute pixel coordinates.
        bbox_width (int): Width of the bounding box, in absolute pixel coordinates.
        bbox_height (int): Height ot the bounding box, in absolute pixel coordinates.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.

    Returns:
        tuple: 
        - yolo_bbox_center_x (int): X-center of bbox in YOLO format.
        - yolo_bbox_center_y (int): Y-center of bbox in YOLO format.
        - yolo_bbox_width (int): Width of bbox in YOLO format.
        - yolo_bbox_height (int): Height of bbox in YOLO format.
    """
    yolo_bbox_center_x = (bbox_x + bbox_width / 2) / image_width
    yolo_bbox_center_y = (bbox_y + bbox_height / 2) / image_height
    yolo_bbox_width = bbox_width/image_width
    yolo_bbox_height = bbox_height/image_height

    return yolo_bbox_center_x, yolo_bbox_center_y, yolo_bbox_width, yolo_bbox_height


def convert_yolo_to_other_image_abs_pixel_coords(yolo_bbox_center_x: int, yolo_bbox_center_y: int, yolo_bbox_width: int, yolo_bbox_height: int,
                                       original_image_width: int, original_image_height: int,
                                       new_image_width: int, new_image_height: int) -> tuple:
    """Converts YOLO bbox coordinates from the original image dimension to another image dimension, in absolute pixel coordinates.

    Args:
        yolo_bbox_center_x (int): X-center of bbox in YOLO format.
        yolo_bbox_center_y (int): Y-center of bbox in YOLO format.
        yolo_bbox_width (int): Width of bbox in YOLO format.
        yolo_bbox_height (int): Height of bbox in YOLO format.
        original_image_width (int): Original image width in pixels.
        original_image_height (int): Original image height in pixels.
        new_image_width (int): New image width in pixels, for final scaling.
        new_image_height (int): New image height in pixels, for final scalinig.

    Returns:
        tuple: 
        - new_bbox_x (int): Scaled X-coordinate of the top-left corner of the bounding box, in absolute pixel coordinates.
        - new_bbox_y (int): Scaled Y-coordinate of the top-left corner of the bounding box, in absolute pixel coordinates.
        - new_bbox_width (int): Scaled width of the bounding box, in absolute pixel coordinates.
        - new_bbox_height (int): Scaled height ot the bounding box, in absolute pixel coordinates.
    """
    # Step 1: Convert YOLO bbox to original image coordinates
    bbox_x, bbox_y, bbox_width, bbox_height = convert_yolo_bbox_to_abs_pixel_coords(
        yolo_bbox_center_x, yolo_bbox_center_y, yolo_bbox_width, yolo_bbox_height,
        original_image_width, original_image_height
    )

    # Step 2: Scale the coordinates to the new image dimensions
    new_bbox_x = bbox_x * (new_image_width / original_image_width)
    new_bbox_y = bbox_y * (new_image_height / original_image_height)
    new_bbox_width = bbox_width * (new_image_width / original_image_width)
    new_bbox_height = bbox_height * (new_image_height / original_image_height)

    return new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height


def convert_yolo_with_image_reshape(yolo_bbox_center_x: int, yolo_bbox_center_y: int, yolo_bbox_width: int, yolo_bbox_height: int,
                                       original_image_width: int, original_image_height: int,
                                       new_image_width: int, new_image_height: int) -> tuple: 
    """_summary_

    Args:
        yolo_bbox_center_x (int): X-center of bbox in YOLO format.
        yolo_bbox_center_y (int): Y-center of bbox in YOLO format.
        yolo_bbox_width (int): Width of bbox in YOLO format.
        yolo_bbox_height (int): Height of bbox in YOLO format.
        original_image_width (int): Original image width in pixels.
        original_image_height (int): Original image height in pixels.
        new_image_width (int): New image width in pixels, for final scaling.
        new_image_height (int): New image height in pixels, for final scalinig.

    Returns:
        tuple: 
        - new_yolo_bbox_center_x (int): Scaled X-coordinate of the top-left corner of the bounding box, in YOLO format.
        - new_yolo_bbox_center_y (int): Scaled Y-coordinate of the top-left corner of the bounding box, in YOLO format.
        - new_yolo_bbox_width (int): Scaled width of the bounding box, in YOLO format.
        - new_yolo_bbox_height (int): Scaled height ot the bounding box, in YOLO format.
    """

    # Step 1: Convert YOLO bbox to original image coordinates
    bbox_x, bbox_y, bbox_width, bbox_height = convert_yolo_bbox_to_abs_pixel_coords(
        yolo_bbox_center_x, yolo_bbox_center_y, yolo_bbox_width, yolo_bbox_height,
        original_image_width, original_image_height
    )
    
    # Step 2: Scale the coordinates to the new image dimensions
    new_bbox_x = bbox_x * (new_image_width / original_image_width)
    new_bbox_y = bbox_y * (new_image_height / original_image_height)
    new_bbox_width = bbox_width * (new_image_width / original_image_width)
    new_bbox_height = bbox_height * (new_image_height / original_image_height)

    # Back to YOLO
    new_yolo_bbox_center_x, new_yolo_bbox_center_y, new_yolo_bbox_width, new_yolo_bbox_height = convert_abs_pixel_coords_bbox_to_yolo(new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height, new_image_width, new_image_height)
    
    return new_yolo_bbox_center_x, new_yolo_bbox_center_y, new_yolo_bbox_width, new_yolo_bbox_height


def is_image_shape_homogeneous_in_sequence(images_in_sequence: list) -> tuple:
    """Check if the shape of all images in a sequence is homogeneous.

    Args:
        images_in_sequence (list): List of images in a sequence.

    Returns:
        tuple: 
        - is_homogeneous (bool): Sequence is homogeneous or not.
        - target_height (int): Target height in pixels.
        - target_width (int): Target width in pixels.

    """
    images_heights = [image.shape[0] for image in images_in_sequence]
    images_widths =  [image.shape[1] for image in images_in_sequence]

    
    if (len(set(images_heights)) > 1) or (len(set(images_widths)) > 1):
        is_homogeneous = False
        target_height = max(images_heights)
        target_width = max(images_widths)
    else: 
        is_homogeneous = True
        target_height = images_heights[0]
        target_width = images_widths[0]

    return is_homogeneous, target_height, target_width


def homogenize_shape_images_in_sequence(images_in_sequence: list, bboxes_in_sequence: list) -> tuple: 
    """Homogenize the shape of images in a sequence, if the shape varies.

    Args:
        images_in_sequence (list): List of images in a sequence.
        bboxes_in_sequence (list): List of bboxes in a sequence.

    Returns:
        tuple: 
        - images_in_sequence (list): List of homogenous images (same shape) in the sequence.
        - bboxes_in_sequence (list): List of homogeneous bboxes in the sequence.
    """
    is_homogeneous, target_height, target_width  = is_image_shape_homogeneous_in_sequence(images_in_sequence)

    if not is_homogeneous:
        # "Resizing the images to ({target_height}, {target_width})
        resized_images = []
        resized_bboxes = []

        for image, bbox in zip(images_in_sequence, bboxes_in_sequence):
                # Resize the image to the target dimensions
                image_height, image_width, _ = image.shape
                resized_image = cv2.resize(image, (target_width, target_height))
                yolo_bb_x = bbox[0]
                yolo_bb_y = bbox[1]
                yolo_bb_w = bbox[2]
                yolo_bb_h = bbox[3]

                new_yolo_bb_x, new_yolo_bb_y, new_yolo_bb_w, new_yolo_bb_h = convert_yolo_with_image_reshape(yolo_bb_x, yolo_bb_y, yolo_bb_w, yolo_bb_h, image_width, image_height, target_width, target_height)
                resized_images.append(resized_image)
                resized_bboxes.append([new_yolo_bb_x, new_yolo_bb_y, new_yolo_bb_w, new_yolo_bb_h])
        return resized_images, resized_bboxes
    else:
        return images_in_sequence, bboxes_in_sequence
    

def add_largest_bbox_to_df(df: pd.DataFrame, data_dir="../", sequence_length=5) -> pd.DataFrame:
    """Add largest bboxes to dataframe.

    Args:
        df (pd.DataFrame): Dataframe with paths to images and bboxes.
        data_dir (str, optional): Directory to data. Defaults to "../".
        sequence_length (int, optional): Length of a sequence (number of images in a sequence). Defaults to 5.

    Returns:
        pd.DataFrame: Updated dataframe with largest bbox for each sequence.
    """
    # Sort and group the dataframe by 'Dataset_prefix_group_id'
    grouped = df.sort_values(by=['Dataset_prefix_group_id', 'Extracted_Datetime']).groupby('Dataset_prefix_group_id')
    
    all_largest_bboxes = []

    for name, group in tqdm(grouped, total=len(grouped), desc="Computing largest bboxes"):
        images_in_sequence = []
        bboxes_in_sequence = []
        images = []
        
        for _, row in group.iterrows():
            image_path = os.path.join(data_dir, row['Rel_Image_Path'])
            image = cv2.imread(image_path)
            images_in_sequence.append(image)
            
            bbox = row[['yolo_bbox_xcenter', 'yolo_bbox_ycenter', 'yolo_bbox_width', 'yolo_bbox_height']].values
            bboxes_in_sequence.append(bbox)

            if len(images) == sequence_length:
                images_in_sequence, bboxes_in_sequence = homogenize_shape_images_in_sequence(images_in_sequence, bboxes_in_sequence)
                
        largest_bbox = get_largest_bbox_from_sequence(images_in_sequence, bboxes_in_sequence, in_yolo_format=True)
        all_largest_bboxes.extend(largest_bbox * len(group))

    largest_bboxes_df = pd.DataFrame(all_largest_bboxes, columns=['yolo_largest_bbox_xcenter', 'yolo_largest_bbox_ycenter', 'yolo_largest_bbox_width', 'yolo_largest_bbox_height'])
    df = pd.concat([df.reset_index(drop=True), largest_bboxes_df], axis=1)

    return df
