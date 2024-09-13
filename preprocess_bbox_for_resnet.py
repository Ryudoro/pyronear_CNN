import os 
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img


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
    left = int(max(center_x - crop_size // 2, 0))
    top = int(max(center_y - crop_size // 2, 0))
    right = int(min(center_x + crop_size // 2, image_width))
    bottom = int(min(center_y + crop_size // 2, image_height))
    
    # Crop the original image 
    cropped_image = image[top:bottom, left:right]

    # If the crop size is larger than the target size for ResNet, resize the image
    if crop_size > target_bbox_resnet:
        cropped_image = cv2.resize(cropped_image, (target_bbox_resnet, target_bbox_resnet), interpolation=cv2.INTER_AREA)
    
    image_array = img_to_array(cropped_image)
    preprocessed_image = preprocess_input(image_array)
        
    return preprocessed_image
        
    