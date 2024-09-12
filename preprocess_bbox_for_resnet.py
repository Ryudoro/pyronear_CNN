import os 
import cv2
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


def abs_pixel_coords2yolo_bbox(bbox_x: int, bbox_y: int, bbox_width: int, bbox_height: int, image_width: int, image_height: int) -> tuple:
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


def preprocess_bbox_for_resnet(bbox_x: int, bbox_y: int, bbox_width: int, bbox_height: int, image_width: int, image_height: int, target_bbox_width_resnet=224, target_bbox_height_resnet = 224, to_downsize=True) -> tuple:
    """Preprocess bounding box for ResNet. If it is smaller than (224, 224), enlarge the bbox in each direction (top/bottom, left, right).


    Args:
        bbox_x (int): X-coordinate of the top-left corner of the bounding box, in absolute pixel coordinates.
        bbox_y (int):  Y-coordinate of the top-left corner of the bounding box, in absolute pixel coordinates.
        bbox_width (int): Width of the bounding box, in absolute pixel coordinates.
        bbox_height (int): Height ot the bounding box, in absolute pixel coordinates.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
        target_bbox_width_resnet (int, optional): Target width in pixels to feed ResNet. Defaults to 224.
        target_bbox_height_resnet (int, optional): Target height in pixels to feed ResNet. Defaults to 224.
        to_downsize (bool, optional): Argument to deal cases where the bbox dimension is > (224, 224). If True, the bbox is return for further downsampling. If False, the bbox is cropped to (224, 224). Defaults to True.

    Returns:
        tuple: Preprocessed bounding box, in absolute pixel coordinates: new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height.
    """
    if (bbox_width < target_bbox_width_resnet):
        # Diffence between target_bbox_width for resnet and the annotated bbox_width
        diff_pixels_width = target_bbox_width_resnet - bbox_width

        # Compute x-position for left side of the bbox 
        x_left = bbox_x - diff_pixels_width/2

        # Compute x-position for right side of the bbox
        x_right = bbox_x + bbox_width +  diff_pixels_width/2

        if x_left >= 0 and x_right <= image_width:
            new_bbox_x = int(x_left)
            new_bbox_width = int(x_right - x_left)

        if x_left <0 and x_right <= image_width:
            new_bbox_x = 0
            x_right += abs(x_left)  
            new_bbox_width = int(x_right - new_bbox_x)

        if x_right >= image_width:
            new_bbox_x = int(x_left - abs(x_right - image_width))
            x_right = image_width
            new_bbox_width = int(x_right - new_bbox_x)
    
        if bbox_height < target_bbox_height_resnet:
            # Diffence between target_bbox_height for resnet and the annotated bbox_height
            diff_pixels_height = target_bbox_height_resnet - bbox_height

            # Compute y-position for top side of the bbox 
            y_top = bbox_y - diff_pixels_height/2

            # Compute y-position for bottom side of the bbox
            y_bottom = bbox_y + bbox_height +  diff_pixels_height/2

            if y_top >= 0 and y_bottom <= image_height:
                new_bbox_y = int(y_top)
                new_bbox_height = int(y_bottom - y_top)

            if y_top <0 and y_bottom <= image_height:
                new_bbox_y = 0
                y_bottom += abs(y_top)  
                new_bbox_height = int(y_bottom - new_bbox_y)

            if y_bottom >= image_height:
                new_bbox_y = int(y_top - abs(y_bottom - image_height))
                y_bottom = image_height
                new_bbox_height = int(y_bottom - new_bbox_y)
            
        else:
            new_bbox_y = bbox_y
            new_bbox_height = bbox_height
    
    if bbox_height < target_bbox_height_resnet:
        # Diffence between target_bbox_height for resnet and the annotated bbox_height
        diff_pixels_height = target_bbox_height_resnet - bbox_height

        # Compute y-position for top side of the bbox 
        y_top = bbox_y - diff_pixels_height/2

        # Compute y-position for bottom side of the bbox
        y_bottom = bbox_y + bbox_height +  diff_pixels_height/2

        if y_top >= 0 and y_bottom <= image_height:
            new_bbox_y = int(y_top)
            new_bbox_height = int(y_bottom - y_top)

        if y_top <0 and y_bottom <= image_height:
            new_bbox_y = 0
            y_bottom += abs(y_top)  
            new_bbox_height = int(y_bottom - new_bbox_y)

        if y_bottom >= image_height:
            new_bbox_y = int(y_top - abs(y_bottom - image_height))
            y_bottom = image_height
            new_bbox_height = int(y_bottom - new_bbox_y)

        if (bbox_width < target_bbox_width_resnet):
            diff_pixels_width = target_bbox_width_resnet - bbox_width
            x_left = bbox_x - diff_pixels_width/2
            x_right = bbox_x + bbox_width +  diff_pixels_width/2

            if x_left >= 0 and x_right <= image_width:
                new_bbox_x = int(x_left)
                new_bbox_width = int(x_right - x_left)

            if x_left <0 and x_right <= image_width:
                new_bbox_x = 0
                x_right += abs(x_left)  
                new_bbox_width = int(x_right - new_bbox_x)

            if x_right >= image_width:
                new_bbox_x = int(x_left - abs(x_right - image_width))
                x_right = image_width
                new_bbox_width = int(x_right - new_bbox_x)

        else: 
            new_bbox_x = bbox_x
            new_bbox_width = bbox_width
        
    if (bbox_width > target_bbox_width_resnet) and (bbox_height <=target_bbox_height_resnet):
        if not to_downsize:
            new_bbox_width = target_bbox_width_resnet
        else:
            new_bbox_width = bbox_width
        new_bbox_x = bbox_x
        new_bbox_y = bbox_y
        new_bbox_height = bbox_height
    
    if (bbox_width > target_bbox_width_resnet) and (bbox_height > target_bbox_height_resnet):
        if not to_downsize:
            new_bbox_width = target_bbox_width_resnet
            new_bbox_height = target_bbox_height_resnet
        else:
            new_bbox_width = bbox_width
            new_bbox_height = bbox_height
        new_bbox_x = bbox_x
        new_bbox_y = bbox_y

    if (bbox_height > target_bbox_height_resnet) and (bbox_width <= target_bbox_width_resnet):
        if not to_downsize:
            new_bbox_height = target_bbox_height_resnet
        else:
            new_bbox_height = bbox_height
        new_bbox_x = bbox_x
        new_bbox_y = bbox_y
        new_bbox_width = bbox_width
        
    return new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height


def load_and_process_image_for_resnet(image_path, bbox, target_size=(224, 224), to_downsize=True):
    """
    Charge et traite une image : recadre selon la bbox et redimensionne pour resnet
    """
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
 
    yolo_bbox_center_x, yolo_bbox_center_y, yolo_bbox_width, yolo_bbox_height = bbox
    bbox_x, bbox_y, bbox_width, bbox_height = yolo_bbox2abs_pix_coords(yolo_bbox_center_x, yolo_bbox_center_y, yolo_bbox_width, yolo_bbox_height, image_width, image_height)

    new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height = preprocess_bbox_for_resnet(bbox_x, bbox_y, bbox_width, bbox_height, image_width, image_height, target_bbox_width_resnet=224, target_bbox_height_resnet=224, to_downsize=to_downsize)

    cropped_image = image[new_bbox_y:new_bbox_y+int(new_bbox_height), new_bbox_x:new_bbox_x+int(new_bbox_width)]
    cropped_height, cropped_width = cropped_image.shape[:2]
    
    resized_image = cv2.resize(cropped_image, target_size)
    image_array = img_to_array(resized_image)
    preprocessed_image = preprocess_input(image_array)
    
    return preprocessed_image
