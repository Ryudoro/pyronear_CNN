import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

def get_label_yolo_bbox(label_path: str) -> list:
    """Get YOLO bounding box coordinates from a label path

    Args:
        label_path (str): path to txt file containing the YOLO bbox

    Returns:
        list: YOLO bbox coordinates [x_center, y_center, width, height]
    """
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, x_center, y_center, width, height = map(float, parts)
                yolo_bbox = [x_center, y_center, width, height]
                return yolo_bbox
            if len(parts) == 6:
                _, x_center, y_center, width, height, pred = map(float, parts)
                yolo_bbox = [x_center, y_center, width, height]
                return yolo_bbox


def plot_yolo_bbox_from_path(img_path: str, label_path: str):
    """Plot YOLO bbox on image

    Args:
        img_path (str): path to image
        label_path (str): path to YOLO bbox label
    """
    image = cv2.imread(img_path)
    image_height, image_width, _ = image.shape
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    yolo_bbox = get_label_yolo_bbox(label_path)
    x_center, y_center, bbox_width, bbox_height = yolo_bbox
    x_min = x_center - bbox_width / 2
    y_min = y_center - bbox_height / 2
    x_max = x_center + bbox_width / 2
    y_max = y_center + bbox_height / 2
    x_min_real = x_min * image_width
    y_min_real = y_min * image_height
    x_max_real = x_max * image_width
    y_max_real = y_max * image_height

    color = (0,255,0)
    thickness = 2
    img_bb = cv2.rectangle(rgb_img, (int(x_min_real), int(y_min_real)), (int(x_max_real), int(y_max_real)), color, thickness)
    plt.imshow(img_bb)
    plt.show()


def yolo_to_albumentations(bbox:list, image_height:int, image_width:int) -> tuple:
    """Converts YOLO format bounding box to Albumentations format

    Args:
        bbox (list): normalized YOLO bbox coordinates [x_center, y_center, bbox_width, bbox_height]
        image_height (int): image height
        image_width (int): image width

    Returns:
        tuple: bbox coordinates [x1, y1, x2, y2]
    """
    x_center, y_center, bbox_width, bbox_height = bbox

    x1 = (x_center - bbox_width / 2) * image_width
    y1 = (y_center - bbox_height / 2) * image_height
    x2 = (x_center + bbox_width / 2) * image_width
    y2 = (y_center + bbox_height / 2) * image_height

    return x1, y1, x2, y2


def albumentations_to_yolo(bbox:list, image_height:int, image_width:int) -> tuple:
    """Converts Albumentations format bounding boxes to YOLO format

    Args:
        bbox (list): bbox coordinates [x1, y1, x2, y2]
        image_height (int): image height
        image_width (int): image width

    Returns:
        tuple: normalized YOLO bbox coordinates [x_center, y_center, bbox_width, bbox_height]
    """
    x1, y1, x2, y2 = bbox

    bbox_width = (x2 - x1) / image_width
    bbox_height = (y2 - y1) / image_height
    x_center = (x1 + x2) / (2 * image_width)
    y_center = (y1 + y2) / (2 * image_height)

    return x_center, y_center, bbox_width, bbox_height
    

def translate_x_once(image:np.ndarray, yolo_bbox:tuple, translate_limit:int) -> tuple: 
    """Translates horizontally an image 

    Args:
        image (np.ndarray): image to translate
        yolo_bbox (tuple): YOLO bounding box
        translate_limit (int): number of pixels for translation

    Returns:
        tuple:  translated_image (np.ndarray): translated image
                yolo_translated_bbox (tuple): translated YOLO bounding box
    """
    # Translation matrix for x-axis only
    M = np.float32([[1, 0, translate_limit],  
                    [0, 1, 0]])   
    
    # Get image dimension and translate it
    image_height, image_width, _ = image.shape
    translated_image = cv2.warpAffine(image, M, (image_width, image_height))

    # Translate the bounding box in absolute coordinates
    x1, y1, x2, y2 = yolo_to_albumentations(yolo_bbox, image_height, image_width)
    x1_trans = x1
    x2_trans = x2
    x1_trans += translate_limit
    x2_trans += translate_limit
    
    # Adjust bounding box coordinates if they exceed image boundaries
    x1_trans = max(0, min(x1_trans, image_width))
    x2_trans = max(0, min(x2_trans, image_width))

    translated_bbox = [x1_trans, y1, x2_trans, y2]

    # Get YOLO format
    yolo_translated_bbox = albumentations_to_yolo(translated_bbox, image_height, image_width)

    return translated_image, yolo_translated_bbox
    

def plot_yolo_bbox(image: np.ndarray, yolo_bbox: tuple):
    """Plot YOLO bbox on image

    Args:
        image (np.ndarray): image
        yolo_bbox (tuple): YOLO bbox
    """
    h, w, _ = image.shape
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    x_center, y_center, width, height = yolo_bbox
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    x_min_real = x_min * w
    y_min_real = y_min * h
    x_max_real = x_max * w
    y_max_real = y_max * h

    color = (0,255,0)
    thickness = 2
    img_bb = cv2.rectangle(rgb_img, (int(x_min_real), int(y_min_real)), (int(x_max_real), int(y_max_real)), color, thickness)
    plt.imshow(img_bb)
    plt.show()


def translate_x_ntimes(image: np.ndarray, yolo_bbox:tuple, translate_limit: int, n:int) -> tuple:
    """Translate n-1 times

    Args:
        image (np.ndarray): image to translate
        yolo_bbox (tuple): YOLO bounding box to translate
        translate_limit (int): number of pixels for translation
        n (int): number of translations to apply on the image (n-1 translations) 

    Returns:
        tuple:  translated_images (list): list of translated images
                translated_bboxes (list): list of translated YOLO bounding boxes
    """
    translated_images = []
    translated_bboxes = []
    
    for i in range(n):
        
        # Translation matrix for each iteration
        M = np.float32([[1, 0, i * translate_limit],  
                        [0, 1, 0]]) 
        
        # Get image shape and apply translation to it
        image_height, image_width, _ = image.shape
        translated_image = cv2.warpAffine(image, M, (image_width, image_height))

        # Convert YOLO bounding box in absolute coordinates
        x1, y1, x2, y2 = yolo_to_albumentations(yolo_bbox, image_height, image_width)

        # Translate the bounding box
        x1_trans = x1 + i  * translate_limit
        x2_trans = x2 + i  * translate_limit

        # Constrain bounding box coordinates to image boundaries
        x1_trans = max(0, min(x1_trans, image_width))
        x2_trans = max(0, min(x2_trans, image_width))

        translated_bbox = [x1_trans, y1, x2_trans, y2]
        yolo_translated_bbox = albumentations_to_yolo(translated_bbox, image_height, image_width)

        translated_images.append(translated_image)
        translated_bboxes.append(yolo_translated_bbox)

    return translated_images, translated_bboxes


def plot_translated_img_bboxes(img_path: str, label_path: str, translate_limit=50, n=10):
    """Plot translated images and related bounding boxes

    Args:
        img_path (str): path of the image to translate
        label_path (str): path of the YOLO bounding box
        translate_limit (int, optional): number of pixels for translation. Defaults to 50.
        n (int, optional): number of translations to apply on the image (n-1 translations). Defaults to 10.
    """
    fig, axes = plt.subplots(2, 5, figsize=(25, 6))
    axes = axes.ravel()
        
    # Load the original image
    image = cv2.imread(img_path)
    
    # Define the bounding box in YOLO format [x_center, y_center, width, height]
    yolo_bbox = get_label_yolo_bbox(label_path)

    
    # Perform translation
    translated_images, translated_bboxes = translate_x_ntimes(image, yolo_bbox, translate_limit, n)

    for i in range(n):    
        # Plot the translated image
        axes[i].imshow(cv2.cvtColor(translated_images[i], cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
        axes[i].set_title(f"Translated Image {i}")
        
        # Get the translated bounding box
        bbox = translated_bboxes[i]
        
        # Convert bounding box coordinates to pixel values
        h, w, _ = image.shape
        x_center, y_center, width, height = bbox
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        x_min_real = x_min * w
        y_min_real = y_min * h
        x_max_real = x_max * w
        y_max_real = y_max * h
    
        color = (0,255,0)
        thickness = 20
        img_bb = cv2.rectangle(cv2.cvtColor(translated_images[i], cv2.COLOR_BGR2RGB), (int(x_min_real), int(y_min_real)), (int(x_max_real), int(y_max_real)), color, thickness)
        axes[i].imshow(img_bb)
        
    plt.show()


def save_labels_to_text_file(bboxes: list, filepath: str, has_wildfire:str):
    """Save labels to text files, separator is ' ': 
        has_wildfire x_center y_center bbox_width bbox_height

    Args:
        bboxes (list): list of tuple for YOLO format bounding boxes
        filepath (str): path for saving labels in txt file
        has_wildfire (str): "1" if yes, "0" if no
    """
    with open(filepath, 'w') as file:
        
        for label, bbox in zip(has_wildfire, bboxes):
            line = f"{label} {' '.join(map(str, bbox))}\n" 
            file.write(line)


def get_new_paths(img_path:str, label_path:str, i:int, translate_limit:int, output_dir:str, str_has_wildfire:str, scene_name:str) -> tuple:
    """Get new paths to save translated image and labels (class has_wildfire + YOLO bounding box)

    Args:
        img_path (str): path to image
        label_path (str): path to its label
        i (int): translation index
        translate_limit (int): number of pixels for translation
        output_dir (str): output directory for saving
        str_has_wildfire (str): "has_wildfire" or "no_wildfire"
        scene_name (str): name of scene (f'scene_{scene_index:02d}')

    Returns:
        tuple:  new_image_path (str): new path to save translated image
                new_label_path (str): new path to save translated labels
    """
    # EXAMPLE : img_path = "/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/images/train/0674.jpg"
    img_name_ext = img_path.split("/")[-1] # 0674.jpg
    # img_set = img_path.split("/")[-2] # train 
    img_dir = img_path.split("/")[-3] # images / labels
    img_dataset = img_path.split("/")[-4] # DS-71c1fd51-sam-synthetic 
    
    img_subdir0 = os.path.join(output_dir, f'{str_has_wildfire}--{img_dataset}')
    img_subdir1 = os.path.join(img_subdir0, img_dir)
    img_subdir2 = os.path.join(img_subdir1, f'{scene_name}_{translate_limit}')
    create_output_dir(img_subdir0)
    create_output_dir(img_subdir1)
    create_output_dir(img_subdir2)

    transformation = f"{'-' if translate_limit < 0 else ''}{abs(i * translate_limit):03d}"
    
    img_name, img_ext = os.path.splitext(img_name_ext)
    new_image_name = f"{img_name}_xtrans_{transformation}_{translate_limit}_{i}{img_ext}"
    new_image_path = os.path.join(img_subdir2, new_image_name)

    label_name_ext = label_path.split("/")[-1]
    label_name, label_ext = os.path.splitext(label_name_ext)
    label_dir = label_path.split("/")[-3]
    label_subdir2 = os.path.join(img_subdir0, label_dir)
    label_subdir3 = os.path.join(label_subdir2, f'{scene_name}_{translate_limit}')
    create_output_dir(label_subdir2)
    create_output_dir(label_subdir3)

    
    new_label_name = f"{label_name}_xtrans_{transformation}_{translate_limit}_{i}{label_ext}"
    new_label_path = os.path.join(label_subdir3, new_label_name)
    
    return new_image_path, new_label_path

def create_output_dir(output_dir: str):
    """Create output directory if it does not exist

    Args:
        output_dir (str): path to output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def save_translated_img_bboxes(img_path:str, label_path:str, n:int, translate_limit:int, output_dir:str, has_wildfire:str, scene_name:str):
    """Compute and save translated images and their YOLO bounding boxes

    Args:
        img_path (str): path to image_
        label_path (str): path to its label
        n (int): number of translations to apply on the image (n-1 translations)
        translate_limit (int): number of pixels for translation
        output_dir (str): output directory
        has_wildfire (str): "1" means "has_wildfire", "0" means "no_wildfire"
        scene_name (str): name of scene (f'scene_{scene_index:02d}')
    """
    # Load the original image
    image = cv2.imread(img_path)
    
    # Define the bounding box in YOLO format [x_center, y_center, bbox_width, bbox_height]
    yolo_bbox = get_label_yolo_bbox(label_path)

    # Perform n horizontal translations
    translated_images, translated_bboxes = translate_x_ntimes(image, yolo_bbox, translate_limit, n)

    if has_wildfire=="1":
        str_has_wildfire = "has_wildfire"
    if has_wildfire=="0":
        str_has_wildfire = "no_wildfire"

    # Save translated images and bboxes
    for i in range(n):    
        # Get paths for saving
        new_image_path, new_label_path =  get_new_paths(img_path, label_path, i, translate_limit, output_dir, str_has_wildfire, scene_name)
        translated_image = translated_images[i]
        # translated_image = cv2.cvtColor(translated_images[i], cv2.COLOR_BGR2RGB)

        # Save translated image
        cv2.imwrite(new_image_path, translated_image)
        
        # Get the translated bounding box
        x_center, y_center, width, height = translated_bboxes[i]
        yolo_bbox = [[x_center, y_center, width, height]]
       # Save it
        save_labels_to_text_file(yolo_bbox, new_label_path, has_wildfire)


def translate_has_wildfire(output_dir, has_wildfire="1", n=10, translate_limit_2right=50, translate_limit_2left=-50):
    img_paths_list_trans2right = [
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/images/train/0674.jpg',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/images/train/1205.jpg',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/images/train/1329.jpg',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/images/train/2224.jpg',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/images/train/13002.jpg',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/images/train/13505.jpg'
    ]


    label_paths_list_trans2right = [
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/labels/train/0674.txt',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/labels/train/1205.txt',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/labels/train/1329.txt',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/labels/train/2224.txt',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/labels/train/13002.txt',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/labels/train/13505.txt'
    ]


    img_paths_list_trans2left = [
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/images/train/0834.jpg',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/images/train/3306.jpg',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/images/train/8620.jpg',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/images/train/11995.jpg',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/images/train/14767.jpg',
    ]

    label_paths_list_trans2left = [
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/labels/train/0834.txt',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/labels/train/3306.txt',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/labels/train/8620.txt',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/labels/train/11995.txt',
        '/Users/marguerite/workspace_DS/DS-71c1fd51-sam-synthetic/labels/train/14767.txt',
    ]

    for scene_index, (img_path, label_path) in enumerate(zip(img_paths_list_trans2right, label_paths_list_trans2right)):
        scene_name = f'scene_{scene_index:02d}'
        save_translated_img_bboxes(img_path, label_path, n, translate_limit_2right, output_dir, has_wildfire, scene_name)

    for scene_index, (img_path, label_path) in enumerate(zip(img_paths_list_trans2left, label_paths_list_trans2left)):
        scene_name = f'scene_{scene_index+6:02d}'
        save_translated_img_bboxes(img_path, label_path, n, translate_limit_2left, output_dir, has_wildfire, scene_name)

def translate_no_wildfire(output_dir, has_wildfire="0", n=10, translate_limit_2right=50, translate_limit_2left=-50):
    no_fire_label_paths_list2right = [
        "/Users/marguerite/workspace_DS/pyronear_fp/labels/train/Pyronear_brison_2_2023_04_29T10_45_10.txt",
        "/Users/marguerite/workspace_DS/pyronear_fp/labels/train/Pyronear_marguerite_3_2023_04_26T07_27_24.txt",
        "/Users/marguerite/workspace_DS/pyronear_fp/labels/train/Pyronear_salaunes_1_3_2023_05_12T07_08_40.txt",
        "/Users/marguerite/workspace_DS/pyronear_fp/labels/train/Pyronear_salaunes_2_4_2023_04_30T17_54_39.txt",
        "/Users/marguerite/workspace_DS/pyronear_fp/labels/train/Pyronear_st_peray_1_2023_04_26T06_01_19.txt",
        "/Users/marguerite/workspace_DS/pyronear_fp/labels/train/Pyronear_st_peray_2_2023_04_26T07_07_54.txt",
        
    ]

    no_fire_img_paths_list2right = [
        "/Users/marguerite/workspace_DS/pyronear_fp/images/train/Pyronear_brison_2_2023_04_29T10_45_10.jpg",
        "/Users/marguerite/workspace_DS/pyronear_fp/images/train/Pyronear_marguerite_3_2023_04_26T07_27_24.jpg",
        "/Users/marguerite/workspace_DS/pyronear_fp/images/train/Pyronear_salaunes_1_3_2023_05_12T07_08_40.jpg",
        "/Users/marguerite/workspace_DS/pyronear_fp/images/train/Pyronear_salaunes_2_4_2023_04_30T17_54_39.jpg",
        "/Users/marguerite/workspace_DS/pyronear_fp/images/train/Pyronear_st_peray_1_2023_04_26T06_01_19.jpg",
        "/Users/marguerite/workspace_DS/pyronear_fp/images/train/Pyronear_st_peray_2_2023_04_26T07_07_54.jpg",
        
    ]

    no_fire_label_paths_list2left = [
        "/Users/marguerite/workspace_DS/pyronear_fp/labels/train/Pyronear_brison_1_2023_04_29T08_53_33.txt",
        "/Users/marguerite/workspace_DS/pyronear_fp/labels/train/Pyronear_brison_3_2023_04_26T06_07_02.txt", 
        "/Users/marguerite/workspace_DS/pyronear_fp/labels/train/Pyronear_brison_4_2023_05_09T15_29_53.txt", 
        "/Users/marguerite/workspace_DS/pyronear_fp/labels/train/Pyronear_marguerite_1_2023_05_09T13_48_52.txt",
        "/Users/marguerite/workspace_DS/pyronear_fp/labels/train/Pyronear_marguerite_2_2023_02_09T16_24_00.txt",
        
    ]

    no_fire_img_paths_list2left = [
        "/Users/marguerite/workspace_DS/pyronear_fp/images/train/Pyronear_brison_1_2023_04_29T08_53_33.jpg",
        "/Users/marguerite/workspace_DS/pyronear_fp/images/train/Pyronear_brison_3_2023_04_26T06_07_02.jpg", 
        "/Users/marguerite/workspace_DS/pyronear_fp/images/train/Pyronear_brison_4_2023_05_09T15_29_53.jpg", 
        "/Users/marguerite/workspace_DS/pyronear_fp/images/train/Pyronear_marguerite_1_2023_05_09T13_48_52.jpg",
        "/Users/marguerite/workspace_DS/pyronear_fp/images/train/Pyronear_marguerite_2_2023_02_09T16_24_00.jpg",
        
    ]  

    for scene_index, (img_path, label_path) in enumerate(zip(no_fire_img_paths_list2right, no_fire_label_paths_list2right)):
        scene_name = f'scene_{scene_index+11:02d}'
        save_translated_img_bboxes(img_path, label_path, n, translate_limit_2right, output_dir, has_wildfire, scene_name )

    for scene_index, (img_path, label_path) in enumerate(zip(no_fire_img_paths_list2left, no_fire_label_paths_list2left)):
        scene_name = f'scene_{scene_index+17:02d}'
        save_translated_img_bboxes(img_path, label_path, n, translate_limit_2left, output_dir, has_wildfire, scene_name)   

  
def main(output_dir):
    print("Perform translations - has_wildfire data")
    translate_has_wildfire(output_dir)
    print("Perform translations - no_wildfire data")
    translate_no_wildfire(output_dir)


if __name__ == "__main__":
    output_dir = '/Users/marguerite/workspace_DS/pyronear_CNN/mini_dataset_translations'
    main(output_dir)