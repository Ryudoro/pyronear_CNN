import pandas as pd
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import math
import matplotlib.patches as patches
from preprocess import Preprocess

def yolo_to_coords(x_center, y_center, bbox_w, bbox_h, img_w, img_h):
    x_min = x_center - bbox_w / 2
    y_min = y_center - bbox_h / 2
    x_max = x_center + bbox_w / 2
    y_max = y_center + bbox_h / 2
    x_min_real = int(x_min * img_w)
    y_min_real = int(y_min * img_h)
    x_max_real = int(x_max * img_w)
    y_max_real = int(y_max * img_h)
    return x_min_real, y_min_real, x_max_real, y_max_real

def create_gif(csv_file, duration, output_dir): 
    df = pd.read_csv(csv_file)
    preprocess = Preprocess(csv_file)
    seq_filtrees, bboxes = preprocess.get_sequence_bbox()

    # Interpolated bbox for iages while no label
    bboxes_inter = preprocess.preprocess_bbox(method="nearest", sigma=1)

    for i, seq in enumerate(seq_filtrees[:10]):
        # print(i, len(seq), seq[0])
        prefix_group_id = df.loc[df['Image_Path'] == seq[0], 'Prefix_group_id'].values[0]
        gif_path = os.path.join(output_dir, f"{prefix_group_id}.gif") 
        # print(gif_path)

        # Get list of bbox coordinates from label
        bbox_seq_xcenter = bboxes["bboxes_xcenter"][i]
        bbox_seq_ycenter = bboxes["bboxes_ycenter"][i]
        bbox_seq_width = bboxes["bboxes_width"][i]
        bbox_seq_height = bboxes["bboxes_height"][i]

        # Get list of interpolated bbox coordinates for no label
        bbox_seq_xcenter_inter = bboxes_inter["bboxes_xcenter"][i]
        bbox_seq_ycenter_inter = bboxes_inter["bboxes_ycenter"][i]
        bbox_seq_width_inter = bboxes_inter["bboxes_width"][i]
        bbox_seq_height_inter = bboxes_inter["bboxes_height"][i]
        
        with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
            for j, img_path in enumerate(seq): 
                # print(j, img_path)
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_h, img_w, _ = img.shape
                                
                # Get the corresponding bbox YOLO 
                x_center = bbox_seq_xcenter[j]
                y_center = bbox_seq_ycenter[j]
                bbox_w = bbox_seq_width[j]
                bbox_h = bbox_seq_height[j]
        
                # If there is no label, we put all the coords to 0
                if math.isnan(x_center): 
                    x_center = 0
                    y_center = 0
                    bbox_w = 0
                    bbox_h = 0
        
                x_min_real, y_min_real, x_max_real, y_max_real = yolo_to_coords(x_center, y_center, bbox_w, bbox_h, img_w, img_h)

                x_center_inter = bbox_seq_xcenter_inter[j]
                y_center_inter = bbox_seq_ycenter_inter[j]
                bbox_w_inter = bbox_seq_width_inter[j]
                bbox_h_inter = bbox_seq_height_inter[j]

                x_min_real_inter, y_min_real_inter, x_max_real_inter, y_max_real_inter = yolo_to_coords(x_center_inter, y_center_inter, bbox_w_inter, bbox_h_inter, img_w, img_h)
                
                # Create figure and axis
                fig, ax = plt.subplots()
                ax.imshow(rgb_img)
                rect = patches.Rectangle((x_min_real, y_min_real), 
                                        x_max_real-x_min_real, 
                                        y_max_real-y_min_real, 
                                        linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                rect_inter = patches.Rectangle((x_min_real_inter, y_min_real_inter), 
                                        x_max_real_inter-x_min_real_inter,
                                        y_max_real_inter-y_min_real_inter, linewidth=1, 
                                        edgecolor='b', facecolor='none')
                ax.add_patch(rect_inter)
                ax.axis('off')
                
                # Save the figure as a temporary image file
                temp_image_path = f'temp_image_{i}_{j}.png'
                plt.savefig(temp_image_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                writer.append_data(imageio.imread(temp_image_path))
                os.remove(temp_image_path)
                
            # Append a dark frame to end gif
            dark_frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            writer.append_data(dark_frame)
        print(f"Save gif: {gif_path}")
        print("------------")


if __name__ == "__main__":
    data_dir = "/Users/marguerite/workspace_DS/"
    csv_file = os.path.join(data_dir, "df_DS_fp_newlines_multiple_bbox.csv")
    # csv_file = os.path.join(data_dir, "df_pyronear_ds_03_2024_train_w_datetime_groups.csv")
    # csv_file = os.path.join(data_dir, "df_pyronear_ds_03_2024_val_w_datetime_groups.csv")

    # Check with gif : blue bbox for interpolated bbox, red bbox from labeled data
    duration = 1000
    output_dir = "/Users/marguerite/workspace_DS/pyronear_CNN/"
    create_gif(csv_file, duration, output_dir) 