import fiftyone as fo
import fiftyone.zoo as foz
import os

def load_custom_dataset(dataset_dir, split):
    dataset = fo.Dataset(f"pyronear-{split}")
    
    images_dir = os.path.join(dataset_dir, "images", split)
    labels_dir = os.path.join(dataset_dir, "labels", split)
    
    for img_name in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_name)
        
        label_name = img_name.split('.')[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        detections = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x_center, y_center, width, height = map(float, parts)
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        x_max = x_center + width / 2
                        y_max = y_center + height / 2
                        detection = fo.Detection(label="smoke", bounding_box=[x_min, y_min, x_max, y_max])
                        detections.append(detection)
        else:
            print(f"Aucun fichier de label trouvé pour {img_name}, ajout d'une image sans annotations.")

        sample = fo.Sample(filepath=img_path, ground_truth=fo.Detections(detections=detections))
        dataset.add_sample(sample)
    
    return dataset

dataset_dir = 'DS-71c1fd51-sam-synthetic'
train_dataset = load_custom_dataset(dataset_dir, 'train')
val_dataset = load_custom_dataset(dataset_dir, 'val')

session = fo.launch_app(train_dataset)

while True:
    pass