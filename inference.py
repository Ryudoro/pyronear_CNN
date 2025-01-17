import os 
import glob
import numpy as np
import cv2
import tensorflow as tf
import argparse
from utils_model import build_teacher_model
from utils_dataset import load_and_process_image, load_and_process_image2, yolo_bbox2abs_pix_coords, load_and_process_image_for_resnet
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO,  # Set the log level to INFO
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Create a logger object
logger = logging.getLogger(__name__)


class ModelBase:
    def __init__(self, model_path=None, model_weights=None):
        self.model = None
        self.model_path = model_path
        self.model_weights = model_weights
    
    def load_model(self):
        """Load the model from the given path or instantiate a new model."""
        raise NotImplementedError("Must be implemented by the subclass.")
    
    def preprocess_input(self, images, bboxes):
        """Preprocess input data (images and bounding boxes) for model prediction."""
        raise NotImplementedError("Must be implemented by the subclass.")
    
    def predict(self, preprocessed_data):
        """Run the prediction on the preprocessed data."""
        raise NotImplementedError("Must be implemented by the subclass.")


class CNNModel(ModelBase):
    def __init__(self, model_path=None, model_weights=None):
        super().__init__(model_path, model_weights=None)
    
    def load_model(self):
        """Loads CNN model architecture and weights if a path is provided."""
        if self.model_path:
            if self.model_path.endswith("*.weights.h5"):
                raise ValueError("These are model weights.")
            else:
                self.model = tf.keras.models.load_model(self.model_path) # model path ends with ".h5"
        if self.model_weights: 
            self.model = build_teacher_model()
            self.model.load_weights(self.model_weights)
        else:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1) 
            ])
            self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def preprocess_input(self, images, bboxes):
        """Preprocess the images and bounding boxes (resize, normalize, etc.)."""
        preprocessed_images = map(
            lambda args: load_and_process_image_for_resnet(args[0], args[1]),
            zip(images_paths, bboxes)
        ) 

        #TODO :change the preprocessing function depending on the model
        # preprocessed_images = map(
        #     lambda args: load_and_process_image(args[0], args[1]),
        #     zip(images_paths, bboxes)
        # ) 

        # preprocessed_images = map(
        #     lambda args: load_and_process_image2(args[0], args[1]),
        #     zip(images_paths, bboxes)
        # ) 

        preprocessed_images = np.array(list(preprocessed_images))
        preprocessed_bboxes = np.array(bboxes)
        return np.array(preprocessed_images), preprocessed_bboxes

    def predict(self, preprocessed_data):
        """Make predictions on the preprocessed images."""
        preprocessed_images, preprocessed_bboxes = preprocessed_data
        input_tensor = tf.convert_to_tensor(preprocessed_images)
        return self.model.predict(input_tensor)


class PredictionPipeline:
    def __init__(self, model):
        self.model = model
    
    def run(self, images, bboxes):
        """Pipeline to load weights, preprocess inputs, and make predictions."""
        self.model.load_model() 
        
        preprocessed_data = self.model.preprocess_input(images, bboxes)
        
        predictions = self.model.predict(preprocessed_data)
        return predictions
    
    def label(self, predictions): 
        if len(predictions.shape): 
            predictions = np.max(predictions)
        final_prediction = "fire" if predictions >= 0.5 else "not fire"
        return final_prediction


def check_basenames(images_paths, bboxes_paths): 
    for img_path, bbox_path in zip(images_paths, bboxes_paths):
        img_basename, _ = os.path.splitext(os.path.basename(img_path))
        bbox_file_basename, _ = os.path.splitext(os.path.basename(bbox_path))
        assert img_basename == bbox_file_basename


def read_bbox_from_file(label_path):
    with open(label_path, "r") as f:    
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                label, x_center, y_center, bbox_width, bbox_height = map(float, parts)

            if len(parts) == 6:
                label, x_center, y_center, bbox_width, bbox_height, pred = map(float, parts)
                
    return [x_center, y_center, bbox_width, bbox_height]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process inference for time-series images and bboxes.')
    parser.add_argument('-w', '--model_weights', type=str, help='Path to the weights.h5 of the model.')
    parser.add_argument('-i', '--input_dir', type=str,
                        help='Directory with time-series images and labels to infer.')
    args = parser.parse_args()
    
    # Get batch of images 
    logger.info(f"Getting images from: {os.path.join(args.input_dir, 'images')}")
    logger.info(f"Getting labels from: {os.path.join(args.input_dir, 'labels')}")
    images_paths = sorted(glob.glob(os.path.join(args.input_dir, 'images', '*.jpg')))
    bboxes_paths = sorted(glob.glob(os.path.join(args.input_dir, 'labels', '*.txt')))

    check_basenames(images_paths, bboxes_paths)
    
    images = [cv2.imread(img_path) for img_path in images_paths]
    bboxes = list(map(read_bbox_from_file, bboxes_paths))
 
    # Instantiate model and run inference
    model = CNNModel(model_weights=args.model_weights)
    pipeline = PredictionPipeline(model)
    predictions = pipeline.run(images, bboxes)
    final_prediction = pipeline.label(predictions)
    print("Predictions:", predictions)
    print("final_prediction: ", final_prediction)
