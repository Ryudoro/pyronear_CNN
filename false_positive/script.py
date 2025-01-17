import os
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import ResNet50, EfficientNetB0, InceptionV3, DenseNet121, Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def load_and_process_image(image_path, bbox, target_size=(224, 224)):
    """
    Traitement d'une image unique (bbox + redim)
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


# Non utilisé !
def merge_bboxes(image, bbox1, bbox2, target_size=(224, 224)):
    """
    Combine deux régions d'intérêt (ROIs) extraites d'une image en une seule image,
    adaptée à une taille cible avec des bordures noires pour compléter si nécessaire.
    
    Args:
        image (np.ndarray): L'image source.
        bbox1 (tuple): Les coordonnées de la première bbox (x_min, y_min, x_max, y_max).
        bbox2 (tuple): Les coordonnées de la deuxième bbox (x_min, y_min, x_max, y_max).
        target_size (tuple): La taille cible (height, width) pour la sortie.

    Retourne:
        np.ndarray: Une image combinée des deux BBoxes, adaptée à la taille cible.
    """
    roi1 = image[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
    roi2 = image[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
    
    height1, width1 = roi1.shape[:2]
    height2, width2 = roi2.shape[:2]
    
    max_height = max(height1, height2)
    scale1 = max_height / height1
    scale2 = max_height / height2
    
    new_width1 = int(width1 * scale1)
    new_width2 = int(width2 * scale2)
    
    roi1_resized = cv2.resize(roi1, (new_width1, max_height))
    roi2_resized = cv2.resize(roi2, (new_width2, max_height))

    combined = np.hstack((roi1_resized, roi2_resized))
    
    combined_height, combined_width = combined.shape[:2]
    
    scale = min(target_size[1] / combined_width, target_size[0] / combined_height)
    resized_width = int(combined_width * scale)
    resized_height = int(combined_height * scale)
    combined_resized = cv2.resize(combined, (resized_width, resized_height))
    
    top_border = (target_size[0] - resized_height) // 2
    bottom_border = target_size[0] - resized_height - top_border
    left_border = (target_size[1] - resized_width) // 2
    right_border = target_size[1] - resized_width - left_border
    
    final_image = cv2.copyMakeBorder(
        combined_resized, top_border, bottom_border, left_border, right_border, 
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    
    return final_image



def process_df_for_resnet(df, target_size=(224, 224), base_path="dataset_pyronear_yolo_lstm/DS_fp/", min_num=100, label_cat="arbre"):
    """
    Traite un DataFrame pour préparer les données d'entraînement d'un modèle ResNet.
    Exclut les catégories sous-représentées en fonction de `min_num`.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les informations des BBoxes et des images.
        target_size (tuple): La taille cible des images (height, width).
        base_path (str): Le chemin de base pour accéder aux images.
        min_num (int): Le nombre minimum d'occurrences d'une catégorie pour être incluse.

    Returns:
        tuple: (X, y) où X est une liste d'images préparées, et y est une liste de labels correspondants.
    """
    X = []
    y = []
    nb_cat = 0
    nb_hors_cat = 0
    
    for _, row in df.iterrows():
        image_path = os.path.join(base_path, row['img_rel_path'])
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to load image at {image_path}")
            continue

        x_center, y_center = row['yolo_bbox_xcenter'], row['yolo_bbox_ycenter']
        width, height = row['yolo_bbox_width'], row['yolo_bbox_height']
        label = row['detection_label']
        



        if row['nb_detections'] == 1:
            y_row = 0
            if label_cat in label:
                nb_cat +=1
                y_row = 1
            else:
                nb_hors_cat+=1
            if y_row == 0 and nb_hors_cat-1000>nb_cat:
                print("too much hors cat")
                continue
            
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])

            roi = image[y_min:y_max, x_min:x_max]

            resized_roi = cv2.resize(roi, target_size)
            X.append(resized_roi)
            y.append(y_row)

        #Non gestion des doubles bbox (pour le moment)
        elif row['nb_detections'] == 2:
            continue

    label_counts = Counter(y)

    filtered_X = []
    filtered_y = []
    for i in range(len(y)):
        if label_counts[y[i]] >= min_num:
            filtered_X.append(X[i])
            filtered_y.append(y[i])



    unique_labels = sorted(set(filtered_y))
    category_mapping = {label: idx for idx, label in enumerate(unique_labels)}


    return filtered_X, filtered_y, category_mapping




def plot_image_with_predictions(image, true_label, predicted_label, confidence):
    """
    Affiche une image avec la catégorie prédite, celle réelle, et le pourcentage de confiance.

    Args:
        image (np.ndarray): L'image à afficher.
        true_label (str): La catégorie réelle.
        predicted_label (str): La catégorie prédite.
        confidence (float): Le pourcentage de confiance du modèle (entre 0 et 1).
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(image.astype(np.uint8))
    plt.axis('off')
    title = f"Réel: {true_label}\nPrédit: {predicted_label} ({confidence})"
    plt.title(title, fontsize=12)
    plt.show()


def create_model(backbone_name='resnet50', num_classes=1, input_shape=(224, 224, 3), freeze_layers=None) -> Model:
    """
    Crée un modèle de classification basé sur un backbone donné.

    Args:
        backbone_name (str): Le nom du backbone (ex: 'resnet50', 'efficientnet', 'inceptionv3', etc.).
        num_classes (int): Le nombre de classes pour la classification.
        input_shape (tuple): La taille des images d'entrée.
        freeze_layers (int): Nombre de couches à geler. Si None, ne gèle pas les couches.

    Returns:
        Model: Le modèle Keras compilé.
    """
    if backbone_name == 'resnet50':
        backbone = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == 'efficientnet':
        backbone = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == 'inceptionv3':
        backbone = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == 'densenet':
        backbone = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == 'xception':
        backbone = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if freeze_layers is not None:
        for layer in backbone.layers[:freeze_layers]:
            layer.trainable = False

    x = GlobalAveragePooling2D()(backbone.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=backbone.input, outputs=x)
    return model

def train_model(model, X_train, y_train, X_test, y_test, num_epochs=20, learning_rate=0.001):

    model.compile(
        optimizer=Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=num_epochs,
        batch_size=4,
        callbacks=[early_stopping]
    )
    return history, model

def custom_train_loop(model, X_train, y_train, X_test, y_test, num_epochs=10, batch_size=8, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name="test_accuracy")

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss.reset_state()
        train_accuracy.reset_state()
        test_loss.reset_state()
        test_accuracy.reset_state()

        for batch, (X_batch, y_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(X_batch, training=True)
                loss = loss_fn(y_batch, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(y_batch, predictions)

            if (batch + 1) % 10 == 0:
                print(f"  Batch {batch + 1}: Loss = {train_loss.result():.4f}, Accuracy = {train_accuracy.result():.4f}")

        for X_batch, y_batch in test_dataset:
            predictions = model(X_batch, training=False)
            loss = loss_fn(y_batch, predictions)

            test_loss(loss)
            test_accuracy(y_batch, predictions)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss.result():.4f}, Train Accuracy: {train_accuracy.result():.4f}")
        print(f"  Test Loss: {test_loss.result():.4f}, Test Accuracy: {test_accuracy.result():.4f}")

    return model

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

def display_predictions(model, X_test, y_test, category_mapping, num_images=5):
    reverse_mapping = {v: k for k, v in category_mapping.items()}
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(X_test[i])
        plt.title(f"True: {reverse_mapping[y_test[i]]}\nPred: {reverse_mapping[predicted_labels[i]]}")
        plt.axis('off')
    plt.show()
    
def get_top_predictions(models, image, top_n=3):
    """
    Teste plusieurs modèles sur une image et retourne les meilleurs prédictions.

    Args:
        models (dict): Un dictionnaire où les clés sont les noms des modèles et les valeurs sont les modèles eux-mêmes.
        image (np.ndarray): L'image à tester.
        top_n (int): Le nombre de meilleures prédictions à retourner.

    Returns:
        dict: Les noms des modèles et leurs scores pour les meilleures prédictions.
    """
    scores = {}

    for model_name, model in models.items():
        prediction = model.predict(np.expand_dims(image, axis=0))
        confidence = np.max(prediction) 
        scores[model_name] = confidence

    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    top_predictions = {k: sorted_scores[k] for k in list(sorted_scores)[:top_n]}
    return top_predictions
