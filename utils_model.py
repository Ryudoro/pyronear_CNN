import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean

class MetricsCallback(Callback):
    def __init__(self, train_dataset, val_dataset, batch_interval=10, log_file = "metrics_log.csv"):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_interval = batch_interval
        self.epoch = 0
        self.log_file = log_file
        self.metrics_history = {
            'batch': [],
            'epoch' : [],
            'train_loss': [],
            'train_accuracy': [],
            'train_precision': [],
            'train_recall': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': []
        }

    def on_batch_end(self, batch, logs=None):
        if batch % self.batch_interval == 0:
            y_val_true, y_val_pred, val_loss = self._get_predictions_and_loss(self.val_dataset)
            val_accuracy = accuracy_score(y_val_true, y_val_pred)
            val_precision = precision_score(y_val_true, y_val_pred, average='weighted')
            val_recall = recall_score(y_val_true, y_val_pred, average='weighted')

            y_train_true, y_train_pred, train_loss = self._get_predictions_and_loss(self.train_dataset)
            train_accuracy = accuracy_score(y_train_true, y_train_pred)
            train_precision = precision_score(y_train_true, y_train_pred, average='weighted')
            train_recall = recall_score(y_train_true, y_train_pred, average='weighted')

            self.metrics_history['batch'].append(batch)
            self.metrics_history['epoch'].append(self.epoch)
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['train_accuracy'].append(train_accuracy)
            self.metrics_history['train_precision'].append(train_precision)
            self.metrics_history['train_recall'].append(train_recall)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['val_accuracy'].append(val_accuracy)
            self.metrics_history['val_precision'].append(val_precision)
            self.metrics_history['val_recall'].append(val_recall)

            print(f'Batch {batch}: train_loss={train_loss}, train_accuracy={train_accuracy}, train_precision={train_precision}, train_recall={train_recall}, val_loss={val_loss}, val_accuracy={val_accuracy}, val_precision={val_precision}, val_recall={val_recall}')
            self._save_metrics_to_csv(batch=batch)

    def on_epoch_end(self, epoch, logs=None):
        y_val_true, y_val_pred, val_loss = self._get_predictions_and_loss(self.val_dataset)
        val_accuracy = accuracy_score(y_val_true, y_val_pred)
        val_precision = precision_score(y_val_true, y_val_pred, average='weighted')
        val_recall = recall_score(y_val_true, y_val_pred, average='weighted')

        y_train_true, y_train_pred, train_loss = self._get_predictions_and_loss(self.train_dataset)
        train_accuracy = accuracy_score(y_train_true, y_train_pred)
        train_precision = precision_score(y_train_true, y_train_pred, average='weighted')
        train_recall = recall_score(y_train_true, y_train_pred, average='weighted')

        self.metrics_history['batch'].append(-1)
        self.metrics_history['epoch'].append(self.epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_accuracy'].append(train_accuracy)
        self.metrics_history['train_precision'].append(train_precision)
        self.metrics_history['train_recall'].append(train_recall)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['val_accuracy'].append(val_accuracy)
        self.metrics_history['val_precision'].append(val_precision)
        self.metrics_history['val_recall'].append(val_recall)
        
        print(f'Epoch {epoch}: train_loss={train_loss}, train_accuracy={train_accuracy}, train_precision={train_precision}, train_recall={train_recall}, val_loss={val_loss}, val_accuracy={val_accuracy}, val_precision={val_precision}, val_recall={val_recall}')
        self.epoch +=1
        self._save_metrics_to_csv(epoch=epoch)

    def _get_predictions_and_loss(self, dataset):
        y_true = []
        y_pred = []
        losses = []
        for batch in dataset:
            x, y = batch
            y_true.append(y.numpy())
            preds = self.model.predict(x)
            y_pred.append((preds > 0.5).astype("int32"))
            loss = self.model.evaluate(x, y, verbose=0)
            losses.append(loss)
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        avg_loss = np.mean(losses)
        return y_true, y_pred, avg_loss
    

    def _save_metrics_to_csv(self, batch=None, epoch=None):
        df = pd.DataFrame(self.metrics_history)
        # if batch is not None or epoch is not None:
        #     df['batch'] = df['batch'].apply(lambda x: f"{self.epoch}_{x}" if self.epoch is not None else x)

        df.to_csv(self.log_file, index=False)




def build_teacher_model(sequence_length=5, num_classes=1):
    resnet = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    resnet.trainable = False

    input_shape = (sequence_length, 224, 224, 3)
    inputs = Input(shape=input_shape)

    time_distributed_resnet = TimeDistributed(resnet)(inputs)


    x = time_distributed_resnet
    for _ in range(2):
        x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256, return_sequences=False)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_student_model(sequence_length=5, num_classes=5):
    resnet = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    resnet.trainable = False

    input_shape = (sequence_length, 224, 224, 3)
    inputs = Input(shape=input_shape)

    time_distributed_resnet = TimeDistributed(resnet)(inputs)

    x = LSTM(256, return_sequences=False)(time_distributed_resnet)

    if sequence_length > 1:
        outputs = Dense(num_classes, activation='softmax')(x)
    else:
        outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model




def validate(val_dataset, student_model):
    accuracy_metric = CategoricalAccuracy()
    
    for images, labels in val_dataset:
        logits = student_model(images, training=False)
        accuracy_metric.update_state(labels, tf.nn.softmax(logits))
        
    return accuracy_metric.result().numpy()

def distillation_loss(teacher_logits, student_logits, true_labels, temperature, alpha):
    soft_labels = tf.nn.softmax(teacher_logits / temperature)

    student_loss = categorical_crossentropy(
        y_true=soft_labels, 
        y_pred=student_logits / temperature, 
        from_logits=True
    )
    
    label_loss = categorical_crossentropy(
        y_true=true_labels, 
        y_pred=student_logits, 
        from_logits=True
    )
    
    student_loss_mean = tf.reduce_mean(student_loss)
    label_loss_mean = tf.reduce_mean(label_loss)
    
    combined_loss = (alpha * student_loss_mean) + ((1 - alpha) * label_loss_mean)
    
    return combined_loss


@tf.function
def train_step(images, labels, student_model, teacher_model, optimizer, temperature=5.0, alpha=0.5):
    with tf.GradientTape() as tape:
        teacher_logits = teacher_model(images, training=False)
        student_logits = student_model(images, training=True)
        
        loss = distillation_loss(teacher_logits, student_logits, labels, temperature, alpha)
        
    gradients = tape.gradient(loss, student_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
    return loss

@tf.function
def train_step_simple(images, labels, model, optimizer):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss




def train_student_with_distillation(student_model, teacher_model, train_dataset, val_dataset, epochs=10, temperature=5.0, alpha=0.5):
    optimizer = Adam()
    loss_metric = Mean()
    accuracy_metric = CategoricalAccuracy()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        loss_metric.reset_states()
        accuracy_metric.reset_states()

        for batch, (images, labels) in enumerate(train_dataset):
            loss = train_step(images, labels, student_model, teacher_model, optimizer, temperature, alpha)
            loss_metric.update_state(loss)

            preds = tf.nn.softmax(student_model(images, training=False))
            accuracy_metric.update_state(labels, preds)

            if batch % 10 == 0:
                print(f"Batch {batch}, Loss: {loss_metric.result().numpy()}, Accuracy: {accuracy_metric.result().numpy()}")

        print(f"Training Loss: {loss_metric.result().numpy()}")
        print(f"Training Accuracy: {accuracy_metric.result().numpy()}")

        val_accuracy = validate(val_dataset, student_model)
        print(f"Validation Accuracy: {val_accuracy:.4f}")



def combined_loss(y_true, y_pred, alpha=1.0):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred_class = tf.round(y_pred)
    
    true_positives = tf.reduce_sum(y_true * y_pred_class)
    possible_positives = tf.reduce_sum(y_true)
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    
    recall_penalty = 1 - recall

    combined_loss = bce + alpha * recall_penalty
    
    return combined_loss




class RecallMaximizingLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.75, gamma=2, name="recall_maximizing_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
        pt = tf.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        
        return tf.reduce_mean(focal_loss)
    
def compile_with_loss(model, alpha=1.0, learning_rate = 1e-3):

    model.compile(optimizer=Adam(learning_rate = learning_rate), loss=RecallMaximizingLoss(alpha=0.75, gamma=2), metrics=["accuracy"])

def compile_and_fit(model, train_dataset, val_dataset, initial_epochs=10, fine_tune_epochs=10, fine_tune_layers=10, initial_lr=1e-3, fine_tune_lr=1e-4, alpha=1.0):
    model.compile(optimizer=Adam(learning_rate = initial_lr), loss=RecallMaximizingLoss(alpha=0.75, gamma=2), metrics=["accuracy"])
    print("Training with frozen layers")
    metrics_callback = MetricsCallback(train_dataset, val_dataset, batch_interval=5, log_file="frozen_metrics_log.csv")
    model.fit(train_dataset, epochs=initial_epochs, validation_data=val_dataset,callbacks =[metrics_callback])
    
    for layer in model.layers:
        if isinstance(layer, TimeDistributed):
            for sub_layer in layer.layer.layers[-fine_tune_layers:]:
                sub_layer.trainable = True

    model.compile(optimizer=Adam(learning_rate = fine_tune_lr), loss=RecallMaximizingLoss(alpha=0.75, gamma=2), metrics=["accuracy"])
    print("Fine-tuning with unfrozen layers")
    metrics_callback2 = MetricsCallback(train_dataset, val_dataset, batch_interval=5, log_file="unfrozen_metrics_log.csv")
    model.fit(train_dataset, epochs=fine_tune_epochs, validation_data=val_dataset,callbacks =[metrics_callback2])