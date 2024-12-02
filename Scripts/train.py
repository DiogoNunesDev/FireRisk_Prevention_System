import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
from model3 import unet
import matplotlib.pyplot as plt

# GPU Setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Constants
input_shape = (512, 896, 3) 
n_labels = 6
batch_size = 2
epochs = 400
learning_rate = 0.001

# Data Preparation
def load_data(images_path, masks_path, image_size):
    image_files = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.jpg')])
    mask_files = sorted([os.path.join(masks_path, f) for f in os.listdir(masks_path) if f.endswith('.png')])
    print(f"Number of images: {len(image_files)}, Number of masks: {len(mask_files)}")

    images, masks = [], []
    for img_file, mask_file in zip(image_files, mask_files):
        img = cv2.imread(img_file)
        img = cv2.resize(img, image_size)
        images.append(img)

        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
        masks.append(mask)

    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.int32)

    masks = np.expand_dims(masks, axis=-1)
    masks = to_categorical(masks, num_classes=n_labels)
    return images, masks

images_path = '../Data/Full'
masks_path = '../Masks'
X, y = load_data(images_path, masks_path, (input_shape[1], input_shape[0]))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Model Setup
model = unet(input_shape=input_shape, num_classes=n_labels)

def iou_metric(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    intersection = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=[1, 2])
    union = tf.reduce_sum(tf.cast(y_true, 'float32'), axis=[1, 2]) + tf.reduce_sum(tf.cast(y_pred, 'float32'), axis=[1, 2]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)

model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy', iou_metric])

# Callbacks
checkpoint = ModelCheckpoint('unet_best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(patience=50, verbose=1)

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // batch_size,
    validation_steps=len(X_val) // batch_size,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

model.save('unet_final_model.h5')

# Visualization
def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    if 'iou_metric' in history.history:
        plt.plot(history.history['iou_metric'], label='Training IoU')
        plt.plot(history.history['val_iou_metric'], label='Validation IoU')
    plt.legend()
    plt.title('Training History')
    plt.show()

def plot_predictions(model, X, y):
    idx = np.random.randint(0, len(X))  # Pick a random sample
    preds = model.predict(X[idx:idx+1])
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(X[idx])
    
    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(np.argmax(y[idx], axis=-1), cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(np.argmax(preds[0], axis=-1), cmap='gray')
    
    plt.show()

plot_history(history)
plot_predictions(model, X_train, y_train)
