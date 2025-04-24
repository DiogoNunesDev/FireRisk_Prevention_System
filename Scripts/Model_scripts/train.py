import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from unet import UNet
from unet2 import unet as unet_v2
from DIResUnet import diResUnet
from deeplabv3_plus import deeplabv3_plus

tf.keras.backend.clear_session()
# GPU Setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


GPUS = ["GPU:0", "GPU:1"]
strategy = tf.distribute.MirroredStrategy( GPUS )
print('Number of devices: %d' % strategy.num_replicas_in_sync)

from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Constants
input_shape = (512, 512, 3) 
n_labels = 6
batch_size = 1
epochs = 2000
learning_rate =1e-4
#class_weights = tf.constant([0.0589, 0.0492, 0.0063, 0.7062, 0.1794], dtype=tf.float32)


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
    masks = masks - 1

    masks = np.expand_dims(masks, axis=-1)
    masks = to_categorical(masks, num_classes=n_labels)
    return images, masks

images_path_train = '../../Data/Train/images'
masks_path_train = '../../Data/Train/Masks'
images_path_val = '../../Data/Val/images'
masks_path_val = '../../Data/Val/Masks'
X_train, y_train = load_data(images_path_train, masks_path_train, (input_shape[1], input_shape[0]))
X_val, y_val = load_data(images_path_val, masks_path_val, (input_shape[1], input_shape[0]))


def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, (-1, n_labels))
        y_pred = tf.reshape(y_pred, (-1, n_labels))

        # Expand class weights to match the shape of y_true
        weight = tf.reduce_sum(y_true * tf.expand_dims(class_weights, 0), axis=-1)
        ce_loss = tf.reduce_sum(y_true * -tf.math.log(y_pred + 1e-7), axis=-1)
        weighted_loss = ce_loss * weight

        return tf.reduce_mean(weighted_loss)
    
    return loss


def iou_metric(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    intersection = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=[1, 2])
    union = tf.reduce_sum(tf.cast(y_true, 'float32'), axis=[1, 2]) + tf.reduce_sum(tf.cast(y_pred, 'float32'), axis=[1, 2]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)

def focal_loss(alpha, gamma):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * (1 - pt) ** gamma * bce
    return loss

def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)
    return 1 - (2 * intersection + smooth) / (union + smooth)



# Model Setup
with strategy.scope(): 
    #model = diResUnet(input_shape=input_shape, num_classes=n_labels)
    #model = diResUnet(input_shape=input_shape, num_classes=n_labels)
    #model = unet_v2(input_shape=input_shape, num_classes=n_labels)
    model = UNet(input_shape=input_shape, num_classes=n_labels)
    #model = deeplabv3_plus(input_shape=input_shape, num_classes=n_labels)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=['accuracy', iou_metric])

# Callbacks
checkpoint = ModelCheckpoint('unet_best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(patience=100, verbose=1)
reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, verbose=1, mode="max", min_lr=1e-6)

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),

    epochs=epochs,
    callbacks=[checkpoint, early_stopping, reduce_on_plateau],
    verbose=1
)

model.save('unet_final_model.h5')