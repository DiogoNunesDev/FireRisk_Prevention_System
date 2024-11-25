import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
from model3 import unet
from tensorflow.keras.mixed_precision import set_global_policy

tf.keras.backend.clear_session()
set_global_policy('mixed_float16')
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)



print(tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("CUDA version:", tf.sysconfig.get_build_info()['cuda_version'])
print("cuDNN version:", tf.sysconfig.get_build_info()['cudnn_version'])

input_shape = (512, 896, 3)  # Adjusted to be divisible by 16
n_labels = 5
batch_size = 4
epochs = 200
learning_rate = 0.001

def load_data(images_path, masks_path, image_size):
    
    image_files = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.png')])
    mask_files = sorted([os.path.join(masks_path, f) for f in os.listdir(masks_path) if f.endswith('.png')])
    print(f"Number of images: {len(image_files)}, Number of masks: {len(mask_files)}")


    images = []
    masks = []

    for img_file, mask_file in zip(image_files, mask_files):
        img = cv2.imread(img_file)
        img = cv2.resize(img, image_size)  
        images.append(img)

        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)  
        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)


    images = images / 255.0
    masks = masks[..., np.newaxis]  
    masks = to_categorical(masks, num_classes=n_labels)

    return images, masks

images_path = '../Data/Full'
masks_path = '../Masks'


X, y = load_data(images_path, masks_path, (input_shape[1], input_shape[0]))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

print(f'Input shape: {input_shape}, Output classes: {n_labels}')

model = unet(input_shape=input_shape, num_classes=n_labels)

model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('unet_best_model.h5', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(patience=50, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

model.save('unet_final_model.h5')
