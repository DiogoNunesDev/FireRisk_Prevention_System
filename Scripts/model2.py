import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

def segnet(input_shape, num_classes):
    # Encoder
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x, mask_1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x, mask_2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x, mask_3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x, mask_4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x, mask_5 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(x)
    
    print("Encoder Built...")

    # Decoder
    tf.print("Shape of input tensor x before unpooling block 5:", x)
    tf.print("Shape of mask_5 before unpooling block 5:", mask_5)
    x = MaxUnpooling2D(size=(2, 2))([x, mask_5])
    tf.print("Shape after unpooling block 5:", tf.shape(x))
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = MaxUnpooling2D(size=(2, 2))([x, mask_4])
    tf.print("Shape after unpooling block 4:", tf.shape(x))
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = MaxUnpooling2D(size=(2, 2))([x, mask_3])
    tf.print("Shape after unpooling block 3:", tf.shape(x))
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = MaxUnpooling2D(size=(2, 2))([x, mask_2])
    tf.print("Shape after unpooling block 2:", tf.shape(x))
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = MaxUnpooling2D(size=(2, 2))([x, mask_1])
    tf.print("Shape after unpooling block 1:", tf.shape(x))
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    print("Decoder Built...")    

    # Final layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    return model