from tensorflow.keras import layers, Model, Input

def residual_block(x, filters):
    shortcut = layers.Conv2D(filters, (1, 1), padding="same")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])  # Residual connection
    x = layers.ReLU()(x)
    return x


def inception_module(x, filters):
    branch1 = layers.Conv2D(filters, (1, 1), padding="same", activation="relu")(x)
    branch2 = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(branch1)
    branch3 = layers.Conv2D(filters, (5, 5), padding="same", activation="relu")(branch1)
    branch4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch4 = layers.Conv2D(filters, (1, 1), padding="same", activation="relu")(branch4)
    output = layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)
    return output


def dgsp_block(x, filters):
    pool = layers.GlobalAveragePooling2D()(x)
    pool = layers.Reshape((1, 1, -1))(pool)
    pool = layers.Conv2D(filters, (1, 1), padding="same")(pool)
    pool = layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation="bilinear")(pool)

    dilated1 = layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=1, activation="relu")(x)
    dilated2 = layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=2, activation="relu")(x)
    dilated3 = layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=3, activation="relu")(x)

    output = layers.concatenate([pool, dilated1, dilated2, dilated3], axis=-1)
    return output


def diResUnet(input_shape, num_classes):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = residual_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = residual_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = residual_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # Bottleneck with Inception and DGSPP
    c4 = dgsp_block(p3, 256)
    c4 = inception_module(c4, 256)
    
    # Decoder
    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    u5 = residual_block(u5, 128)
    
    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u5)
    u6 = layers.concatenate([u6, c2])
    u6 = residual_block(u6, 64)
    
    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u6)
    u7 = layers.concatenate([u7, c1])
    u7 = residual_block(u7, 32)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(u7)
    model = Model(inputs, outputs)
    return model
