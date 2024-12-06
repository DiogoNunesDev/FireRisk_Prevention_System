import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, UpSampling2D, Concatenate, GlobalAveragePooling2D, Reshape, Dense
from tensorflow.keras.models import Model

# Encoder Block
def encoder_block(inputs, filters, kernel_size=3, strides=1, dilation_rate=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', dilation_rate=dilation_rate)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# ASPP Block
def aspp_block(inputs, filters):
    # 1x1 Convolution
    conv1x1 = Conv2D(filters, 1, padding='same', activation='relu')(inputs)
    
    # 3x3 Convolutions with different dilation rates
    conv3x3_1 = Conv2D(filters, 3, padding='same', dilation_rate=6, activation='relu')(inputs)
    conv3x3_2 = Conv2D(filters, 3, padding='same', dilation_rate=12, activation='relu')(inputs)
    conv3x3_3 = Conv2D(filters, 3, padding='same', dilation_rate=18, activation='relu')(inputs)
    
    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(inputs)
    global_avg_pool = Reshape((1, 1, -1))(global_avg_pool)  # Reshape to match Conv2D dimensions
    global_avg_pool = Conv2D(filters, 1, padding='same', activation='relu')(global_avg_pool)
    global_avg_pool = UpSampling2D(size=(inputs.shape[1], inputs.shape[2]), interpolation='bilinear')(global_avg_pool)
    
    # Concatenate all branches
    x = Concatenate()([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_avg_pool])
    x = Conv2D(filters, 1, padding='same', activation='relu')(x)  # Final projection
    return x

# DeepLabV3+ Decoder
def decoder_block(inputs, skip_connection, filters, target_height, target_width):
    # Resize to the target height and width using tf.image.resize
    x = tf.image.resize(inputs, (target_height, target_width), method='bilinear')
    x = Concatenate()([x, skip_connection])  # Concatenate with the skip connection
    
    # Further convolutions to refine the output
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    
    return x


# DeepLabV3+ Architecture
def deeplabv3_plus(input_shape=(512, 896, 3), num_classes=5):
    inputs = Input(shape=input_shape)
    
    # Encoder: Feature extraction (Backbone)
    x = encoder_block(inputs, 64, strides=2)
    x = encoder_block(x, 128, strides=2)
    x = encoder_block(x, 256, strides=2)
    
    # ASPP: Multi-scale context
    aspp = aspp_block(x, 256)
    
    # Decoder: Combine with low-level features
    low_level_features = encoder_block(inputs, 48)  # Extract low-level features from early layers
    decoder = decoder_block(aspp, low_level_features, 256, target_height=512, target_width=896)
    
    # Final output layer
    outputs = Conv2D(num_classes, 1, activation='softmax', padding='same')(decoder)
    
    # Create Model
    model = Model(inputs, outputs)
    return model