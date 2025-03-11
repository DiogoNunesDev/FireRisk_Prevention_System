import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, UpSampling2D, Concatenate, GlobalAveragePooling2D, Reshape, Dense
from tensorflow.keras.models import Model

def encoder_block(inputs, filters, kernel_size=3, strides=1, dilation_rate=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', dilation_rate=dilation_rate)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def aspp_block(inputs, filters):
    conv1x1 = Conv2D(filters, 1, padding='same', activation='relu')(inputs)
    
    conv3x3_1 = Conv2D(filters, 3, padding='same', dilation_rate=6, activation='relu')(inputs)
    conv3x3_2 = Conv2D(filters, 3, padding='same', dilation_rate=12, activation='relu')(inputs)
    conv3x3_3 = Conv2D(filters, 3, padding='same', dilation_rate=18, activation='relu')(inputs)
    
    global_avg_pool = GlobalAveragePooling2D()(inputs)
    global_avg_pool = Reshape((1, 1, -1))(global_avg_pool)  
    global_avg_pool = Conv2D(filters, 1, padding='same', activation='relu')(global_avg_pool)
    global_avg_pool = UpSampling2D(size=(inputs.shape[1], inputs.shape[2]), interpolation='bilinear')(global_avg_pool)
    
    x = Concatenate()([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_avg_pool])
    x = Conv2D(filters, 1, padding='same', activation='relu')(x)  # Final projection
    return x

def decoder_block(inputs, skip_connection, filters, target_height, target_width):
    x = tf.image.resize(inputs, (target_height, target_width), method='bilinear')
    x = Concatenate()([x, skip_connection]) 
    
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    
    return x


def deeplabv3_plus(input_shape=(512, 896, 3), num_classes=5):
    inputs = Input(shape=input_shape)
    
    x = encoder_block(inputs, 64, strides=2)
    x = encoder_block(x, 128, strides=2)
    x = encoder_block(x, 256, strides=2)
    
    aspp = aspp_block(x, 256)
    
    low_level_features = encoder_block(inputs, 48)  
    decoder = decoder_block(aspp, low_level_features, 256, target_height=512, target_width=896)
    
    outputs = Conv2D(num_classes, 1, activation='softmax', padding='same')(decoder)
    
    model = Model(inputs, outputs)
    return model