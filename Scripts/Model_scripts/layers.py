import tensorflow as tf
from tensorflow.keras.layers import Layer

class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='SAME', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        pooled, argmax = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=[1, *self.pool_size, 1],
            strides=[1, *self.strides, 1],
            padding=self.padding
        )
        argmax = tf.cast(argmax, tf.int32)
        return pooled, argmax

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // self.pool_size[0], input_shape[2] // self.pool_size[1], input_shape[3]), input_shape



class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs):
        updates, argmax = inputs
        input_shape = tf.shape(updates)
        batch_size = input_shape[0]
        output_height = input_shape[1] * self.size[0]
        output_width = input_shape[2] * self.size[1]
        num_channels = input_shape[3]

        output_shape = [batch_size, output_height, output_width, num_channels]

        flat_input_size = tf.size(updates)
        flat_output_size = tf.reduce_prod(output_shape)

        tf.print("Flat input size:", flat_input_size)
        tf.print("Argmax size:", tf.size(argmax))

        if tf.size(argmax) != flat_input_size:
            raise ValueError("Shape mismatch: updates size does not match argmax size")

        # Create indices for scatter_nd
        batch_range = tf.range(batch_size, dtype=tf.int32)
        batch_range = tf.reshape(batch_range, [-1, 1, 1, 1])
        batch_range = tf.tile(batch_range, [1, input_shape[1], input_shape[2], num_channels])

        b = tf.reshape(batch_range, [flat_input_size])
        y = tf.reshape(argmax // (output_width * num_channels), [flat_input_size]) % output_height
        x = tf.reshape(argmax // num_channels, [flat_input_size]) % output_width
        f = tf.reshape(tf.range(num_channels, dtype=tf.int32), [1, 1, 1, num_channels])
        f = tf.tile(f, [batch_size, input_shape[1], input_shape[2], 1])
        f = tf.reshape(f, [flat_input_size])

        indices = tf.stack([b, y, x, f], axis=1)
        values = tf.reshape(updates, [flat_input_size])

        ret = tf.scatter_nd(indices, values, output_shape)
        ret.set_shape((None, None, None, num_channels))

        return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3]
        )
