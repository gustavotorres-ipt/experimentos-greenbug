import tensorflow as tf
from tensorflow.keras import layers, models

# Channel Attention Module
class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        filters = input_shape[-1]
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(filters // self.reduction_ratio, activation='relu', use_bias=False)
        self.fc2 = layers.Dense(filters, activation='sigmoid', use_bias=False)

    def call(self, inputs):
        avg_out = self.avg_pool(inputs)
        avg_out = self.fc1(avg_out)
        avg_out = self.fc2(avg_out)
        avg_out = tf.reshape(avg_out, (-1, 1, 1, avg_out.shape[-1]))
        return inputs * avg_out

# Spatial Attention Module
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(1, kernel_size=self.kernel_size, strides=1, padding="same", activation='sigmoid')

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        return inputs * self.conv(concat)

# CSNN Model with Channel and Spatial Attention
class CSNNModel(tf.keras.Model):
    def __init__(self, num_classes=10, input_shape=(128, 128, 1)):
        super(CSNNModel, self).__init__()
        self.conv1 = layers.Conv2D(64, kernel_size=3, padding='same',
                                   activation='relu', input_shape=input_shape)
        self.ca1 = ChannelAttention()
        self.sa1 = SpatialAttention()

        self.conv2 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.ca2 = ChannelAttention()
        self.sa2 = SpatialAttention()

        self.pool = layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation='relu')

        self.dropout1 = layers.Dropout(0.5)  # Dropout layer with 50% drop probability
        self.fc2 = layers.Dense(256, activation='relu')

        self.dropout2 = layers.Dropout(0.5)  # Dropout layer with 50% drop probability
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.ca1(x)
        x = self.sa1(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = self.ca2(x)
        x = self.sa2(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return self.output_layer(x)


if __name__ == "__main__":
    # Create the model
    model = CSNNModel(num_classes=10)

    model.build((None, 128, 128, 1))  # Batch size is None, input shape is (128, 128, 3)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()
