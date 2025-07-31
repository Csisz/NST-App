class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale", shape=(input_shape[-1],), initializer="ones", trainable=True)
        self.offset = self.add_weight(name="offset", shape=(input_shape[-1],), initializer="zeros", trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, [1, 2], keepdims=True)
        return self.scale * (x - mean) / tf.sqrt(var + self.epsilon) + self.offset

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, activation=True):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')
        self.norm = InstanceNormalization()
        self.activation = tf.keras.layers.ReLU() if activation else tf.identity

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = ConvLayer(filters, 3, 1)
        self.conv2 = ConvLayer(filters, 3, 1, activation=False)

    def call(self, x):
        return x + self.conv2(self.conv1(x))

class StyleTransferNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = tf.keras.Sequential([
            ConvLayer(32, 9, 1),
            ConvLayer(64, 3, 2),
            ConvLayer(128, 3, 2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            tf.keras.layers.UpSampling2D(),
            ConvLayer(64, 3, 1),
            tf.keras.layers.UpSampling2D(),
            ConvLayer(32, 3, 1),
            tf.keras.layers.Conv2D(3, 9, 1, padding='same', activation='sigmoid')
        ])

    def call(self, inputs):
        return self.model(inputs)