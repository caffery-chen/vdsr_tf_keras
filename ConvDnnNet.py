import tensorflow as tf
class ConvDnnNet:
    def __init__(self, d = 64, s = 32, m = 18, input_shape = None):
        self.d = d
        self.s = s
        self.m = m
        self.input_shape = input_shape
        self.activation = tf.nn.leaky_relu
        self.kernel_size = [1, 1]

    def build_model(self):
        inputs = tf.keras.Input(shape = self.input_shape)
        x = tf.keras.layers.Conv2D(filters = self.d, kernel_size = self.kernel_size, padding='same', kernel_initializer = 'he_uniform')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(self.activation)(x)

        for i in range(0,self.m):
            x = tf.keras.layers.Conv2D(filters = self.s, kernel_size = self.kernel_size, padding='same', kernel_initializer = 'he_uniform')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(self.activation)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=x.get_shape()[1], activation=self.activation, kernel_initializer = 'he_uniform')(x)
        prediction = tf.keras.layers.Dense(units=x.get_shape()[1], activation=self.activation, kernel_initializer='he_uniform')(x)

        model = tf.keras.Model(inputs = inputs, outputs = prediction)
        model.summary()
        return model


if __name__ =='__main__':
    ConvDnnNet(d = 8, s = 4, m= 2, input_shape=[1, 360, 12]).build_model()

