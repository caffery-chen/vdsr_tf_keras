import tensorflow as tf
class VDSR:
    def __init__(self, d = 64, s = 32, m = 18, input_shape = None):
        self.d = d
        self.s = s
        self.m = m
        self.input_shape = input_shape
        self.activation = tf.nn.leaky_relu

    def build_model(self):
        inputs = tf.keras.Input(shape = self.input_shape)
        x = tf.keras.layers.Conv2D(filters = self.d, kernel_size = [3,3], padding='same', activation = self.activation, kernel_initializer = 'he_uniform')(inputs)
        for i in range(0,self.m):
            x = tf.keras.layers.Conv2D(filters = self.s, kernel_size = [3,3], padding='same', activation = self.activation, kernel_initializer = 'he_uniform')(x)

        x = tf.keras.layers.Conv2D(filters = self.input_shape[-1], kernel_size = [3,3], padding='same', activation = tf.keras.activations.linear, kernel_initializer = 'he_uniform')(x)
        prediction = tf.keras.layers.Add()([inputs, x])
        model = tf.keras.Model(inputs = inputs, outputs = prediction)
        model.summary()
        return model


if __name__ =='__main__':
    VDSR(d = 8, s = 4, m= 2, input_shape=[10, 180, 12]).build_model()

