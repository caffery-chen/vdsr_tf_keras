import io
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.summary import summary as tf_summary
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

class ConstellationCallbacks(Callback):
    def __init__(self, logdir, period, val_data):
        super(ConstellationCallbacks, self).__init__()
        self.logdir = logdir
        self.period = period
        self.last_rcd = 0
        self.writer = tf_summary.FileWriter(self.logdir)
        self.validation_data = val_data

    def gen_plot(self, y_predict):
        real_part = np.reshape(y_predict[:,0,0:180,:], [-1])
        imag_part = np.reshape(y_predict[:,0,180:360,:],[-1])
        plt.figure()
        plt.scatter(real_part, imag_part)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf


    def on_epoch_end(self, epoch, logs=None):
        self.last_rcd = self.last_rcd + 1
        if self.last_rcd >= self.period:
            self.last_rcd = 0
            y_predict = self.model.predict(self.validation_data, steps = 32)

            # Prepare the plot
            plot_buf = self.gen_plot(y_predict)

            # Convert PNG buffer to TF image
            image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

            # Add the batch dimension
            image = tf.expand_dims(image, 0)

            # Add image summary
            # Session
            with tf.Session() as sess:
                # Run
                summary_op = tf.summary.image("plot", image)
                summary = sess.run(summary_op)
                # Write summary
                
                self.writer.add_summary(summary)
            
            #self.writer.add_summary(summary_op.eval())
        

    def on_train_end(self, logs=None):
        self.writer.close()
