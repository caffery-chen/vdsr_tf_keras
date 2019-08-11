import os
from nn_utility import *
from VDSR import VDSR
import numpy as np
import scipy.io as sio
def data_preprocessing(data_path):
    f = os.listdir(data_path)
    for mat_file in f:
        a =single_file_proc(os.path.join(data_path, mat_file))
    return training_data


def single_file_proc(single_data_path):
    sio.loadmat(single_data_path)
    # data: preamble + payload
    # data normalization to sqrt(2)
    # label: premable + payload



if __name__ == '__main__':
    data_path = r'C:\\FMF_NN_EQ\\'
    training_data = data_preprocessing(data_path)
    model = VDSR(d=8, s=4, m=2, input_shape=[10, 180, 12]).build_model()
    train_model(training_data, model, tf.train.AdamOptimizer(0.001))

