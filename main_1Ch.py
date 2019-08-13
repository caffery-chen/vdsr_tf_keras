import os
from nn_utility import *
from VDSR import VDSR
import numpy as np
import scipy.io as sio

def data_preprocessing(data_path):
    f = os.listdir(data_path)
    for mat_file in f:
        try:
            a =single_file_proc(os.path.join(data_path, mat_file))
        except:
            print('%s has problem.' % mat_file)
    return a


def single_file_proc(single_data_path):
    frame_tested = 5
    tmp = sio.loadmat(single_data_path)
    kMemPol = tmp['kMemPol'][0][0]
    for frame in range(0, frame_tested):
        y_data = x_data = np.zeros(shape=[74 + 24, 180, 12]) + 1j * np.zeros(shape=[74 + 24, 180, 12])
        for tri in range(0, 11):
            x_data[0:24,:,  tri] =tmp['RxPreambleArray_NN'][tri][0][0 + frame * 24: 24 + frame * 24, :]
            x_data[24:98,:,  tri] = tmp['RxPayloadArray_NN'][tri][0][0+frame*74 : 74+frame*74 , :]

            y_data[0:24,:,  tri] = tmp['TxPreambleArray_NN'][tri][0][0 + frame * 24: 24 + frame * 24, :]
            y_data[24:98,:,  tri] = tmp['TxPayloadArray_NN'][tri][0][0+frame*74 : 74+frame*74 , :]

        frame_calc =  (frame + kMemPol) if (frame + kMemPol) < 10 else (frame + kMemPol) - 9
        sio.savemat('%s\\complete_frame\\%s_cframe_%d.mat'%(os.path.dirname(os.path.dirname(single_data_path)),
                                                           os.path.basename(single_data_path).split('.mat')[0],frame_calc), {'x_data':x_data, 'y_data':y_data})

def get_data_by_frame(file_path, frame_idx):
    f = os.listdir(file_path)
    x_data = []
    y_data = []
    for mat_file in f:
        if '_cframe_%d' % frame_idx in mat_file:
            xx = sio.loadmat(os.path.join(file_path, mat_file))
            x_data.append(xx['x_data'])
            y_data.append(xx['y_data'])


if __name__ == '__main__':
    #data_path = r'C:\\FMF_NN_EQ\\ori_form'
    #training_data = data_preprocessing(data_path)
    data_path = r'C:\\FMF_NN_EQ\\complete_frame'
    get_data_by_frame(data_path, 5)
    #model = VDSR(d=8, s=4, m=2, input_shape=[10, 180, 12]).build_model()
    #train_model(training_data, model, tf.train.AdamOptimizer(0.001))

