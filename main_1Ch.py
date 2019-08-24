import os
from nn_utility import *
from VDSR import VDSR
import numpy as np
import scipy.io as sio
import moxing as mox
mox.file.shift('os', 'mox')
import pdb

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

    L = np.size(x_data, 0)
    x_data = np.reshape(x_data, [L * 98, 1, 180,12])
    y_data = np.reshape(y_data, [L * 98, 1, 180,12])

    L = np.size(x_data, 0)
    train_len = int(np.ceil(0.8 * L))
    test_len = int(L-train_len)

    x_train = np.concatenate([np.real(x_data[0:train_len, :, :, :]), np.imag(x_data[0:train_len, :, :, :])], axis = 2)
    y_train = np.concatenate([np.real(y_data[0:train_len, :, :, :]), np.imag(y_data[0:train_len, :, :, :])], axis = 2)

    x_test = np.concatenate([np.real(x_data[train_len:train_len + test_len, :, :, :]), np.imag(x_data[train_len:train_len + test_len, :, :, :])], axis = 2)
    y_test = np.concatenate([np.real(y_data[train_len:train_len + test_len, :, :, :]), np.imag(y_data[train_len:train_len + test_len, :, :, :])], axis = 2)

    training_data = {'train_data':x_train, 'train_label':y_train, 'test_data':x_test, 'test_label':y_test}
    return training_data

def get_test_data_by_frame(file_path, frame_idx):
    f = os.listdir(file_path)
    x_data = []
    y_data = []
    for mat_file in f:
        if '_cframe_%d' % frame_idx in mat_file:
            xx = sio.loadmat(os.path.join(file_path, mat_file))
            x_data.append(xx['x_data'])
            y_data.append(xx['y_data'])

    L = np.size(x_data, 0)
    x_data = np.reshape(x_data, [L * 98, 1, 180,12])
    y_data = np.reshape(y_data, [L * 98, 1, 180,12])

    L = np.size(x_data, 0)
    x_test = np.concatenate([np.real(x_data[0:L, :, :, :]), np.imag(x_data[0:L, :, :, :])], axis = 2)
    y_test = np.concatenate([np.real(y_data[0:L, :, :, :]), np.imag(y_data[0:L, :, :, :])], axis = 2)

    test_data = {'test_data':x_test, 'test_label':y_test}
    return test_data

def get_val_data_by_frame(file_path, frame_idx):
    f = os.listdir(file_path)
    x_data = []
    y_data = []
    for mat_file in f:
        if '_cframe_%d' % frame_idx in mat_file:
            xx = sio.loadmat(os.path.join(file_path, mat_file))
            x_data.append(xx['x_data'])
            y_data.append(xx['y_data'])

    L = np.size(x_data, 0)
    x_data = np.reshape(x_data, [L * 98, 1, 180,12])
    y_data = np.reshape(y_data, [L * 98, 1, 180,12])

    L = np.size(x_data, 0)
    x_val = np.concatenate([np.real(x_data[0:L, :, :, :]), np.imag(x_data[0:L, :, :, :])], axis = 2)
    y_val = np.concatenate([np.real(y_data[0:L, :, :, :]), np.imag(y_data[0:L, :, :, :])], axis = 2)

    val_data = {'val_data':x_val, 'val_label':y_val}
    return val_data

#if __name__ == '__main__':
    #np.reshape([[[1,2,3],[1,3,4]],[[3,4,5],[5,6,7]]], [4,1,3])
    # data_path = r'C:\\FMF_NN_EQ\\ori_form'
    # training_data = data_preprocessing(data_path)
def main():
    data_path = r's3://obs-fmf-eq/frame_data'
    log_dir = r's3://obs-fmf-eq/model/08-24'
    training_data = get_data_by_frame(data_path, 1)
    test_data = get_test_data_by_frame(data_path, 5)#{'test_data': training_data['train_data'], 'test_label': training_data['train_label']}
    val_data = get_val_data_by_frame(data_path, 3)#{'val_data': training_data['train_data'], 'val_label': training_data['train_label']}
    model = VDSR(d=64, s=32, m=5, input_shape=[1, 360, 12]).build_model()
    train_model(training_data, test_data, val_data, model, tf.train.AdamOptimizer(0.001),log_dir)
    
if __name__ == '__main__':
    main()

