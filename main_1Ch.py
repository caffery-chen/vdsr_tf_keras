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
        frame_data = np.zeros(shape=[12, 180, 74]) + 1j * np.zeros(shape=[12, 180, 74])
        for tri in range(0, 11):
            frame_data[tri, :,:] = np.transpose(tmp['RxPayloadArray_NN'][tri][0][0+frame*74 : 74+frame*74 , :])

        frame_calc =  (frame + kMemPol) if (frame + kMemPol) < 10 else (frame + kMemPol) - 9
        sio.savemat('%s\\frame_wise\\%s_frame_%d.mat'%(os.path.dirname(os.path.dirname(single_data_path)), os.path.basename(single_data_path).split('.mat')[0],frame_calc), {'frame_data':frame_data})
    # data: preamble + payload
    # data normalization to sqrt(2)
    # label: premable + payload



if __name__ == '__main__':
    data_path = r'C:\\FMF_NN_EQ\\ori_form\\'
    training_data = data_preprocessing(data_path)
    #model = VDSR(d=8, s=4, m=2, input_shape=[10, 180, 12]).build_model()
    #train_model(training_data, model, tf.train.AdamOptimizer(0.001))

