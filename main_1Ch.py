import os
from nn_utility import *
from VDSR import VDSR
from ConvNet import ConvNet
import numpy as np
import scipy.io as sio
import moxing as mox
mox.file.shift('os', 'mox')
import pdb
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import traceback

def data_preprocessing(data_path):
    f = os.listdir(data_path)
    for mat_file in f:
        try:
            a =single_file_proc(os.path.join(data_path, mat_file))
        except:
            print(traceback.format_exc())
            print('%s has problem.' % mat_file)
    return a


def single_file_proc(single_data_path):
    tmp = sio.loadmat(single_data_path)
    frame_tested = int(np.size(tmp['RxPreambleArray_NN'][0][0],0)/24)
    kMemPol = tmp['kMemPol'][0][0]
    for frame in range(0, frame_tested):
        y_data = x_data = np.zeros(shape=[74 + 24, 180, 12]) + 1j * np.zeros(shape=[74 + 24, 180, 12])
        for tri in range(0, 12):
            x_data[0:24,:,  tri] =tmp['RxPreambleArray_NN'][tri][0][0 + frame * 24: 24 + frame * 24, :]
            x_data[24:98,:,  tri] = tmp['RxPayloadArray_NN'][tri][0][0 + frame * 74 : 74 + frame * 74 , :]

            y_data[0:24,:,  tri] = tmp['TxPreambleArray_NN'][tri][0][0 + frame * 24: 24 + frame * 24, :]
            y_data[24:98,:,  tri] = tmp['TxPayloadArray_NN'][tri][0][0 + frame * 74 : 74 + frame * 74 , :]

        y_data1 = np.reshape(y_data, [98, 1, 180, 12])
        y_data1 = np.concatenate([np.real(y_data1[0:98, :, :, :]), np.imag(y_data1[0:98, :, :, :])], axis=2)
        frame_calc = calc_ber(y_data1)
        
        sio.savemat('%s\\complete_frame\\%s_cframe_%d.mat'%(os.path.dirname(os.path.dirname(single_data_path)),
                                                          os.path.basename(single_data_path).split('.mat')[0],frame_calc), {'x_data':x_data, 'y_data':y_data})

def get_data_by_frame(file_path, frame_idx, usage):
    f = os.listdir(file_path)
    x_data = []
    y_data = []
    for mat_file in f:
        if '_cframe_%d.mat' % frame_idx in mat_file:
            xx = sio.loadmat(os.path.join(file_path, mat_file))
            x_data.append(xx['x_data'])
            y_data.append(xx['y_data'])

    L = np.size(x_data, 0)    
    x_data = np.reshape(x_data, [L * 98, 1, 180,12])
    y_data = np.reshape(y_data, [L * 98, 1, 180,12])

    L = np.size(x_data, 0)   

    nn_input = np.concatenate([np.real(x_data[0:L, :, :, :]), np.imag(x_data[0:L, :, :, :])], axis = 2)
    nn_output = np.concatenate([np.real(y_data[0:L, :, :, :]), np.imag(y_data[0:L, :, :, :])], axis = 2)
    
    if usage == 'train':
        data_tmp = {'train_data':nn_input, 'train_label':nn_output}
    elif usage == 'test':
        data_tmp = {'test_data':nn_input, 'test_label':nn_output}
    elif usage == 'validation':
        data_tmp = {'val_data':nn_input, 'val_label':nn_output}
        
    return data_tmp

def plot_constellation(c_number):
    plt.figure(200)
    plt.scatter(np.reshape(np.real(c_number),[-1]), np.reshape(np.imag(c_number),[-1]))
    plt.title('Constellation')
    plt.grid(True)
    plt.show()

def plot_constellation_save(y_predict):
    print('plot predicted constellation')
    real_part = np.reshape(y_predict[:,0,0:180,:], [-1])
    imag_part = np.reshape(y_predict[:,0,180:360,:],[-1])
    print('start plotting')
    print(len(real_part))
    plt.figure(100)
    plt.scatter(real_part, imag_part)
    plt.title('Constellation')
    plt.grid(True)
    plt.show()
    plt.savefig(r's3://obs-fmf-eq/model/08-24/constellation.png')
    print('plot ends')    
    
def model_validation(nn_type='VDSR', d= 64, s=32, m=5, data_path=None, chkpt_path=None, frame_idx=None):
    if nn_type == 'VDSR':
        model = VDSR(d,s,m, input_shape=[1, 360, 12]).build_model()
    elif nn_type == 'ConvNet':
        model = ConvNet(d,s,m, input_shape=[1, 360, 12]).build_model()
    
    model.load_weights(chkpt_path)
    val_data = get_data_by_frame(data_path, frame_idx, 'validation')['val_data']    
    y_predict = model.predict(val_data, steps = 4)    
    ber = calc_ber(y_predict)

def qam_demod(c_number):
    '''
    if real_part == -1 and imag_part == 1:
        return np.reshape([0,0],(2,1))
    elif real_part == 1 and imag_part == 1:
        return np.reshape([1,0], (2,1))
    elif real_part == 1 and imag_part == -1:
        return np.reshape([1,1], (2,1))
    elif real_part == -1 and imag_part == -1:
        return np.reshape([0,1], (2,1))
    '''
    real_part = np.real(c_number)
    imag_part = np.imag(c_number)
    
    real_part = np.round(real_part)
    imag_part = np.round(imag_part)
    
    real_part_bit = np.round((real_part + 1)/2)
    imag_part_bit = np.round(abs((imag_part -1 ))/2)

    output_bits = np.zeros([int(2*np.size(real_part_bit, 0)), int(np.size(real_part_bit, 1))])
    
    output_bits[0::2, :] = real_part_bit.astype(int)
    output_bits[1::2, :] = imag_part_bit.astype(int)
    
    return output_bits    
    
def calc_ber(y_predict):
    
    frame_length = 98
    rx_symbol = y_predict[:,0,0:180,:] + 1j * y_predict[:,0,180:360,:]
    frames_tested = int(np.size(y_predict,0)/frame_length)
    
    ber = np.zeros([frames_tested,12])
    rx_payload = []
    
    for i in range(0, frames_tested):        
        fut = rx_symbol[i*frame_length + 24:(i+1)*frame_length,:,:]       
        
        for Tri in range(0,12):
            output_bits = qam_demod(fut[Tri * 2 : 52 + Tri * 2,:,Tri])
            kMem = 0
            best_ber =1            
            for frame_idx in range(1,11):
                payload_bits = sio.loadmat("./payload_bits/bits_frame_%d.mat" % frame_idx)['Payload_bits']
                total_num_bits = output_bits.size
                total_error = np.sum(np.sum(np.abs(output_bits-payload_bits),0))
                ber_frame_idx = total_error / total_num_bits
                if best_ber>ber_frame_idx:
                    best_ber = ber_frame_idx
                    k=frame_idx
            
            print('Matched Frame Index %d; Frame#%d/%d; Tri:%d/%d: BER = %e' % (k, i+1, frames_tested, Tri+1, 12, best_ber))
            ber[i, Tri] = best_ber
    return k
            
def chkmkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def create_framewise_database():
    data_path = r'C:\\FMF_NN_EQ\\ori_frame'
    training_data = data_preprocessing(data_path)

def main(nn_type='ConvNet', d= 64, s=32, m=5, epoch = 2000, lr = 0.001, batch_size =256):    
    now = datetime.datetime.now()      
    #print(flags.data_url)
    #print(flags.train_url)
    log_dir = r's3://obs-fmf-eq/model/%s_d%d_s%d_m%d_ep%d_lr%.3f_bs%d/V0002' % (nn_type, d, s, m, epoch, lr, batch_size)
    chkmkdir(log_dir)
    
    data_path = r's3://obs-fmf-eq/complete_frame'    
    
    training_data = get_data_by_frame(data_path, 2, 'train')
    test_data = get_data_by_frame(data_path, 5, 'test')#{'test_data': training_data['train_data'], 'test_label': training_data['train_label']}
    val_data = get_data_by_frame(data_path, 3, 'validation')#{'val_data': training_data['train_data'], 'val_label': training_data['train_label']}
    
    if nn_type == 'VDSR':
        model = VDSR(d, s, m, input_shape=[1, 360, 12]).build_model()
    elif nn_type == 'ConvNet':
        model = ConvNet(d, s, m, input_shape=[1, 360, 12]).build_model()
    
    train_model(training_data, test_data, val_data, model, tf.train.AdamOptimizer(0.001),epoch, batch_size, log_dir)
    
if __name__ == '__main__':
    main(nn_type = 'ConvNet')

