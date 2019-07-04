import scipy.io as sio
import numpy as np
import h5py


# dataset = 'Salinas'
# dataset_HSI = 'salinas_corrected'
# dataset_gt = 'salinas_gt'
dataset = 'Pavia'
dataset_HSI = 'pavia'
dataset_gt = 'pavia_gt'

data_mat_dir = '../../../data/HSI_SRNet/data_mat/'
data_h5_dir = '../../../data/HSI_SRNet/data_h5/'

dataset_mat_dir = data_mat_dir + '{}/{}.mat'.format(dataset, dataset)
dataset_gt_dir = data_mat_dir + '{}/{}_gt.mat'.format(dataset, dataset)
dataset_h5_save_dir = data_h5_dir + '{}.h5'.format(dataset)

HSI_data = sio.loadmat(dataset_mat_dir)[dataset_HSI]
HSI_gt = sio.loadmat(dataset_gt_dir)[dataset_gt]

with h5py.File(dataset_h5_save_dir, 'w') as f:
    f['data'] = HSI_data
    f['label'] = HSI_gt