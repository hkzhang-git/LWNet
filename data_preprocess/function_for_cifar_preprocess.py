import os
import numpy as np
import _pickle as cPickle
from glob import glob
from scipy.misc import imsave, imread


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def cifar_10_to_jpg(source_dir):
    # training samples
    cifar_10_foders_create(source_dir, 'cifar-10-jpg/train/')
    for j in range(1, 6):
        batch_dir = source_dir + 'cifar-10-python/data_batch_' + str(j)
        img_dict = unpickle(batch_dir)

        for i in range(0, 10000):
            img = np.reshape(img_dict['data'][i], (3, 32, 32)).transpose(1, 2, 0)
            save_dir = source_dir + 'cifar-10-jpg/train/' + str(img_dict['labels'][i])+ '/{}.jpg'.format(i+(j-1)*10000)
            imsave(save_dir, img)
    # testing samples
    cifar_10_foders_create(source_dir, 'cifar-10-jpg/test/')

    batch_dir = source_dir + 'cifar-10-python/test_batch'
    img_dict = unpickle(batch_dir)
    for i in range(0, 10000):
        img = np.reshape(img_dict['data'][i], (3, 32, 32)).transpose(1, 2, 0)
        save_dir = source_dir + 'cifar-10-jpg/test/' + str(img_dict['labels'][i]) + '/{}.jpg'.format(
            i + (5 - 1) * 10000)
        imsave(save_dir, img)


def cifar_10_foders_create(source_dir, format='cifar-10-jpg/train'):
    for i in range(10):
        folder_dir = source_dir + format + str(i)
        make_if_not_exist(folder_dir)


def cifar_100_to_jpg(source_dir):
    # training samples
    cifar_100_foders_create(source_dir, 'cifar-100-jpg/train/')

    batch_dir = source_dir + 'cifar-100-python/train'
    img_dict = unpickle(batch_dir)
    for i in range(0, len(img_dict['fine_labels'])):
        img = np.reshape(img_dict['data'][i], (3, 32, 32)).transpose(1, 2, 0)
        save_dir = source_dir + 'cifar-100-jpg/train/' + str(img_dict['fine_labels'][i]) + '/{}.jpg'.format(
            i + 50000)
        imsave(save_dir, img)

    # testing samples
    cifar_100_foders_create(source_dir, 'cifar-100-jpg/test/')

    batch_dir = source_dir + 'cifar-100-python/test'
    img_dict = unpickle(batch_dir)
    for i in range(0, len(img_dict['fine_labels'])):
        img = np.reshape(img_dict['data'][i], (3, 32, 32)).transpose(1, 2, 0)
        save_dir = source_dir + 'cifar-100-jpg/test/' + str(img_dict['fine_labels'][i]) + '/{}.jpg'.format(
            i + 10000)
        imsave(save_dir, img)


def cifar_100_foders_create(source_dir, format='cifar-100-jpg/train'):
    for i in range(100):
        folder_dir = source_dir + format + str(i)
        make_if_not_exist(folder_dir)


def cifar_jpg_to_npy(jpg_dir, npy_dir, img_list_dir, dataset, frames):
    # training samples
    train_list_dir = img_list_dir + dataset + '_train.txt'
    train_txt=[]
    subset_list = os.listdir(os.path.join(jpg_dir, 'train'))
    for subset in subset_list:
        subset_save_dir = npy_dir + 'train/{}/'.format(subset)
        make_if_not_exist(subset_save_dir)
        subset_img_list = glob(jpg_dir + 'train/{}'.format(subset) + '/*.jpg')
        count=0
        for img_dir in subset_img_list:
            count+=1
            img =  imread(img_dir).transpose([2, 0, 1])
            img_npy = np.repeat(img, frames, 0)
            img_npy_save_dir = subset_save_dir + '{}.npy'.format(count)
            np.save(img_npy_save_dir, img_npy)
            train_txt.append(img_npy_save_dir + ' {}\n'.format(subset))

    with open(train_list_dir, 'w') as f:
        for info in train_txt:
            f.writelines(info)

    # testing samples
    train_list_dir = img_list_dir + dataset + '_test.txt'
    train_txt = []
    subset_list = os.listdir(os.path.join(jpg_dir, 'test'))
    for subset in subset_list:
        subset_save_dir = npy_dir + 'test/{}/'.format(subset)
        make_if_not_exist(subset_save_dir)
        subset_img_list = glob(jpg_dir + 'test/{}'.format(subset) + '/*.jpg')
        count = 0
        for img_dir in subset_img_list:
            count += 1
            img = imread(img_dir).transpose([2, 0, 1])
            img_npy = np.repeat(img, frames, 0)
            img_npy_save_dir = subset_save_dir + '{}.npy'.format(count)
            np.save(img_npy_save_dir, img_npy)
            train_txt.append(img_npy_save_dir + ' {}\n'.format(subset))

    with open(train_list_dir, 'w') as f:
        for info in train_txt:
            f.writelines(info)








