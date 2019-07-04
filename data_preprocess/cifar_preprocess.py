import argparse
from function_for_cifar_preprocess import *

source_dir = '../../../data/cifar/'
img_list_dir = './cifar_list/'

parser = argparse.ArgumentParser(description='cifar classification')
parser.add_argument('--dataset', type=str, default='cifar-100')
parser.add_argument('--frames', type=int, default= 12)
args = parser.parse_args()

if args.dataset=='cifar-10':
    # cifar_10_to_jpg(source_dir)
    jpg_dir = source_dir + 'cifar-10-jpg/'
    npy_dir = source_dir + 'cifar-10-npy/'
    cifar_jpg_to_npy(jpg_dir, npy_dir, img_list_dir, args.dataset, args.frames)

elif args.dataset=='cifar-100':
    # cifar_100_to_jpg(source_dir)
    jpg_dir = source_dir + 'cifar-100-jpg/'
    npy_dir = source_dir + 'cifar-100-npy/'
    cifar_jpg_to_npy(jpg_dir, npy_dir, img_list_dir, args.dataset, args.frames)

else:
    print('dataset not found')




