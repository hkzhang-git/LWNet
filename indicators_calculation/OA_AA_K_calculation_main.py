import os
import argparse
import numpy as np
import sys
sys.path.append('../training/models/')
from models_lw_3D import *
from functions_for_evaluating import *
from torch.nn import DataParallel


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Training settings
parser = argparse.ArgumentParser(description='model evaluation')
parser.add_argument('--test_data', type=str, default='../data_preprocess/data_list/Indian_train.txt')
parser.add_argument('--result_dir', type=str, default='./evaluation/')
parser.add_argument('--model_dir', type=str, default='./evaluation_models/Indian_trained_model.pkl')
parser.add_argument('--model_name', type=str, default='LWNet_3')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=20)
args = parser.parse_args()

samples_loader = torch.utils.data.DataLoader(
    data_loader(args.test_data)
    , batch_size=args.batch_size, shuffle=False)

model = DataParallel(dict[args.model_name](num_classes=16))
model.load_state_dict(torch.load(args.model_dir))

if args.use_cuda:
    model.cuda()

acc = acc_calculation(model, samples_loader, args)
print(acc.round(2))

