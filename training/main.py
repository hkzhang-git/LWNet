import time
import xlwt
import argparse
import datetime
import torch.optim as optim
import sys
sys.path.append('./models/')
from models_3D import dict
from functions_for_training import *
from functions_for_evaluating import acc_calculation
from torch.nn import DataParallel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Training settings
parser = argparse.ArgumentParser(description='HSI classification')
parser.add_argument('--dataset', type=str, default='Indian')
parser.add_argument('--model_name', type=str, default='resnet26')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--restore', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--test_batch_size', type=int, default=30)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--seed', default=1)
parser.add_argument('--model_save_interval', type=int, default=1)
parser.add_argument('--sheet_id', type=int, default=1)
args = parser.parse_args()

train_data_txt = '../data_preprocess/data_list/{}_train.txt'.format(args.dataset)
test_data_txt = '../data_preprocess/data_list/{}_test_part.txt'.format(args.dataset)

trained_model_dir = './train_info/' + args.model_name + '/'
train_info_record = trained_model_dir + args.model_name + '_{}'.format(args.sheet_id) + '.xls'

train_info_record_txt = trained_model_dir + args.model_name + '.txt'

torch.manual_seed(args.seed)

if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# data loaders
train_loader = torch.utils.data.DataLoader(
    data_loader(train_data_txt)
    , batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    data_loader(test_data_txt)
    , batch_size=args.test_batch_size)

make_if_not_exist(trained_model_dir)

if args.dataset == 'PaviaU':
    num_cla=9
elif args.dataset == 'Indian':
    num_cla=16
else:
    num_cla=13
model = DataParallel(dict[args.model_name](num_classes=num_cla, dropout_keep_prob=0))
if args.use_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)

start_epoch = 0
if args.restore and len(os.listdir(trained_model_dir)):
    model, start_epoch = model_restore(model, trained_model_dir)


for epoch in range(start_epoch+1, args.epochs+1):
    start = time.time()
    train(epoch, model, train_loader, optimizer, args)
    end = time.time()
    print('epoch: {} , cost {} seconds'.format(epoch,  end-start))

    if epoch % args.model_save_interval == 0:
        model_name = trained_model_dir + '/trained_model{}.pkl'.format(epoch)
        torch.save(model.cpu().state_dict(), model_name)
        if args.use_cuda: model.cuda()
        train_loss, train_acc = val(model, train_loader, args)
        print('train_loss: {:.4f}, train_acc: {:.2f}%'.format(train_loss, train_acc))
        val_loss, val_acc = val(model, test_loader, args)
        print('val_loss: {:.4f}, val_acc: {:.2f}%'.format(val_loss, val_acc))
        with open(train_info_record_txt, 'a') as f:
            f.write('timecost:{:.2f}, lr:{}, epoch:{}, train_loss:{:.4f}, train_acc:{:.2f}, val_loss:{:.6f}, val_acc:{:.2f}'.format(
                (end-start)/60, optimizer.param_groups[0]['lr'], epoch, train_loss, train_acc, val_loss, val_acc) + '\n'
            )

# acc = acc_calculation(model, test_loader, args)
# excel_write(train_info_record, acc)

# now_time = datetime.datetime.now().strftime('%Y-%m-%d')
# with open(train_info_record, 'a') as f:
#     f.write('date:{} acc:{}\n'.format(now_time, acc))

# info_plot(train_info_record)