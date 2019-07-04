import os
# import gpustat
import torch
import xlwt
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn.functional as F

from glob import glob
from torch.autograd import Variable
import torchvision.transforms

plt.switch_backend('agg')

def show_memusage(device=0):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    return 'memory used: {}MiB, total: {}MiB'.format(gpu_stats.gpus[device].memory_used, gpu_stats.gpus[device].memory_total)


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)


class data_loader(data.Dataset):
    def __init__(self, list_dir, augmentation=False):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)
        self.au = augmentation

    def __getitem__(self, index):
        sample_path = self.list_txt[index].split(' ')
        data_path = sample_path[0]
        label = sample_path[1][:-1]

        if not self.au:
            data = np.load(data_path)
        else:
            data = self.random_flip_lr(np.load(data_path))
            data = self.random_flip_tb(data)
            data = self.random_rot(data)

        label = int(label)-1
        return torch.from_numpy(data).float(), label

    def __len__(self):
        return self.length

    def random_flip_lr(self, data):
        if np.random.randint(0, 2):
            c, d, h, w = data.shape
            index = np.arange(w, 0, -1)-1
            return data[:,:,:, index]
        else:
            return data

    def random_flip_tb(self, data):
        if np.random.randint(0, 2):
            c, d, h, w = data.shape
            index = np.arange(h, 0, -1)-1
            return data[:,:,index,:]
        else:
            return data

    def random_rot(self, data):
        rot_k = np.random.randint(0, 4)
        return np.rot90(data, rot_k, (2, 3)).copy()


def model_restore(model, trained_model_dir):
    model_list = glob(trained_model_dir + "/*.pkl")
    a = []
    for i in range(len(model_list)):
        index = int(model_list[i].split('model')[-1].split('.')[0])
        a.append(index)
    epoch = np.sort(a)[-1]
    model_path = trained_model_dir + 'trained_model{}.pkl'.format(epoch)
    model.load_state_dict(torch.load(model_path))
    return model, epoch


def get_lr(epoch, lr, max_epochs):
    if epoch <= max_epochs * 0.84:
        lr = lr
    # elif epoch <= max_epochs * 0.8:
    #     lr = 0.1 * lr
    else:
        lr = 0.1 * lr
    return lr


def train(epoch, model, train_loader, optimizer, args):
    lr = get_lr(epoch, args.lr, args.epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('epoch: {}, lr: {}'.format(epoch , optimizer.param_groups[0]['lr']))
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # print(show_memusage(device=0))


def val(model, val_loader, args):
    model.eval()
    val_loss = 0
    correct = 0
    for data, target in val_loader:
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        val_loss += F.nll_loss(output, target, size_average=False).data.cpu().numpy()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()

    val_loss /= len(val_loader.dataset)
    val_acc = 100. * correct / len(val_loader.dataset)

    return val_loss.data[0], val_acc

def info_plot(info_txt):
    train_val_info = open(info_txt).readlines()
    epoch = [int(f.split('epoch:')[1].split(',')[0]) for f in train_val_info]
    train_acc = [float(f.split('train_acc:')[1].split(',')[0]) for f in train_val_info]
    val_acc = [float(f.split('val_acc:')[1].split(',')[0][:-1]) for f in train_val_info]

    plt.figure(info_txt[:-4])
    plt.plot(epoch, train_acc, label='train_acc')
    plt.plot(epoch, val_acc, label='test_acc')
    plt.plot(epoch, train_acc, 'g*')
    plt.plot(epoch, val_acc, 'b*')
    plt.grid(True)
    plt.legend(loc=4)
    plt.axis('tight')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    im_name = info_txt[:-4] + '.jpg'
    plt.savefig(im_name)


def excel_write(excel_dir, acc):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('sheet')
    for column, info in enumerate(acc):
        worksheet.write(0, column, info)
    workbook.save(excel_dir)


