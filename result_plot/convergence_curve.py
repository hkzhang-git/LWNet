import matplotlib.pyplot as plt
import numpy as np

txt_dir = 'convergence_curve/resnet26_Indian.txt'

info_list = open(txt_dir).readlines()
index = np.arange(1, 60, 2)


epoch = [int(info.split()[2][:-1].split(':')[1]) for info in info_list]
train_loss = [float(info.split()[3][:-1].split(':')[1]) for info in info_list]
val_loss = [float(info.split()[5][:-1].split(':')[1]) for info in info_list]


epoch = [epoch[id] for id in index]
train_loss = [train_loss[id] for id in index]
val_loss = [val_loss[id] for id in index]


plt.figure('Indian_convergence_curve')

font = {'family': 'normal',
        'size': 14}
plt.rc('font', **font)

plt.plot(epoch, train_loss, label = 'training loss')
plt.plot(epoch, val_loss, label = 'validation loss')

plt.grid(True)
plt.legend(loc=1)
plt.axis('tight')
plt.xlabel('Epoch')
plt.ylabel('Loss')



im_name = './img/' + 'Indian_curve.jpg'
plt.savefig(im_name)
plt.close()



