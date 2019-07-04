import matplotlib.pyplot as plt


# basic models
basic_on_PaviaU = [98.365, 98.51, 98.665, 98.82, 98.545]
basic_on_Indian = [98.103,	98.387,	98.28,	98.363,	98.363]
basic_on_KSC = [95.567, 96.233, 96.173, 95.917, 95.327]
# bottleneck models
bottle_on_PaviaU = [98.945, 99.18, 98.94, 98.925, 98.51]
bottle_on_Indian = [98.277,	98.577,	98.193,	98.1, 97.84]
bottle_on_KSC = [95.8, 96.49, 96.57, 95.973, 95.77]

# X axle
epoch_ba = ['ResNet10/14_b', 'ResNet14_a/20', 'ResNet18/26', 'ResNet34/50', 'ResNet38/56']

# epoch_bt = ['ResNet14', 'ResNet20', 'ResNet26', 'ResNet50', 'ResNet56']

# Indian Pines***********************************************************************
plt.figure('Indian Pines')
plt.plot(epoch_ba, basic_on_Indian, label = 'basic block')
plt.plot(epoch_ba, bottle_on_Indian, label = 'bottle neck')

plt.plot(epoch_ba, basic_on_Indian, 'g*')
plt.plot(epoch_ba, bottle_on_Indian, 'bv')

plt.grid(True)
plt.legend(loc=1)
plt.axis('tight')
plt.xlabel('models')
plt.ylabel('overall accuracy')

font = {'family': 'normal',
        'size': 11}
plt.rc('font', **font)
# plt.show()
im_name = './img/' + 'Indian.jpg'
plt.savefig(im_name)
plt.close()


# Pavia University*******************************************************************
plt.figure('Pavia University')
plt.plot(epoch_ba, basic_on_PaviaU, label = 'basic block')
plt.plot(epoch_ba, bottle_on_PaviaU, label = 'bottle neck')

plt.plot(epoch_ba, basic_on_PaviaU, 'g*')
plt.plot(epoch_ba, bottle_on_PaviaU, 'bv')

plt.grid(True)
plt.legend(loc=1)
plt.axis('tight')
plt.xlabel('models')
plt.ylabel('overall accuracy')
# plt.rc('xtick', labelsize=11)
# plt.rc('ytick', labelsize=11)
font = {'family': 'normal',
        'size': 11}
plt.rc('font', **font)

# plt.show()
im_name = './img/' + 'PaviaU.jpg'
plt.savefig(im_name)
plt.close()


# Indian Pines***********************************************************************
plt.figure('Kennedy Space Center')
plt.plot(epoch_ba, basic_on_KSC, label = 'basic block')
plt.plot(epoch_ba, bottle_on_KSC, label = 'bottle neck')

plt.plot(epoch_ba, basic_on_KSC, 'g*')
plt.plot(epoch_ba, bottle_on_KSC, 'bv')

plt.grid(True)
plt.legend(loc=1)
plt.axis('tight')
plt.xlabel('models')
plt.ylabel('overall accuracy')

font = {'family': 'normal',
        'size': 11}
plt.rc('font', **font)
# plt.show()
im_name = './img/' + 'KSC.jpg'
plt.savefig(im_name)
plt.close()