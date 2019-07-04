import matplotlib.pyplot as plt


# basic models
# basic_on_KSC = [95.567, 96.233, 96.173, 95.917, 95.327]
basic_on_KSC = [95.76, 96.30, 96.18, 95.94, 95.55]

# bottleneck models

# bottle_on_KSC = [95.8, 96.49, 96.57, 95.973, 95.77]
bottle_on_KSC = [95.89, 96.55, 96.89, 95.94, 95.62]


# X axle
epoch_ba = ['10/14_b', '14_a/20', '18/26', '34/50', '38/56']


# KSC***********************************************************************
mean1 = basic_on_KSC

mean2 = bottle_on_KSC


plt.figure('Indian')

plt.plot(epoch_ba, mean1, label = 'basic block')
plt.plot(epoch_ba, mean2, label = 'bottle neck')

plt.grid(True)
plt.legend(loc=1)
plt.axis('tight')
plt.xlabel('Models')
plt.ylabel('Overall accuracy')

font = {'family': 'normal',
        'size': 14}
plt.rc('font', **font)

im_name = './img/' + 'KSC_cross_val.jpg'
plt.savefig(im_name)
plt.close()



