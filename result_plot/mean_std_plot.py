import matplotlib.pyplot as plt


# basic models
basic_on_PaviaU = [98.365, 98.51, 98.665, 98.82, 98.545]
basic_on_Indian = [98.103,	98.387,	98.28,	98.363,	98.363]
basic_on_KSC = [95.567, 96.233, 96.173, 95.917, 95.327]

basic_on_PaviaU_var = [0.328, 0.205, 0.211, 0.133, 0.009]
basic_on_Indian_var = [0.293, 0.004, 0.148, 0.053, 0.063]
basic_on_KSC_var = [0.974, 1.226, 1.526, 0.770, 0.013]
# bottleneck models
bottle_on_PaviaU = [98.945, 99.18, 98.94, 98.925, 98.51]
bottle_on_Indian = [98.277,	98.577,	98.193,	98.1, 97.84]
bottle_on_KSC = [95.8, 96.49, 96.57, 95.973, 95.77]

bottle_on_PaviaU_var = [0.015, 0.148, 0.034, 0.061, 0.405]
bottle_on_Indian_var = [0.066, 0.116, 0.160, 0.044, 0.033]
bottle_on_KSC_var = [0.270, 0.58, 1.549, 0.423, 3.176]

# X axle
epoch_ba = ['10/14_b', '14_a/20', '18/26', '34/50', '38/56']




# Indian***********************************************************************
mean1 = basic_on_Indian
var1 = basic_on_Indian_var

mean2 = bottle_on_Indian
var2 = bottle_on_Indian_var

plt.figure('Indian')
plt.errorbar(epoch_ba, mean1, yerr=var1, fmt='v', color='dodgerblue', ecolor='yellowgreen',elinewidth=2)
plt.errorbar(epoch_ba, mean2, yerr=var2, fmt='*', color='lime', ecolor='teal', elinewidth=2)
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

im_name = './img/' + 'Indian.jpg'
plt.savefig(im_name)
plt.close()


# KSC***********************************************************************
mean1 = basic_on_KSC
var1 = basic_on_KSC_var

mean2 = bottle_on_KSC
var2 = bottle_on_KSC_var

plt.figure('Indian')
plt.errorbar(epoch_ba, mean1, yerr=var1, fmt='v', color='dodgerblue', ecolor='yellowgreen', elinewidth=2)
plt.errorbar(epoch_ba, mean2, yerr=var2, fmt='*', color='lime', ecolor='teal', elinewidth=2)
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

im_name = './img/' + 'KSC.jpg'
plt.savefig(im_name)
plt.close()


# PaviaU***********************************************************************
mean1 = basic_on_PaviaU
var1 = basic_on_PaviaU_var

mean2 = bottle_on_PaviaU
var2 = bottle_on_PaviaU_var

plt.figure('PaviaU')
plt.errorbar(epoch_ba, mean1, yerr=var1, fmt='v', color='dodgerblue', ecolor='yellowgreen', elinewidth=2)
plt.errorbar(epoch_ba, mean2, yerr=var2, fmt='*', color='lime', ecolor='teal', elinewidth=2)
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

im_name = './img/' + 'PaviaU.jpg'
plt.savefig(im_name)
plt.close()


