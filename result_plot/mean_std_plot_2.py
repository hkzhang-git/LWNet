import pylab
import numpy

arr = numpy.array


# basic models
basic_on_PaviaU = arr([98.365, 98.51, 98.665, 98.82, 98.545])
basic_on_PaviaU_var = arr([0.328, 0.205, 0.211, 0.133, 0.009])

# bottleneck models
bottle_on_PaviaU = ([98.945, 99.18, 98.94, 98.925, 98.51])
bottle_on_PaviaU_var = ([0.015, 0.148, 0.034, 0.061, 0.405])

# X axle
epoch_ba = ['10/14_b', '14_a/20', '18/26', '34/50', '38/56']




# PaviaU***********************************************************************
mean1 = basic_on_PaviaU
var1 = basic_on_PaviaU_var

mean2 = bottle_on_PaviaU
var2 = bottle_on_PaviaU_var

pylab.figure('PaviaU')
# pylab.errorbar(epoch_ba, mean1, yerr=var1, fmt='', ecolor='yellowgreen', elinewidth=2)
pylab.errorbar()
pylab.show()
# plt.errorbar(epoch_ba, mean1, yerr=var1, fmt='v', color='dodgerblue', ecolor='yellowgreen',elinewidth=2)
# plt.errorbar(epoch_ba, mean2, yerr=var2, fmt='*', color='lime', ecolor='teal', elinewidth=2)
# plt.plot(epoch_ba, mean1, label = 'basic block')
# plt.plot(epoch_ba, mean2, label = 'bottle neck')
#
# plt.grid(True)
# plt.legend(loc=1)
# plt.axis('tight')
# plt.xlabel('Models')
# plt.ylabel('Overall accuracy')

font = {'family': 'normal',
        'size': 14}
plt.rc('font', **font)

im_name = './img/' + 'Indian.jpg'
plt.savefig(im_name)
plt.close()



