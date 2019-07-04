import matplotlib.pyplot as plt


# basic models
PaviaU_n_oa = [97.87, 98.02, 97.67, 98.01, 97.89]
PaviaU_n_aa = [97.87, 98.02, 97.67, 98.01, 97.89]
PaviaU_n_k = [97.60, 97.78, 97.38, 97.77, 97.63]

Indian_n_oa = [98.02, 98.25, 98.48, 98.30, 98.29]
Indian_n_aa = [98.54, 98.70, 98.89, 98.75, 98.73]
Indian_n_k = [97.85, 98.06, 98.36, 98.16, 98.15]

KSC_n_oa = [92.56, 94.10, 94.80, 93.40, 90.45]
KSC_n_aa = [87.91, 90.31, 92.19, 88.30, 86.26]
KSC_n_k = [91.67, 93.40, 94.19, 92.62, 89.33]

# bottle_models
PaviaU_b_oa = [98.24, 98.40, 98.26, 98.01, 98.06]
PaviaU_b_aa = [98.24, 98.40, 98.26, 98.01, 98.06]
PaviaU_b_k = [98.02, 98.20, 98.05, 97.77, 97.81]

Indian_b_oa = [98.44, 98.68, 98.58, 98.31, 97.94]
Indian_b_aa = [98.85, 99.03, 98.96, 98.75, 98.24]
Indian_b_k = [98.31, 98.57, 98.47, 98.40, 97.63]

KSC_b_oa = [94.80, 95.35, 95.51, 93.96, 89.41]
KSC_b_aa = [92.93, 93.23, 92.84, 92.78, 83.60]
KSC_b_k = [94.19, 94.79, 94.98, 93.25, 88.12]


dataset = 'PaviaU'

if dataset == 'PaviaU':
    n_oa, n_aa, n_k = PaviaU_n_oa, PaviaU_n_aa, PaviaU_n_k
    b_oa, b_aa, b_k = PaviaU_b_oa, PaviaU_b_aa, PaviaU_b_k
elif dataset == 'Indian':
    n_oa, n_aa, n_k = Indian_n_oa, Indian_n_aa, Indian_n_k
    b_oa, b_aa, b_k = Indian_b_oa, Indian_b_aa, Indian_b_k
elif dataset == 'KSC':
    n_oa, n_aa, n_k = KSC_n_oa, KSC_n_aa, KSC_n_k
    b_oa, b_aa, b_k = KSC_b_oa, KSC_b_aa, KSC_b_k

# X axle
epoch_ba = ['10/14_b', '14_a/20', '18/26', '34/50', '38/56']


plt.figure(dataset)

font = {'family': 'normal',
        'size': 14}
plt.rc('font', **font)

plt.plot(epoch_ba, n_oa, label = 'basic block')
plt.plot(epoch_ba, b_oa, label = 'bottleneck')

plt.grid(True)
plt.legend(loc=1)
plt.axis('tight')
plt.xlabel('Models')
plt.ylabel('Overall Accuracy')


im_name = './img/' + '{}_OA.jpg'.format(dataset)
plt.savefig(im_name)
plt.close()


plt.figure(dataset)

font = {'family': 'normal',
        'size': 14}
plt.rc('font', **font)

plt.plot(epoch_ba, n_k, label = 'basic block')

plt.plot(epoch_ba, b_k, label = 'bottleneck')

plt.grid(True)
plt.legend(loc=1)
plt.axis('tight')
plt.xlabel('Models')
plt.ylabel('Kappa coefficient (x100)')


im_name = './img/' + '{}_K.jpg'.format(dataset)
plt.savefig(im_name)
plt.close()


