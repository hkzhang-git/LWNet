import matplotlib.pyplot as plt

name_list=['3D-LWNet', '3D-LWNet\n+Salinas-4', '3D-LWNet\n+Salinas-8', '3D-LWNet\n+Salinas-12', '3D-LWNet\n+Salinas-16']

data_PaviaU_25=[88.37, 87.78, 90.00, 91.99, 92.54]
data_Indian_25=[84.08, 83.52, 84.34, 84.90, 85.23]
data_KSC_25=[93.73, 93.35, 94.48, 95.58, 96.48]

data_PaviaU_50=[95.57, 95.61, 97.11, 97.68, 98.25]
data_Indian_50=[94.18, 92.04, 92.37, 92.65, 94.23]
data_KSC_50=[98.48, 98.15, 99.04, 99.18, 99.46]

data=data_Indian_25
im_name = './img/' + 'Indian_25.jpg'

index=[0, 1, 2, 3, 4]

font = {'family': 'normal',
        'size': 12}
plt.rc('font', **font)

plt.ylim(ymax=100, ymin=80)
plt.xticks(index, name_list)
plt.ylabel('Overall accuracy')
# plt.xlabel('Methods')


colors=['yellowgreen', 'cyan', 'dodgerblue', 'dodgerblue', 'dodgerblue']
# colors=['yellowgreen', 'dodgerblue', 'dodgerblue', 'dodgerblue', 'dodgerblue']
# colors=['yellowgreen', 'cyan', 'cyan', 'cyan', 'dodgerblue']
for i in range(len(colors)):
    plt.bar(left=index[i], height=data[i], facecolor=colors[i], width=0.4, align='center', )
    plt.text(index[i]-0.25, data[i]+0.3, '{}%'.format(data[i]))

# rects=plt.bar(left=index, height=dataaviaU_25, facecolor=('chartreuse', 'plum', 'plum', 'plum', 'plum'), width=0.3, align="center")


plt.savefig(im_name)
plt.close()

