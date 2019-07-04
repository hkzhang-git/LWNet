import numpy as np


set_list = 'Salinas.txt'
subset_size=12


samples_txt = open(set_list).readlines()
label_array = np.array([f.split(' ')[-1][:-1] for f in samples_txt], int)

subset_loc = np.where(label_array<=subset_size)[0]

save_dir = set_list.split('.')[0]+'-{}.txt'.format(subset_size)

with open(save_dir, 'w') as f:
    for i in range(len(subset_loc)):
        f.write(samples_txt[subset_loc[i]])
