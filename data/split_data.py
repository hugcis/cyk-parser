import numpy as np

np.random.seed(333)

f = open('sequoia-corpus+fct.mrg_strict')
all_lines = np.array(f.read().split('\n'))
index = np.arange(len(all_lines))
np.random.shuffle(index)

train_set = all_lines[index[:int(0.8 * len(index))]]
val_set = all_lines[index[int(0.8 * len(index)):int(0.9 * len(index))]]
test_set = all_lines[index[int(0.9 * len(index)):]]

with open('train_data.mrg_strict', 'w') as f:
    f.write('\n'.join(train_set))

with open('val_data.mrg_strict', 'w') as f:
    f.write('\n'.join(val_set))

with open('test_data.mrg_strict', 'w') as f:
    f.write('\n'.join(test_set))
