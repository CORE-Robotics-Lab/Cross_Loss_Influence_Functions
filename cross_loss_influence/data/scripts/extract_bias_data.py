# Created by Andrew Silva
import pickle
from cross_loss_influence.config import DATA_DIR
import os


bias_data_fn = '/media/user/bias_data/WNC/biased.full'

bias_data = []
neutral_data = []
with open(bias_data_fn, 'r') as f:
    for line in f.readlines():
        data_dump = line.split('	')
        bias_data.append(data_dump[1])
        neutral_data.append(data_dump[2])

pickle.dump(bias_data, open(os.path.join(DATA_DIR, 'biased_data.pkl'), 'wb'))
pickle.dump(neutral_data, open(os.path.join(DATA_DIR, 'neutral_data.pkl'), 'wb'))