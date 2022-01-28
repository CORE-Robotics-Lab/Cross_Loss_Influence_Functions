from cross_loss_influence.config import DATA_DIR
import pickle
import os
from cross_loss_influence.data.scripts.my_pytorch_dataset import InfluenceMedDataset
import numpy as np


def write_indices_given_final_txts():
    N = 10000
    for txt_file in os.listdir(DATA_DIR):
        if 'test_results' in txt_file:
            file_prefix = txt_file.split('test_results')[0]
            with open(os.path.join(DATA_DIR, txt_file), 'r') as f:
                all_data = f.readlines()
            for line in all_data:
                if line.split(':')[0] == 'Harmful':
                    harmfuls = line
            all_ints = [t.split(',')[0] for t in harmfuls.split('[')[2:]]
            with open(os.path.join(DATA_DIR, file_prefix+'helpful_ids.txt'), 'w') as f:
                for index in all_ints[:N]:
                    f.writelines(str(index) + ' ')
            with open(os.path.join(DATA_DIR, file_prefix+'harmful_ids.txt'), 'w') as f:
                for index in all_ints[-N:]:
                    f.writelines(str(index) + ' ')


def test_results_to_ordered_txts():
    for txt_file in os.listdir(DATA_DIR):
        if txt_file.endswith('.txt') and 'test_results' in txt_file:
            fn = txt_file.split('test_results')[0]
            data = []
            with open(os.path.join(DATA_DIR, txt_file), 'r') as f:
                data = f.readlines()

            # harmfuls = [t.split(',')[0] for t in data[4].split('[')[2:]]
            # helpfuls = [t.split(',')[0] for t in data[5].split('[')[2:]]
            harmfuls = [t.split(',')[0] for t in data[2].split('[')[2:]]
            helpfuls = [t.split(',')[0] for t in data[3].split('[')[2:]]

            if 'scifi' in txt_file:
                dataset = InfluenceMedDataset(data_dir=DATA_DIR,
                                              filename='all_scripts_numericalized_dataset.pkl',
                                              window_size=3)
                stoi = pickle.load(open(os.path.join(DATA_DIR, 'all_scripts_stoi.pkl'), 'rb'))
            else:
                continue
                stoi = pickle.load(open(os.path.join(DATA_DIR, 'biased_data_stoi.pkl'), 'rb'))
                if 'biased' in txt_file:
                    dataset = InfluenceMedDataset(data_dir=DATA_DIR,
                                                  filename='biased_data_numericalized_dataset.pkl',
                                                  window_size=10)
                else:
                    dataset = InfluenceMedDataset(data_dir=DATA_DIR,
                                                  filename='neutral_data_numericalized_dataset.pkl',
                                                  window_size=10)

            keys = []
            for key, val in stoi.items():
                keys.append(key)

            f2 = os.path.join(DATA_DIR, fn + 'harmful_ordered.txt')
            for h in harmfuls:
                bad_indices = np.array(dataset.get_raw_sample(int(h[0])))
                b_ws = ''
                for index in range(len(bad_indices)):
                    try:
                        b_ws += keys[bad_indices[index]]
                        b_ws += ' '
                    except:
                        continue
                with open(f2, 'a', encoding='utf-8') as f:
                    f.write(str(b_ws))
                    f.write('\n')

            f3 = os.path.join(DATA_DIR, fn + 'helpful_ordered.txt')
            for h in helpfuls:
                good_indices = np.array(dataset.get_raw_sample(int(h[0])))
                g_ws = ''
                for index in range(len(good_indices)):
                    try:
                        g_ws += keys[good_indices[index]]
                        g_ws += ' '
                    except:
                        continue
                with open(f3, 'a', encoding='utf-8') as f:
                    f.write(str(g_ws))
                    f.write('\n')

if __name__ == "__main__":
    test_results_to_ordered_txts()
    write_indices_given_final_txts()