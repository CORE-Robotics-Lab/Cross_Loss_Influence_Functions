# Created by Andrew Silva
import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
try:
    from cross_loss_influence.data.scripts.data.scripts import py_data_speeder
except ImportError:
    from cross_loss_influence.data.scripts.cross_loss_influence.data.scripts import py_data_speeder


class MedText(Dataset):
    """Medical text dataset, adapted from the PyTorch tutorial at:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html."""

    def __init__(self, data_dir, window_size=4, num_negs=15):
        """
        Args:
            data_dir: directory where the dataset files live (from generate_and_save and generate_negatives)
            window_size: context window size
            num_negs: number of negatives per word-context pair
        """
        master_wc_dataset_fn = os.path.join(data_dir, f"wc_pairs_window_{window_size}.pkl")
        negative_fn = os.path.join(data_dir, f"negatives_{num_negs}.pkl")
        min_sentence_length = 3
        self.dataset = pickle.load(open(master_wc_dataset_fn, 'rb'))  # should be a master list of lists of tokens [[tok1, tok2, tok3, tok4], [tok5, tok2, tok1, tok6]]
        # self.dataset = h5py.File(os.path.join(data_dir, f'wc_pairs_window_{window_size}.h5'), 'r')
        # self.negatives = pickle.load(open(negative_fn, 'rb'))
        abstract_fn = os.path.join(data_dir, 'master_numericalized_dataset.pkl')  # Pickle is much faster than h5 to load in counts
        raw_data = pickle.load(open(abstract_fn, 'rb'))
        # raw_data = h5py.File(os.path.join(data_dir, 'master_numericalized_dataset.h5'), 'r')
        raw_counts = []

        for sent_index in range(len(raw_data)):
            # sent_index = str(sent_index)
            if len(raw_data[sent_index]) > min_sentence_length:  # add all words...? or just min sentence length ones? probably the latter
                raw_counts.extend(raw_data[sent_index])
        py_data_speeder.set_master_list(raw_counts)
        del raw_counts
        del raw_data

    def negatives_per_sample(self, sample_len):
        # return np.random.choice(self.negatives, sample_len)
        return py_data_speeder.get_negatives(15, sample_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = np.array(self.dataset[idx]).T.tolist()
        negatives = self.negatives_per_sample(len(sample))
        for sample_index in range(len(sample)):
            sample[sample_index].append(negatives[sample_index])
        return sample


class MedTextFiles(Dataset):
    """Medical text dataset, adapted from the PyTorch tutorial at:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html."""

    def __init__(self, data_dir, window_size=4, num_negs=15, min_abstract_len=4):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.dataset = [sam for sam in os.listdir(data_dir) if sam.endswith('.pkl')]
        self.context_window = window_size
        raw_data = []
        self.abstracts = []
        self.num_negs = num_negs
        for fn in self.dataset:
            data = pickle.load(open(os.path.join(data_dir, fn), 'rb'))
            good_abs = True
            for sent in data:
                if len(sent) < min_abstract_len:
                    good_abs = False
            if good_abs:
                for sent in data:
                    raw_data.extend(sent)
                self.abstracts.append(data)
        py_data_speeder.set_master_list(raw_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # data = pickle.load(open(os.path.join(self.data_dir, self.dataset[idx]), 'rb'))
        data = self.abstracts[idx]
        sample = py_data_speeder.get_abstract_sample(data, self.num_negs, self.context_window)
        return sample


class CythonMedDataset(Dataset):
    """Medical text dataset, adapted from the PyTorch tutorial at:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html."""

    def __init__(self, data_dir, filename, window_size=4, num_negs=15, min_sentence_length=3):
        """
        Args:
            data_dir (string): directory containing master pkl file
            filename (string): master pkl file with all data separated into sentences
            window_size (int, optional): Window size for contexts
            num_negs (int, optional): number of negatives to sample per input
            min_sentence_length (int, optional): throw out sentences shorter than this
        """
        abstract_fn = os.path.join(data_dir, filename)
        self.raw_data = pickle.load(open(abstract_fn, 'rb'))  # should be a master list of lists of tokens [[tok1, tok2, tok3, tok4], [tok5, tok2, tok1, tok6]]
        raw_counts = []
        self.raw_data = [s for s in self.raw_data if len(s) > min_sentence_length]
        for data in self.raw_data:
            if len(data) > min_sentence_length:  # add all words...? or just min sentence length ones? probably the latter
                raw_counts.extend(data)
        py_data_speeder.set_master_list(raw_counts)
        self.num_negs = num_negs
        self.context_window = window_size

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.raw_data[idx]
        sample = py_data_speeder.get_sample(sample, self.num_negs, self.context_window)  # text, num_negs, context window
        return sample


class InfluenceMedDataset(Dataset):
    """Medical text dataset, adapted from the PyTorch tutorial at:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html."""

    def __init__(self, data_dir, filename, window_size=4, num_negs=15, min_sentence_length=-1):
        """
        Args:
            data_dir (string): directory containing master pkl file
            filename (string): master pkl file with all data separated into sentences
            window_size (int, optional): Window size for contexts
            num_negs (int, optional): number of negatives to sample per input
            min_sentence_length (int, optional): throw out sentences shorter than this
        """
        abstract_fn = os.path.join(data_dir, filename)
        self.raw_data = pickle.load(open(abstract_fn, 'rb'))  # should be a master list of lists of tokens [[tok1, tok2, tok3, tok4], [tok5, tok2, tok1, tok6]]
        self.raw_data = [s for s in self.raw_data if len(s) > min_sentence_length]
        self.num_negs = num_negs
        self.context_window = window_size

    def __len__(self):
        return len(self.raw_data)

    def get_raw_sample(self, idx):
        return self.raw_data[idx]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.raw_data[idx]
        returned_sample = []
        for i, word in enumerate(sample):
            first_context_word_index = max(0, i - self.context_window)
            last_context_word_index = min(i + self.context_window, len(sample))
            for j in range(first_context_word_index, last_context_word_index):
                if i != j:
                    returned_sample.append([word, sample[j]])
        return returned_sample


class TransformerMedDataset(Dataset):
    """Medical text dataset, adapted from the PyTorch tutorial at:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html."""

    def __init__(self, data_dir, filename, block_size=510, min_sentence_length=3):
        """
        Args:
            data_dir (string): directory containing master pkl file
            filename (string): master pkl file with all data separated into sentences
            block_size (int, optional): block size for training
            min_sentence_length (int, optional): throw out sentences shorter than this
        """
        abstract_fn = os.path.join(data_dir, filename)
        raw_data = pickle.load(open(abstract_fn, 'rb'))  # should be a master list of lists of tokens [[tok1, tok2, tok3, tok4], [tok5, tok2, tok1, tok6]]
        raw_counts = []
        raw_data = [s for s in raw_data if len(s) > min_sentence_length]
        for data in raw_data:
            if len(data) > min_sentence_length:  # add all words...? or just min sentence length ones? probably the latter
                raw_counts.extend(data)
        # Based on vocab_builder.py line 42 special_tokens = ...
        self.UNK_token = 0
        self.PAD_token = 1
        self.SOS_token = 2
        self.EOS_token = 3
        self.MASK_token = 4
        self.NUM_token = 5
        self.special_tokens = [0, 1, 2, 3, 4, 5]
        self.raw_data = raw_counts
        self.dataset = []
        for sample_index in range(0, len(raw_counts) - block_size + 1, block_size):  # Truncate in block of block_size
            self.dataset.append([self.SOS_token] + raw_counts[sample_index:sample_index+block_size] + [self.EOS_token])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx])


if __name__ == "__main__":
    from cross_loss_influence.data.scripts.vocab_builder import stringify_token
    from cross_loss_influence.config import PROJECT_HOME, DATA_DIR
    import time
    start_time = time.time()
    # test_med_dataset = CythonMedDataset(data_dir=DATA_DIR,
    #                                     filename='master_numericalized_dataset.pkl',
    #                                     window_size=4,
    #                                     num_negs=15)
    print(f"Full body dataset loads in {time.time() - start_time}")
    start_time = time.time()
    file_med_dataset = MedText(data_dir=DATA_DIR,
                               window_size=4,
                               num_negs=15)
    print(f"File-separated dataset loads in {time.time() - start_time}")

    for i in range(5):
        # print(i)
        # print(test_med_dataset[i])
        sam = file_med_dataset[i]
        print(sam)
        print(sam[0])
        print(sam[1])
        print(sam[2])

    text_dataloader = DataLoader(dataset=file_med_dataset, batch_size=2, shuffle=False)
    for index, batch in enumerate(text_dataloader):
        print(index)
    #     all_words = []
    #     all_contexts = []
    #     all_negatives = []
    #     for b in batch:
    #         all_words.extend(b[0])
    #         all_contexts.extend(b[1])
    #         for a in range(len(b[2][0])):
    #             temp_arr = []
    #             for index in range(len(b[2])):
    #                 temp_arr.append(b[2][index][a])
    #             all_negatives.append(temp_arr)
    #     all_words = torch.tensor(all_words, dtype=torch.int32)
    #     all_contexts = torch.tensor(all_contexts, dtype=torch.int32)
    #     all_negatives = torch.tensor(all_negatives, dtype=torch.int32)
    #

    test_dataloader = DataLoader(dataset=test_med_dataset, batch_size=2, shuffle=False)
    for index, batch in enumerate(test_dataloader):
        print(index)
        print(batch)
        print(stringify_token([batch[0].numpy().tolist(), batch[1].numpy().tolist(), [sam.numpy().tolist() for sam in batch[2]]]))
        if index > 5:
            break
