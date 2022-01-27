from libcpp.vector cimport vector
from libc.stdlib cimport rand, RAND_MAX

cdef list master_list
cdef int master_list_size

def set_master_list(list data_in):
    global master_list
    master_list = data_in
    global master_list_size
    master_list_size = len(data_in)

def sample_from_master():
    global master_list_size
    return master_list[rand()%master_list_size]

def get_sample(list text, int num_negs, int window_size):
    cdef list dataset = []
    # cdef list negatives = []
    cdef list interim_negatives = []
    for i, word in enumerate(text):
        first_context_word_index = max(0, i - window_size)
        last_context_word_index = min(i + window_size, len(text))
        for j in range(first_context_word_index, last_context_word_index):
            if i != j:
                interim_negatives = []
                for _ in range(num_negs):
                    interim_negatives.append(sample_from_master())
                # negatives.append(interim_negatives)
                dataset.append((word, text[j], interim_negatives))
    return dataset

def get_abstract_sample(list abstract, int num_negs, int window_size):
    cdef list dataset = []
    # cdef list negatives = []
    cdef list interim_negatives = []
    for text in abstract:
        for i, word in enumerate(text):
            first_context_word_index = max(0, i - window_size)
            last_context_word_index = min(i + window_size, len(text))
            for j in range(first_context_word_index, last_context_word_index):
                if i != j:
                    interim_negatives = []
                    for _ in range(num_negs):
                        interim_negatives.append(sample_from_master())
                    dataset.append((word, text[j], interim_negatives))
    return dataset


def get_sample_without_negatives(list text, int window_size):
    cdef list dataset = []
    for i, word in enumerate(text):
        first_context_word_index = max(0, i - window_size)
        last_context_word_index = min(i + window_size, len(text))
        for j in range(first_context_word_index, last_context_word_index):
            if i != j:
                dataset.append((word, text[j]))
    return dataset


def get_negatives(int num_negs, int total_sample_count):
    cdef list negatives = []
    cdef list interim_negatives = []
    for _ in range(total_sample_count):
        interim_negatives = []
        for _ in range(num_negs):
            interim_negatives.append(sample_from_master())
        negatives.append(interim_negatives)
    return negatives


# cdef extern from "data_speeder.h":
#     cdef void set_master_list(vector<int> master_list_to_set)
#     cdef vector<int> sample_negatives(int number_to_sample)
#     cdef vector<int> master_list