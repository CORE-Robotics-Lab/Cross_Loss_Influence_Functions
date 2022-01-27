import torchtext
from cross_loss_influence.config import PROJECT_HOME, DATA_DIR, MODEL_SAVE_DIR, PROJECT_NAME
# from cross_loss_influence.data.scripts.mat2vec_tokenize_test import preprocess_data_for_vocab, \
# preprocess_data_for_model, preprocess_single_abstract_for_model
# import pyximport; pyximport.install()
# import mat2vec_tokenize_test
from cross_loss_influence.data.scripts import mat2vec_tokenize_test
from torchtext.data import Field
import pickle
import os


def splitter(string_in):
    return string_in.split()


def load_raw_abstracts(data_fn='pure_text_dataset.pkl'):
    """
    Loads raw abstract data from the config.json dir and the immunology fn
    :return:
    """

    filename_abstracts = os.path.join(DATA_DIR, data_fn)  # TODO Change name for pre-2015
    abstracts = pickle.load(open(filename_abstracts, 'rb'))
    return abstracts


def build_vocab(data_fn='pure_text_dataset.pkl'):
    """
    Uses config.json to find data dir and currently just gets immunology data
    Creates a torchtext vocab and saves the string to index dict and the index to string array
    See mat2vec_tokenize_test.preprocess_data for the real tokenization stuff
    :param data_fn: filename for the text dataset pickle
    """
    # TODO: PHRASING?

    text_field = Field(init_token='<SOS>', eos_token='<EOS>', preprocessing=splitter, postprocessing=None,
                       lower=False, tokenize=None, include_lengths=False, pad_token='<PAD>', unk_token='<UNK>',
                       stop_words=None, is_target=False)

    abstracts = load_raw_abstracts(data_fn=data_fn)
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>', '<NUM>']
    processor = mat2vec_tokenize_test.SciFiTextProcessor()
    all_tokens = mat2vec_tokenize_test.preprocess_data_for_vocab(abstracts, processor)
    field_to_pass = [("Text", text_field)]
    dataset = torchtext.data.Dataset(all_tokens, field_to_pass)
    text_field.build_vocab(all_tokens,
                           max_size=None,
                           min_freq=5,
                           specials=special_tokens,
                           vectors=None,
                           unk_init=None,
                           vectors_cache=None,
                           specials_first=True)
    pickle.dump(text_field.vocab.stoi, open(os.path.join(DATA_DIR, data_fn.split('.')[0]+'_stoi.pkl'), 'wb'))
    pickle.dump(text_field.vocab.itos, open(os.path.join(DATA_DIR, data_fn.split('.')[0]+'_itos.pkl'), 'wb'))
    # TODO: FIX
    #### THROWING ENCODING ERRORS??
    # with open(os.path.join(DATA_DIR, 'stoi.txt'), 'w') as stoi_file:
    #     for key, val in text_field.vocab.stoi.items():
    #         stoi_file.write(key + ': ' + str(val) + '\n')
    # with open(os.path.join(DATA_DIR, 'itos.txt'), 'w') as itos_file:
    #     for entry in text_field.vocab.itos:
    #         itos_file.write(entry + '\n')


def stringify_token(data_in):
    """
    Take in a token or list of tokens and return the string
    :param data_in: token list of tokens to tokenize
    :return: string
    """
    strings_at_indices = pickle.load(open(os.path.join(DATA_DIR, 'itos.pkl'), 'rb'))  # TODO Change name for pre-2015
    if isinstance(data_in, list) or isinstance(data_in, tuple):
        return " ".join([stringify_token(tok) for tok in data_in])
    return strings_at_indices[data_in]


def tokenize_string(data_in, string_indices=False):
    """
    Take in a new string and return a set of indices for the string tokens
    :param data_in: string or list of tokens to tokenize
    :param string_indices: indices of strings for numericalization. or True for a global (sorry...)
    :return: list of integers
    """
    if not string_indices:
        string_indices = pickle.load(open(os.path.join(DATA_DIR, 'stoi.pkl'), 'rb'))  # TODO Change name for pre-2015
    else:
        string_indices = STRING_INDICES
    current_sentence = []
    if isinstance(data_in, str):
        data_in = data_in.split()
    if isinstance(data_in, int):
        data_in = [data_in]
    for token in data_in:
        try:
            current_sentence.append(string_indices[token])
        except KeyError as e:
            # print(e)
            current_sentence.append(string_indices['<UNK>'])
    return current_sentence


def tokenize_dataset(preprocess=False):
    if preprocess:
        abstracts = load_raw_abstracts()
        print("Data loaded...")
        all_tokens = mat2vec_tokenize_test.preprocess_data_for_model(abstracts)
        print("Data preprocessed")
    else:
        all_tokens = pickle.load(open(os.path.join(PROJECT_HOME, 'data', 'objects', 'model_preprocessed_abstracts.pkl'), 'rb'))
    master_list = []
    for abstract_index, abstract in enumerate(all_tokens):
        sentences = break_into_sentences(abstract)
        # print(f"On a new abstract of length {len(sentences)} at index {abstract_index}")
        for sentence in sentences:
            sentence_tokens = tokenize_string(sentence)
            master_list.append(sentence_tokens)
    print("Saving numericalized dataset")
    pickle.dump(master_list, open('model_numericalized_dataset.pkl', 'wb'))
    print("Done")
    return master_list


def break_into_sentences(data_in, exclude_toks=['<SOS>']):
    """
    Break a list of tokens into a list of lists, removing EOS tokens and having each list be a standalone sentence
    :param data_in: long list of sentences or string of tokens
    :param exclude_toks: tokens to skip (like SOS. leave in EOS because we need it to know when to quit)
    :return: list of lists of tokens, each list is a sentence (master list is a collection of sentences)
    """
    if isinstance(data_in, str):
        data_in = data_in.split()
    data_collector = []
    curr_sent = []
    for token in data_in:
        if token in exclude_toks:
            continue
        if token == '<EOS>':
            data_collector.append(curr_sent)
            curr_sent = []
            continue
        curr_sent.append(token)
    if len(curr_sent) > 0:
        data_collector.append(curr_sent)
    return data_collector


def replace_with_most_common(dataset, n_to_keep):
    """
    Take in a list of lists of tokens and find the most frequent n_to_keep, and replace all less frequent with UNK
    Not necessary yet because prior work dataset is 2x ours (theirs is 500k)
    :param dataset:
    :param n_to_keep:
    :return:
    """
    pass


def save_vocab_and_model_tokens():
    """
    If I need to test some numericalization or something, I will need to load/preprocess a bunch. so let's just do it
    once and save the results
    :return: Nothing
    """
    abstracts = load_raw_abstracts()
    print("Data loaded...")
    print("Model cleaning beginning...")
    all_tokens = mat2vec_tokenize_test.preprocess_data_for_model(abstracts)
    pickle.dump(all_tokens, open('model_preprocessed_abstracts.pkl', 'wb'))
    print("Model Data Dumped")
    print("Vocab cleaning beginning...")
    vocab_tokens = mat2vec_tokenize_test.preprocess_data_for_vocab(abstracts)
    pickle.dump(vocab_tokens, open("vocab_preprocessed_abstracts.pkl", 'wb'))
    print("Vocab Data Dumped")


def save_text_and_nums(tuple_in):
    """
    Takes an abstract and id number and saves the abstract text to a folder and the int values to a folder
    :param tuple_in: tuple of form (abstract, int_tokens)
    :return: True on success, False on error out
    """
    data_dir = os.path.join(DATA_DIR, 'pubmed_abstracts')
    txt_dir = 'text_abstracts/'
    num_dir = 'numericalized_abstracts/'
    abstract_in = tuple_in[0]
    id_number = tuple_in[1]
    str_tokens = mat2vec_tokenize_test.preprocess_single_abstract_for_model(abstract_in)
    if str_tokens:
        abs_num_tokens = []
        sentences = break_into_sentences(str_tokens)
        for sentence in sentences:
            sentence_tokens = tokenize_string(sentence, string_indices=True)
            abs_num_tokens.append(sentence_tokens)
        abs_fn = open(os.path.join(data_dir, txt_dir, str(id_number) + '.txt'), 'w', encoding='utf-8')
        abs_fn.writelines(abstract_in)
        abs_fn.close()
        pickle.dump(abs_num_tokens, open(os.path.join(data_dir, num_dir, str(id_number) + '.pkl'), 'wb'))
        return True
    else:
        return False


def save_all_text_abstracts():
    abstracts = load_raw_abstracts()
    counter = 0
    for abstract in abstracts:
        if save_text_and_nums((abstract, counter)):
            counter += 1


def multiprocessing_save_all_text_nums():
    all_data = load_raw_abstracts()
    # all_data = pickle.load(open(os.path.join(PROJECT_HOME, 'data', 'objects', 'test_dataset.pkl'), 'rb'))
    all_ids = range(len(all_data))

    from multiprocessing import Pool

    pool = Pool()

    pool.map(save_text_and_nums, zip(all_data, all_ids))
    print("Done")


def get_model_text(abstract_in):
    try:
        str_tokens = TEXT_PROCESSOR.process_for_model(abstract_in, exclude_punct=True)
    except TypeError as e:
        print(e)
        return False
    return str_tokens


def multiprocessing_write_abstracts_for_gensim():
    """
    Takes an abstract and id number and saves the abstract text to a folder and the int values to a folder
    :param tuple_in: tuple of form (abstract, int_tokens)
    :return: True on success, False on error out
    """
    from multiprocessing import Pool

    pool = Pool()
    all_data = load_raw_abstracts()
    # all_data = pickle.load(open(os.path.join(PROJECT_HOME, 'data', 'objects', 'test_dataset.pkl'), 'rb'))
    all_data = [x for x in all_data if x is not None]

    results = pool.map(get_model_text, all_data)
    with open('immunology_gensim_text.txt', 'w') as f:
        for abstract in results:
            if not abstract:
                continue
            new_abstract = []
            for token in abstract:
                if token == '<SOS>':
                    token = '\n'
                elif token == '<EOS>':
                    continue
                token = token + ' '
                new_abstract.append(token)
            f.writelines(new_abstract)
    return True


def create_master_pkl():
    master_list = []
    for pkl in os.listdir(os.path.join(DATA_DIR, 'pubmed_abstracts', 'numericalized_abstracts')):
        abstract_fn = os.path.join(DATA_DIR, 'pubmed_abstracts', 'numericalized_abstracts', pkl)
        sentences = pickle.load(open(abstract_fn, 'rb'))
        master_list.extend(sentences)
    master_dataset_fn = os.path.join(DATA_DIR, 'master_numericalized_dataset.pkl')
    pickle.dump(master_list, open(master_dataset_fn, 'wb'))


def get_nums(abstract_in):
    """
    Takes an abstract and returns tokens
    :param abstract_in: abstract
    :return: numbers/tokens for words
    """
    try:
        str_tokens = TEXT_PROCESSOR.process_for_model(abstract_in, exclude_punct=True, make_phrases=True)
    except TypeError as e:
        print(e)
        return False
    if str_tokens:
        abs_num_tokens = []
        # sentences = break_into_sentences(str_tokens)
        # for sentence in sentences:
        sentence_tokens = tokenize_string(str_tokens, string_indices=True)  # tokenize_string(sentences, True)
        abs_num_tokens.append(sentence_tokens)
        return abs_num_tokens
    else:
        return False


def multiprocessing_create_master_pkl(data_fn='pure_text_dataset.pkl'):
    all_data = load_raw_abstracts(data_fn=data_fn)
    # all_data = pickle.load(open(os.path.join(PROJECT_HOME, 'data', 'objects', 'test_dataset.pkl'), 'rb'))
    all_data = [x for x in all_data if x is not None]
    from multiprocessing import Pool
    #
    pool = Pool()
    #
    master_dataset = pool.map(get_nums, all_data)
    all_numbers = []
    for sample in master_dataset:
        if not sample:
            continue
        all_numbers.extend(sample)
    master_dataset_fn = os.path.join(DATA_DIR, data_fn.split('.')[0]+'_numericalized_dataset.pkl')
    pickle.dump(all_numbers, open(master_dataset_fn, 'wb'))
    print("Done")


if __name__ == "__main__":

    data_fn = 'biased_data.pkl'
    #### To build a new vocabulary, build_vocab (requires changed up replacement dicts/csvs) ####
    # build_vocab(data_fn)
    #### to save all abstracts as text and tokens, multiprocessing_save_all_text_nums (and load STRING_INDICES)
    STRING_INDICES = pickle.load(
        open(os.path.join(DATA_DIR, data_fn.split('.')[0]+'_stoi.pkl'), 'rb'))
    # TEXT_PROCESSOR = mat2vec_tokenize_test.SciFiTextProcessor(BERT=False,
    #                                                             phraser_path=os.path.join(MODEL_SAVE_DIR, 'immunologywindow-8_negative-20_size-200_phraser.pkl'))  # Global required
    TEXT_PROCESSOR = mat2vec_tokenize_test.SciFiTextProcessor()
    data_fn = 'neutral_data.pkl'
    multiprocessing_create_master_pkl(data_fn)

    #### For Gensim:
    # multiprocessing_write_abstracts_for_gensim()