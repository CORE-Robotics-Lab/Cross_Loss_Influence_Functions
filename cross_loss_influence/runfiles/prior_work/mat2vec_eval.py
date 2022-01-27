# Created by Andrew Silva
"""
This is a copy of the mat2vec training file, which I will be making my own adjustments to over time.
"""

from mat2vec.training.helpers.utils import compute_epoch_accuracies
import logging
import os
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from cross_loss_influence.config import PROJECT_NAME, MODEL_SAVE_DIR, DATA_DIR
import glob
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

import re


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after every epoch."""

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, m):
        output_path = "{}_epoch{}.model".format(self.path_prefix, self.epoch)
        print("Save model to {}.".format(output_path))
        m.save(output_path)
        self.epoch += 1


def compute_epoch_accuracies_returning(root, prefix, analogy_file):
    """Modified from mat2vec code"""
    filenames = glob.glob(os.path.join(root, prefix+"_epoch*.model"))
    nr_epochs = len(filenames)
    accuracies = dict()
    for filename in filenames:
        epoch = int(re.search("\d+\.model", filename).group()[:-6])
        m = Word2Vec.load(filename)
        sections = m.wv.accuracy(analogy_file)
        for sec in sections:
            if sec["section"] not in accuracies:
                accuracies[sec["section"]] = [(0, 0, 0)] * 50  # This should be fixed? to be like
            correct, incorrect = len(sec["correct"]), len(sec["incorrect"])
            if incorrect > 0:
                accuracy = correct / (correct + incorrect)
            else:
                accuracy = 0
            accuracies[sec["section"]][epoch] = (correct, incorrect, accuracy)
    return accuracies


def get_accuracy_for_one_model(window_size, negatives, size, base_name, analogy_file):
    model_name = f"{base_name}window-{window_size}_negative-{negatives}_size-{size}"
    return compute_epoch_accuracies_returning(os.path.join(MODEL_SAVE_DIR), model_name, analogy_file)


def get_and_plot_accuracy_for_one_model_single_input(truple_in):
    """
    Multiprocessing is annoying and only takes a single input. so here i am taking a tuple and hard coding some things
    for ease/speed
    :param truple_in:
    :return:
    """
    window_size = truple_in[0]
    base_name = truple_in[1]
    analogy_file = truple_in[2]
    negatives = 10
    size = 300
    model_name = f"{base_name}window-{window_size}_negative-{negatives}_size-{size}"
    accuracy = compute_epoch_accuracies_returning(os.path.join(MODEL_SAVE_DIR), model_name, analogy_file)
    plot_accuracies(accuracy, model_name)


def plot_accuracies(accuracies_in, model_name):

    fig, ax = plt.subplots(figsize=(12, 12))
    markers = ['o', 's', 'p', 'h', '<', '>', '^', 'x']
    bad_analogies = ['crystal structures (zincblende, wurtzite, rutile, rocksalt, etc.)',
                     'metals and their oxides (most common)',
                     'crystal symmetry (cubic, hexagonal, tetragonal, etc.)',
                     'elemental crystal structures (bcc, fcc, hcp, dhcp)',
                     'magnetic properties']
    grammar_scores = np.zeros((1, 50))
    for key, val in accuracies_in.items():
        if key in bad_analogies:
            continue  # Skip analogies that don't work with my data
        if 'gram' in key:
            grammar_scores += [acc[-1] for acc in val]  # change total to reflect grammar total
        if 'total' in key:
            grammar_scores /= 7  # normalize for 7 grammar categories
            val = [[0, score] for score in grammar_scores[0]]  # reset total to be the new grammar total
        ax.plot(np.arange(len(val)), [acc[-1] for acc in val], marker=random.choice(markers), color=np.random.rand(3),
                label=key)
        with open(os.path.join(DATA_DIR, 'txt', model_name+'.txt'), 'a') as f:
            f.write(f"Max of {key} is {max([acc[-1] for acc in val])} for plot on {model_name} \n")

    legend = ax.legend(loc='upper left', ncol=3, shadow=True, fontsize='x-large', bbox_to_anchor=(-0.02, 1.15))
    plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(DATA_DIR, "fig", model_name+'.png'))
    # plt.show()


def plot_accuracies_for_all_models(analogy_file, base_name='immunology'):
    for window_size in range(2, 11):
        for negatives in [10]:  # [5, 10, 15, 20]
            for size in [50, 100]:  # [50, 100, 200, 300]
                model_acc = get_accuracy_for_one_model(window_size, negatives, size, base_name, analogy_file)
                model_name = f"{base_name}window-{window_size}_negative-{negatives}_size-{size}"
                plot_accuracies(model_acc, model_name)


def mp_accs_all_models(analogy_file, base_name='immunology'):
    # Making the assumption that negatives are hard-coded down the line. sorry
    # Making the assumption that size is hard coded down the line. sorry^2
    from multiprocessing.pool import Pool
    me_pool = Pool()
    inputs = []
    for i in range(2, 11):
        inputs.append((i, base_name, analogy_file))
    me_pool.map(get_and_plot_accuracy_for_one_model_single_input, inputs)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", help="Name for saving the model (in the models folder).", default="immunology_model_basic")
    parser.add_argument("--recompute", action="store_true", help="Recompute model accuracies?")
    parser.add_argument("--analogies", help="filename in the /data/ directory for analogies", default="analogies.txt")
    args = parser.parse_args()
    args.recompute = True
    if args.recompute:
        analogy_file = os.path.join(DATA_DIR, args.analogies)
        # Plot the accuracies in the tmp folder.
        mp_accs_all_models(analogy_file, base_name='sci_fi')
    else:
        all_accuracies = pickle.load(open(f'models/{args.model_name}_accuracies.pkl', 'rb'))
