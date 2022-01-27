from cross_loss_influence.helpers.bolukbasi_prior_work import we
import json
import numpy as np
from cross_loss_influence.config import MODEL_SAVE_DIR
import sys
import os
from cross_loss_influence.helpers.sklearn_cluster_embeddings import get_embeddings
if sys.version_info[0] < 3:
    import io
    open = io.open
"""
Hard-debias embedding

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""


def debias(E, gender_specific_words, definitional, equalize):
    gender_direction = we.doPCA(definitional, E).components_[0]
    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    E.normalize()
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    print(candidates)
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = we.drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()


def extract_txt_embeddings(txt_fn):
    """
    Read a text file produced by this script and extract keys and embeddings
    Args:
        txt_fn: full path and filename of debiased script

    Returns: embeddings, words
    """
    words = []
    embeddings = []
    with open(txt_fn, 'r') as f:
        for line in f.readlines():
            line = line.split()
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            words.append(word)
            embeddings.append(embedding)
    embeddings = np.array(embeddings)
    return embeddings, words


if __name__ == "__main__":

    base = 'biased'
    gender = False
    for model_fn in os.listdir(MODEL_SAVE_DIR):
        if not model_fn.endswith('.tar'):
            continue
        if gender:
            def_fn = 'definitional_pairs.json'
            gend_fn = 'gender_specific_seed.json'
            eq_fn = 'equalize_pairs.json'
            tag = 'gender'
        else:
            def_fn = 'd_race.json'
            gend_fn = 'r_race.json'
            eq_fn = 'e_race.json'
            tag = 'race'

        deb_fn = f'bolukbasi_original_{model_fn}_debiased-{tag}.txt'

        with open(def_fn, "r") as f:
            defs = json.load(f)
        print("definitional", defs)

        with open(eq_fn, "r") as f:
            equalize_pairs = json.load(f)

        with open(gend_fn, "r") as f:
            gender_specific_words = json.load(f)
        print(f"{tag} specific {len(gender_specific_words)} {gender_specific_words[:10]}")

        biased_cluster_data, all_keys = get_embeddings(
            model_fn=model_fn,
            model_dir='/home/user/models/biased_models/',
            vocab_fn='biased_data_stoi.pkl',
            vocab_dir='/home/user/data/biased_data/')
        E = we.WordEmbedding(fname='given', given_data=[biased_cluster_data, all_keys])

        print("Debiasing...")
        debias(E, gender_specific_words, defs, equalize_pairs)

        print("Saving to file...")
        E.save('/home/user/data/biased_data/'+deb_fn)

        print("\n\nDone!\n")