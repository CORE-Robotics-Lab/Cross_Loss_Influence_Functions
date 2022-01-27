# Created by Andrew Silva
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from cross_loss_influence.helpers.sklearn_cluster_embeddings import get_embeddings
from cross_loss_influence.config import DATA_DIR
import torch
import os
from sklearn.neighbors import KDTree


def cos_diff(x, y):
    # return np.dot(x, y) / (np.sqrt(x ** 2) * np.sqrt(y ** 2))
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))


def weat(embeddings, keys, test='career', just_return=False):
    if test == 'career':
        X = ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill']
        Y = ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna']
        A = ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career']
        B = ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']
    elif test == 'math':
        A = ['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition']
        B = ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture']
        X = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
        Y = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
        # X = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son', 'father', 'uncle', 'grandfather']
        # Y = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter', 'mother', 'aunt', 'grandmother']
        # X = ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him']
        # Y = ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'her', 'hers']
    elif test == 'science':
        A = ['science', 'technology', 'physics', 'chemistry', 'einstein', 'nasa', 'experiment', 'astronomy']
        B = ['poetry', 'art', 'shakespeare', 'dance', 'literature', 'novel', 'symphony', 'drama']
        X = ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him']
        Y = ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'her', 'hers']
    elif test == 'race':
        A = ['freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond',
             'gentle', 'honest', 'lucky', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family',
             'happy', 'laughter', 'vacation']
        B = ['crash', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink',
             'assault', 'disaster', 'hatred', 'tragedy', 'bomb', 'divorce', 'jail', 'poverty', 'ugly',
             'cancer', 'evil', 'kill', 'rotten', 'vomit']
        X = ['josh', 'alan', 'ryan', 'andrew', 'jack', 'greg', 'amanda', 'katie', 'nancy', 'ellen']
        Y = ['theo', 'jerome', 'leroy', 'lamar', 'lionel', 'malik', 'tyrone', 'ebony', 'jasmine', 'tia', ]
    x_embed = [embeddings[keys.index(x)] for x in X]
    y_embed = [embeddings[keys.index(x)] for x in Y]
    a_embed = [embeddings[keys.index(x)] for x in A]
    b_embed = [embeddings[keys.index(x)] for x in B]
    test_stat = test_statistic(x_embed, y_embed, a_embed, b_embed)
    effect = effect_size(x_embed, y_embed, a_embed, b_embed)
    p_val = p_value(x_embed, y_embed, a_embed, b_embed)
    if not just_return:
        print(f"Test statistic = {test_stat} || Effect = {effect} || P = {p_val}")
    return test_stat, effect, p_val


def load_analogy_sets(keys, filename_in='grammar_analogies.txt',
                      data_dir=DATA_DIR,
                      ):
    all_data = None
    with open(os.path.join(data_dir, filename_in), 'r') as f:
        all_data = f.readlines()
    all_data = [x.strip('\n') for x in all_data]
    sets = {}
    current_set = None
    for sample in all_data:
        if ':' in sample:
            sets[sample] = []
            current_set = sample
        else:
            good_analogy = True
            for word in sample.split(' '):
                if word not in keys:
                    good_analogy = False
            if good_analogy:
                sets[current_set].append(sample.split())
    return sets


def run_analogy_test(all_pairs_in, embeddings, keys):
    # embeddings = np.array([embed/np.linalg.norm(embed) for embed in embeddings])
    tree = KDTree(embeddings)
    correct = 0
    total = 0
    for paired_pair in all_pairs_in:
        a = embeddings[keys.index(paired_pair[0])]
        b = embeddings[keys.index(paired_pair[2])]
        a_star = embeddings[keys.index(paired_pair[1])]
        # nearest_neighbor, score, score_list = find_answer_cos(a, a_star, b, embeddings, keys)
        # nearest_neighbor, score_list = find_answer_dot(a, a_star, b, embeddings, keys)
        nearest_neighbor, score_list = find_answer_kd_tree(a, a_star, b, keys, tree)

        n_ind = 0
        while nearest_neighbor in [paired_pair[0], paired_pair[1], paired_pair[2]]:
            n_ind += 1
            nearest_neighbor = score_list[n_ind]
        if nearest_neighbor == paired_pair[3]:
            print(paired_pair)
            correct += 1
        total += 1
    return float(correct)/float(total) * 100


def find_answer_kd_tree(in_1, in_2, out_1, keys, tree):
    nn_vec = in_2 - in_1 + out_1
    dist, ind = tree.query([nn_vec], k=5)
    nearest_words = [keys[word_ind] for word_ind in ind[0]]
    return nearest_words[0], nearest_words


def find_answer_dot(in_1, in_2, out_1, embeddings, keys):
    embed = in_2 - in_1 + out_1
    nearest = np.argsort(np.matmul(embed, embeddings.transpose()))[-10:][::-1]
    return keys[nearest[0]], [keys[x] for x in nearest]


def find_answer_cos(in_1, in_2, out_1, embedding_list, keys, eps=1e-3):
    best_yet = -9999
    best_word = -1
    scores = []
    for index, word in enumerate(keys):
        embedding = embedding_list[index]
        embed_score = cos_diff(embedding, out_1)*cos_diff(embedding, in_2)
        embed_score /= (cos_diff(embedding, in_1) + eps)
        scores.append([word, embed_score])
        if embed_score > best_yet:
            best_yet = embed_score
            best_word = word
    scores = sorted(scores, key=lambda x: -x[-1])
    scores = [s[0] for s in scores]
    return best_word, best_yet, scores


def run_all_analogies(analogy_sets, embeddings, keys):
    for key, value in analogy_sets.items():
        print(f"Running analogy test {key}")
        percent_correct = run_analogy_test(value, embeddings, keys)
        print(f"Got {percent_correct}% correct")

if __name__ == "__main__":
    window_size = 10
    negatives = 10
    # checkpoint = '60'
    checkpoint = '60_math-N-1000-K-1000-harm--undone'
    # 'DENSE_biased_window-10_negatives-10__checkpoint.pth'
    biased_cluster_data, all_keys = get_embeddings(model_fn=f'DENSE_biased_window-{window_size}_negatives-{negatives}_{checkpoint}_checkpoint.pth.tar',
                                                   vocab_fn='biased_data_stoi.pkl')
    all_analogies = load_analogy_sets(all_keys)
    run_all_analogies(all_analogies, biased_cluster_data, all_keys)