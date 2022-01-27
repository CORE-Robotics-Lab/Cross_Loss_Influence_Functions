# Created by Andrew Silva
"""
This is basically weat.py but for my own personal tests
"""
from cross_loss_influence.helpers import weat
from cross_loss_influence.helpers.sklearn_cluster_embeddings import get_embeddings
import numpy as np


def similarity_diff_sing(word, attrs_A):
    cos_attr_one = []
    for a_A in attrs_A:
        cos_attr_one.append(weat.cos_diff(word, a_A))
    return np.mean(cos_attr_one)

if __name__ == "__main__":
    window_size = 10
    negatives=10
    checkpoint='60'
    embeddings, all_keys = get_embeddings(model_fn=f'DENSE_neutral_window-{window_size}_negatives-{negatives}_{checkpoint}_checkpoint.pth.tar',
                                          vocab_fn='biased_data_stoi.pkl')

    X = ['josh', 'alan', 'ryan', 'andrew', 'jack', 'greg', 'amanda', 'katie', 'nancy', 'ellen']
    Y = ['theo', 'jerome', 'leroy', 'lamar', 'lionel', 'malik', 'tyrone', 'ebony', 'jasmine', 'tia', ]
    A = ['jew', 'jewish', 'islam', 'religion', 'islamic', 'muslim']

    x_embed = [embeddings[all_keys.index(x)] for x in X]
    y_embed = [embeddings[all_keys.index(x)] for x in Y]
    a_embed = [embeddings[all_keys.index(x)] for x in A]
    mean_one = []
    mean_two = []
    std_all = []
    for x, y in zip(x_embed, y_embed):
        m1 = similarity_diff_sing(x, a_embed)
        m2 = similarity_diff_sing(y, a_embed)
        mean_one.append(m1)
        mean_two.append(m2)
        std_all.append(m1)
        std_all.append(m2)
    print(f"Average European similarity to religion: {np.mean(mean_one)}")
    print(f"Average African similarity to religion: {np.mean(mean_two)}")
    effect_size = (np.mean(mean_one) - np.mean(mean_two)) / np.std(std_all)
    print(f"Effect = {effect_size}")

    window_size = 3
    negatives=5
    checkpoint='last'
    embeddings, all_keys = get_embeddings(model_fn=f'DENSE_scifi_window-{window_size}_negatives-{negatives}_{checkpoint}_checkpoint.pth.tar',
                                          vocab_fn='all_scripts_stoi.pkl')

    X = ['anakin', 'yoda', 'kanan', 'ezra', 'ahsoka']
    Y = ['vader', 'sidious', 'dooku', 'maul', 'inquisitor']
    A = ['sith', 'evil', 'anger', 'hate', 'fear']
    B = ['jedi', 'good', 'defense', 'knowledge', 'peace']

    x_embed = [embeddings[all_keys.index(x)] for x in X]
    y_embed = [embeddings[all_keys.index(x)] for x in Y]
    a_embed = [embeddings[all_keys.index(x)] for x in A]
    b_embed = [embeddings[all_keys.index(x)] for x in B]
    mean_one = []
    mean_two = []
    std_all = []
    for x, y in zip(x_embed, y_embed):
        m1 = weat.similarity_diff(x, a_embed, b_embed)
        m2 = weat.similarity_diff(y, a_embed, b_embed)
        mean_one.append(m1)
        mean_two.append(m2)
        std_all.append(m1)
        std_all.append(m2)
    print(f"Average Anakin similarity to Sith-Jedi: {np.mean(mean_one)}")
    print(f"Average Vader similarity to Sith-Jedi: {np.mean(mean_two)}")
    print(np.std(std_all))
    mean_one = []
    mean_two = []
    std_all = []
    for x, y in zip(x_embed, y_embed):
        m1 = similarity_diff_sing(x, a_embed)
        m2 = similarity_diff_sing(y, a_embed)
        mean_one.append(m1)
        mean_two.append(m2)
        std_all.append(m1)
        std_all.append(m2)
    print(f"Average Anakin similarity to Sith: {np.mean(mean_one)}")
    print(f"Average Vader similarity to Sith: {np.mean(mean_two)}")
    mean_one = []
    mean_two = []
    std_all = []
    for x, y in zip(x_embed, y_embed):
        m1 = similarity_diff_sing(x, b_embed)
        m2 = similarity_diff_sing(y, b_embed)
        mean_one.append(m1)
        mean_two.append(m2)
        std_all.append(m1)
        std_all.append(m2)
    print(f"Average Anakin similarity to Jedi: {np.mean(mean_one)}")
    print(f"Average Vader similarity to Jedi: {np.mean(mean_two)}")
    test_stat = weat.test_statistic(x_embed, y_embed, a_embed, b_embed)
    effect = weat.effect_size(x_embed, y_embed, a_embed, b_embed)
    p_val = weat.p_value(x_embed, y_embed, a_embed, b_embed)
    effect_size = (np.mean(mean_one) - np.mean(mean_two)) / np.std(std_all)
    print(f"Test statistic = {test_stat} || Effect = {effect} || P = {p_val}")
