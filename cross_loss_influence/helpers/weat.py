# Created by Andrew Silva
import numpy as np
from cross_loss_influence.helpers.sklearn_cluster_embeddings import get_embeddings
from cross_loss_influence.helpers.bolukbasi_prior_work.prior_pca_debiasing import extract_txt_embeddings
import copy
import os
from cross_loss_influence.config import DATA_DIR, MODEL_SAVE_DIR, PROJECT_NAME
import argparse


def test_statistic(targets_one, targets_two, attribute_one, attribute_two):
    sum_target_one = 0
    sum_target_two = 0
    for x in targets_one:
        sum_target_one += similarity_diff(x, attribute_one, attribute_two)
    for y in targets_two:
        sum_target_two += similarity_diff(y, attribute_one, attribute_two)
    return sum_target_one - sum_target_two


def effect_size(targets_one, targets_two, attribute_one, attribute_two):
    mean_one = []
    mean_two = []
    std_all = []
    for x, y in zip(targets_one, targets_two):
        m1 = similarity_diff(x, attribute_one, attribute_two)
        m2 = similarity_diff(y, attribute_one, attribute_two)
        mean_one.append(m1)
        mean_two.append(m2)
        std_all.append(m1)
        std_all.append(m2)
    return (np.mean(mean_one) - np.mean(mean_two)) / np.std(std_all)


def p_value(targets_one, targets_two, attribute_one, attribute_two):
    overall_stat = test_statistic(targets_one, targets_two, attribute_one, attribute_two)

    all_indices = []
    for s in range(len(targets_one)):
        for x in range(s, len(targets_one)):
            base_array = [y for y in range(s, x)]
            final_array = [z for z in range(x, len(targets_one))]
            for arr in final_array:
                appender = copy.deepcopy(base_array)
                appender.append(arr)
                if appender not in all_indices:
                    all_indices.append(appender)

    partial_stats = []
    for subset in all_indices:
        x_i = [targets_one[ind] for ind in subset]
        for subset_two in all_indices:
            if len(subset_two) == len(subset):
                y_i = [targets_two[ind] for ind in subset_two]
                partial_stats.append(test_statistic(x_i, y_i, attribute_one, attribute_two))

    partial_stats = np.array(partial_stats)

    return np.sum(partial_stats > overall_stat) / len(partial_stats)


def similarity_diff(word, attrs_A, attrs_B):
    cos_attr_one = []
    cos_attr_two = []
    for a_A, a_B in zip(attrs_A, attrs_B):
        cos_attr_one.append(cos_diff(word, a_A))
        cos_attr_two.append(cos_diff(word, a_B))
    return np.mean(cos_attr_one) - np.mean(cos_attr_two)


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


def sweep_weat_n_k(model_base, test_name, N=[5, 10, 100, 1000], K=[5, 10, 100, 1000]):
    for h in ['harm-', 'help-', 'both']:
        for n in N:
            for k in K:
                model_name = model_base + '_' + test_name + f'-N-{n}-K-{k}-{h}-undone'
                model_fn = f'{model_name}_checkpoint.pth.tar'
                if not os.path.exists(os.path.join(MODEL_SAVE_DIR, PROJECT_NAME, 'checkpoints', model_fn)):
                    print(f"{model_fn} does not exist -- Skipping...")
                    continue
                biased_cluster_data, all_keys = get_embeddings(
                    model_fn=model_fn,
                    vocab_fn='biased_data_stoi.pkl')

                print(model_name)
                print(f"For biased data, WEAT on career:")
                c_test_stat, c_effect, c_p_val = weat(biased_cluster_data, all_keys, 'career')
                print()

                print(f"For biased data, WEAT on math:")
                m_test_stat, m_effect, m_p_val = weat(biased_cluster_data, all_keys, 'math')
                print()

                print(f"For biased data, WEAT on science:")
                s_test_stat, s_effect, s_p_val = weat(biased_cluster_data, all_keys, 'science')
                print()

                print(f"For biased data, WEAT on race:")
                r_test_stat, r_effect, r_p_val = weat(biased_cluster_data, all_keys, 'race')
                test_results = {
                    'career': [c_test_stat, c_effect, c_p_val],
                    'math': [m_test_stat, m_effect, m_p_val],
                    'science': [s_test_stat, s_effect, s_p_val],
                    'race': [r_test_stat, r_effect, r_p_val]
                }
                with open(os.path.join(DATA_DIR, model_name+'.txt'), 'w') as f:
                    for key, val in test_results.items():
                        f.write(f"For biased data, WEAT on {key}: Test stat: {val[0]}, effect: {val[1]}, p-val: {val[2]}")
                        f.write('\n')


def prior_work_weat(debiased_fn):
    embeddings, all_keys = extract_txt_embeddings(debiased_fn)
    # all_weats_for_embeds_keys(embeddings, all_keys)
    if 'career' in debiased_fn:
        c_test_stat, c_effect, c_p_val = weat(embeddings, all_keys, 'career')
    elif 'math' in debiased_fn:
        c_test_stat, c_effect, c_p_val = weat(embeddings, all_keys, 'math')
    elif 'science' in debiased_fn:
        c_test_stat, c_effect, c_p_val = weat(embeddings, all_keys, 'science')
    elif 'race' in debiased_fn:
        pass
    else:
        all_weats_for_embeds_keys(embeddings, all_keys)


def all_weats_one_model(model_fn):
    embeddings, all_keys = get_embeddings(model_fn=model_fn,
                                          model_dir=MODEL_SAVE_DIR,
                                          vocab_fn='biased_data_stoi.pkl')
    # all_weats_for_embeds_keys(embeddings, all_keys)
    if 'career' in model_fn:
        c_test_stat, c_effect, c_p_val = weat(embeddings, all_keys, 'career')
    elif 'math' in model_fn:
        c_test_stat, c_effect, c_p_val = weat(embeddings, all_keys, 'math')
    elif 'science' in model_fn:
        c_test_stat, c_effect, c_p_val = weat(embeddings, all_keys, 'science')
    elif 'race' in model_fn:
        c_test_stat, c_effect, c_p_val = weat(embeddings, all_keys, 'race')
    else:
        all_weats_for_embeds_keys(embeddings, all_keys)


def all_weats_for_embeds_keys(embeddings, all_keys):
    print(f"For biased data, WEAT on career:")
    c_test_stat, c_effect, c_p_val = weat(embeddings, all_keys, 'career')
    print()

    print(f"For biased data, WEAT on math:")
    m_test_stat, m_effect, m_p_val = weat(embeddings, all_keys, 'math')
    print()

    print(f"For biased data, WEAT on science:")
    s_test_stat, s_effect, s_p_val = weat(embeddings, all_keys, 'science')
    print()

    print(f"For biased data, WEAT on race:")
    r_test_stat, r_effect, r_p_val = weat(embeddings, all_keys, 'race')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model_name", help="Model name to investigate", type=str, required=True)
    args = parser.parse_args()
    MODEL_NAME = args.model_name

    for test in ['math', 'science', 'career', 'race']:
        sweep_weat_n_k(model_base=MODEL_NAME,
                       test_name=test)

    # Run WEATs over debiased txt embeds from bolukbasi
    for debias_txt in os.listdir(DATA_DIR):
        if 'bolukbasi' in debias_txt and debias_txt.endswith('.txt') and 'debiased' in debias_txt:
            prior_work_weat(os.path.join(DATA_DIR, debias_txt))
        # else:
        #     # model_fn = debias_txt.split('_debiased')[0]
        #     # model_fn = model_fn.split('_original_')[-1]
        #     all_weats_one_model(MODEL_NAME)
