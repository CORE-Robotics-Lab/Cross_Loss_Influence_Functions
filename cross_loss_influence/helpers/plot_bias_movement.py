# Created by Andrew Silva
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from cross_loss_influence.config import DATA_DIR


def get_floats(string_in):
    floats_out = []
    for token in string_in.split():
        try:
            token = token.strip(',')
            tok = float(token)
            floats_out.append(tok)
        except:
            continue
    return floats_out


def plot_tests(model_name, test_name='race'):
    """
    Input a test name below (and the appropriate model name) and this will generate 3*5*5 plots for you, showing
    how each model (for example, help-n-5-k-5) scores on the 4 WEAT tests. This is used to find the best performers"""
    N = [5, 10, 100, 1000]
    K = [5, 10, 100, 1000]
    model_base = model_name
    modes = ['help-', 'harm-', 'both']
    for h in modes:
        for n in N:
            career_effects = []
            math_effects = []
            science_effects = []
            race_effects = []
            for k in K:
                model_name = model_base + '_' + test_name + f'-N-{n}-K-{k}-{h}-undone.txt'
                if not os.path.exists(os.path.join(DATA_DIR, model_name)):
                    print(f"{os.path.join(DATA_DIR, model_name)} does not exist -- Skipping...")
                    continue
                with open(os.path.join(DATA_DIR, model_name), 'r') as f:
                    results = f.readlines()
                career_result = results[0]
                math_result = results[1]
                science_result = results[2]
                race_result = results[3]
                career_effects.append(get_floats(career_result)[1])
                math_effects.append(get_floats(math_result)[1])
                science_effects.append(-get_floats(science_result)[1])
                race_effects.append(get_floats(race_result)[1])
            x_values = np.arange(0, len(career_effects))
            plt.title(f"{h} {test_name} with {n} samples")
            plt.plot(x_values, career_effects, marker='h', linewidth=4, color='red')
            plt.plot(x_values, math_effects, marker='s', linewidth=4, color='blue')
            plt.plot(x_values, science_effects, marker='o', linewidth=4, color='green')
            plt.plot(x_values, race_effects, marker='^', linewidth=4, color='orange')
            plt.legend(['Career', 'Math', 'Art', 'Race'], loc='lower left')
            plt.ylim(-2, 2)
            plt.xticks([0, 1, 2, 3], ['K=5', 'K=10', 'K=100', 'K=1000'])
            plt.yticks([-2, 0, 2], ['Female', 'Neutral', 'Male'])
            plt.show()


def get_best_result(model_name, test_name='career'):
    N = [5, 10, 100, 1000]
    K = [5, 10, 100, 1000]
    model_base = model_name
    modes = ['help-', 'harm-', 'both']
    effects = []
    for h in modes:
        for n in N:
            for k in K:
                result = 99999
                model_name = model_base + '_' + test_name + f'-N-{n}-K-{k}-{h}-undone.txt'
                if not os.path.exists(os.path.join(DATA_DIR, model_name)):
                    print(f"{os.path.join(DATA_DIR, model_name)} does not exist -- Skipping...")
                    continue
                with open(os.path.join(DATA_DIR, model_name), 'r') as f:
                    results = f.readlines()
                if test_name == 'career':
                    result = results[0]
                elif test_name == 'math':
                    result = results[1]
                elif test_name == 'science':
                    result = results[2]
                elif test_name == 'race':
                    result = results[3]
                floats_out = get_floats(result)
                effects.append([floats_out[1], floats_out[-1], model_name])
    least_biased = ''
    minimum_val = 9999
    p_val = 0.0
    for effect in effects:
        if abs(effect[0]) < abs(minimum_val):
            minimum_val = effect[0]
            p_val = effect[1]
            least_biased = effect[2]
    return minimum_val, p_val, least_biased


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model_name", help="Model name to investigate", type=str, required=True)
    args = parser.parse_args()
    MODEL_NAME = args.model_name
    for test in ['science', 'math', 'career', 'race']:
        lowest_val, p_val, model_name = get_best_result(MODEL_NAME, test)
        print(f"{model_name} scored as low as {lowest_val} at p={p_val}")
        plot_tests(MODEL_NAME, test)
