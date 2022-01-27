# Created by Andrew Silva
"""
With the known top-performers, plot each against each test.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from cross_loss_influence.config import DATA_DIR
from cross_loss_influence.helpers.plot_bias_movement import get_floats

if __name__ == "__main__":
    biased = True
    if biased:
        model_base = "DENSE_biased_window-10_negatives-10_60_"

        best_career = {
            'test': 'career',
            'mode': 'help-',
            'N': 100,
            'career': [],
            'math': [],
            'art': [],
            'race': []
        }
        best_math = {
            'test': 'math',
            'mode': 'help-',
            'N': 100,
            'career': [],
            'math': [],
            'art': [],
            'race': []
        }
        best_science = {
            'test': 'science',
            'mode': 'help-',
            'N': 100,
            'career': [],
            'math': [],
            'art': [],
            'race': []
        }
        best_race = {
            'test': 'race',
            'mode': 'help-',
            'N': 5,
            'career': [],
            'math': [],
            'art': [],
            'race': []
        }
    else:
        model_base = "DENSE_neutral_window-10_negatives-10_60_"

        best_career = {
            'test': 'career',
            'mode': 'harm-',
            'N': 5,
            'career': [],
            'math': [],
            'art': [],
            'race': []
        }
        best_math = {
            'test': 'math',
            'mode': 'help-',
            'N': 100,
            'career': [],
            'math': [],
            'art': [],
            'race': []
        }
        best_science = {
            'test': 'science',
            'mode': 'harm-',
            'N': 1000,
            'career': [],
            'math': [],
            'art': [],
            'race': []
        }
        best_race = {
            'test': 'race',
            'mode': 'help-',
            'N': 5,
            'career': [],
            'math': [],
            'art': [],
            'race': []
        }
    model_scores = []
    for model in [best_career, best_math, best_science, best_race]:
        n = model['N']
        h = model['mode']
        test_type = model['test']
        for k in [5, 10, 100, 1000]:
            model_name = model_base + test_type + f'-N-{n}-K-{k}-{h}-undone.txt'
            with open(os.path.join(DATA_DIR, model_name), 'r') as f:
                results = f.readlines()
            career_result = results[0]
            math_result = results[1]
            science_result = results[2]
            race_result = results[3]
            model['career'].append(get_floats(career_result)[1])
            model['math'].append(get_floats(math_result)[1])
            model['art'].append(-get_floats(science_result)[1])
            model['race'].append(get_floats(race_result)[1])
    for test_type in ['math', 'art', 'career', 'race']:
        x_values = np.arange(0, 4)
        plt.title(f"WEAT Scores on {test_type} with best models")
        plt.plot(x_values, np.zeros(len(x_values)), linewidth=1, color='darkgrey')
        plt.plot(x_values, best_career[test_type], marker='h', linewidth=4, markersize=8, color='red')
        plt.plot(x_values, best_math[test_type], marker='s', linewidth=4, markersize=8, color='blue')
        plt.plot(x_values, best_science[test_type], marker='o', linewidth=4, markersize=8, color='green')
        plt.plot(x_values, best_race[test_type], marker='^', linewidth=4, markersize=8, color='orange')
        if biased:
            plt.legend(['Career Correction, A=0, M=100', 'Math Correction, A=0, M=100',
                        'Art Correction, A=0, M=100', 'Race Correction, A=0, M=5'], loc='lower left')
        else:
            plt.legend(['Career Correction, A=5, M=0', 'Math Correction, A=0, M=100',
                        'Art Correction, A=1000, M=0', 'Race Correction, A=0, M=5'], loc='upper left')
        plt.ylim(-2, 2)
        plt.xticks([0, 1, 2, 3], ['K=5', 'K=10', 'K=100', 'K=1000'])
        plt.yticks([-2, 0, 2], ['Female', 'Neutral', 'Male'])
        model_type = 'neutral'
        if biased:
            model_type = 'biased'
        plt.savefig(os.path.join(DATA_DIR, f'best_{model_type}_{test_type}.png'))
        plt.close()