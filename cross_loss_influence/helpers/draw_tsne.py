# Created by Andrew Silva
from cross_loss_influence.helpers.sklearn_cluster_embeddings import get_embeddings
from cross_loss_influence.config import DATA_DIR
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_names_pleasantness(data_in):
    cluster_data, all_keys, perp, fn_ext = data_in
    A = ['freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond',
         'gentle', 'honest', 'lucky', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family',
         'happy', 'laughter', 'vacation']
    B = ['crash', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink',
         'assault', 'disaster', 'hatred', 'tragedy', 'bomb', 'divorce', 'jail', 'poverty', 'ugly',
         'cancer', 'evil', 'kill', 'rotten', 'vomit']
    C = ['jew', 'jewish', 'islam', 'religion', 'islamic', 'muslim']
    X = ['josh', 'alan', 'ryan', 'andrew', 'jack', 'greg', 'amanda', 'katie', 'nancy', 'ellen']
    Y = ['theo', 'jerome', 'leroy', 'lamar', 'lionel', 'malik', 'tyrone', 'ebony', 'jasmine', 'tia', ]
    A = [all_keys.index(x) for x in A]
    X = [all_keys.index(x) for x in X]
    Y = [all_keys.index(x) for x in Y]
    B = [all_keys.index(x) for x in B]
    C = [all_keys.index(x) for x in C]

    pca = PCA(n_components=50, svd_solver='full')
    cluster_data = pca.fit_transform(cluster_data)
    cluster_data = TSNE(n_components=2, perplexity=perp).fit_transform(cluster_data)

    colors = [[0.75, 0.75, 0.75]]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    plt.scatter(cluster_data[A, 0], cluster_data[A, 1], marker='o', s=50, lw=0, alpha=0.95, c='cornflowerblue', edgecolor='k')
    plt.scatter(cluster_data[B, 0], cluster_data[B, 1], marker='s', s=50, lw=0, alpha=0.95, c='red', edgecolor='k')
    plt.scatter(cluster_data[C, 0], cluster_data[C, 1], marker='h', s=50, lw=0, alpha=0.95, c='goldenrod', edgecolor='k')

    plt.scatter(cluster_data[X, 0], cluster_data[X, 1], marker='^', s=50, lw=0, alpha=0.95, c='forestgreen', edgecolor='k')
    plt.scatter(cluster_data[Y, 0], cluster_data[Y, 1], marker='*', s=50, lw=0, alpha=0.95, c='black', edgecolor='k')
    plt.savefig(os.path.join(DATA_DIR, f'neutral_race_tsne_{fn_ext}_{perp}.png'))
    plt.close()

if __name__ == "__main__":
    from multiprocessing import Pool

    cluster_data, all_keys = get_embeddings(model_fn='DENSE_neutral_window-10_negatives-10_60_checkpoint.pth.tar',
                                            vocab_fn='biased_data_stoi.pkl')

    over_cluster_data, over_all_keys = get_embeddings(model_fn='DENSE_neutral_window-10_negatives-10_60_race-N-100-K-1000-help--undone_checkpoint.pth.tar',
                                            vocab_fn='biased_data_stoi.pkl')
    perp = 20
    data_in_1 = [cluster_data, all_keys, perp, 'init']
    data_in_2 = [over_cluster_data, over_all_keys, perp, 'overcorrected']
    data_run = [data_in_1, data_in_2]
    pool = Pool()
    pool.map(plot_names_pleasantness, data_run)