# Created by Andrew Silva
from cross_loss_influence.models.skip_gram_word2vec import SkipGramModel
import numpy as np
from cross_loss_influence.config import DATA_DIR, MODEL_SAVE_DIR, PROJECT_NAME
import torch
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import copy
from sklearn.neighbors import KDTree
from sklearn.manifold import TSNE


def get_embeddings(model_fn='mdl_fn',
                   model_dir=os.path.join(MODEL_SAVE_DIR, PROJECT_NAME, 'checkpoints'),
                   vocab_fn='stoi.pkl',
                   vocab_dir=DATA_DIR):
    # Load in model
    most_recent_model = torch.load(os.path.join(model_dir, model_fn), map_location='cpu')

    # Load in vocabulary
    stoi = pickle.load(open(os.path.join(vocab_dir, vocab_fn), 'rb'))

    # embeddings = most_recent_model['model_data']['word_embedding.weight']  # Transformer
    embeddings = most_recent_model['model_data']['u_embeddings.weight']  # Word2Vec
    # Get vocab words and indices for the embedding dictionary
    all_keys = []
    all_values = []
    for key, val in stoi.items():
        all_keys.append(key)
        all_values.append(val)

    # Creating "cluster data" (what goes into k-means) and "plot data" (what goes into matplotlib, but is labeled with cluster data)
    # embeddings = model.u_embeddings
    cluster_data = embeddings.detach().numpy()[:len(all_keys)]
    print(f"All told we have {cluster_data.shape[0]} embeddings")
    return cluster_data, all_keys[:len(cluster_data)]


if __name__ == "__main__":
    RESULT_SAVE_DIR = 'cluster_sci_fi_pytorch'


    cluster_data, all_keys = get_embeddings(model_fn='DENSE_scifi_window-3_negatives-5_last_checkpoint.pth.tar',
                                                vocab_fn='all_scripts_stoi.pkl')
    # 'DENSE_biased_window-10_negatives-10_60_math-N-100-K-1000-help--undone_checkpoint.pth'
    # cluster_data, all_keys = get_embeddings(model_fn='DENSE_biased_window-10_negatives-10_60_checkpoint.pth.tar',
    #                                         vocab_fn='biased_data_stoi.pkl')

    plot_data = copy.deepcopy(cluster_data)
    # KDTree for nearest neighbors
    tree = KDTree(cluster_data)
    plot_data = TSNE(n_components=2, perplexity=15).fit_transform(plot_data)
    print("TSNE finished.")
    # if os.path.exists(os.path.join(DATA_DIR, 'DENSE_plot_data_for_clusters.pkl')):
    #     plot_data = pickle.load(open(os.path.join(DATA_DIR, 'DENSE_plot_data_for_clusters.pkl'), 'rb'))
    # else:
    #     plot_data = TSNE(n_components=2, perplexity=30).fit_transform(plot_data)
    #     print("TSNE finished.")
    #     pickle.dump(plot_data, open(os.path.join(DATA_DIR, 'DENSE_plot_data_for_clusters.pkl'), 'wb'))
    # How many to look for
    range_n_clusters = range(2, 30)

    # directory within DATA_DIR/fig/ or DATA_DIR/txt/ to save results
    for n_clusters in [8]:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(plot_data) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random seed 42
        clf = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clf.fit_predict(cluster_data)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(cluster_data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(cluster_data, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(plot_data[:, 0], plot_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clf.cluster_centers_
        # Draw white circles at cluster centers
        new_centers = []
        with open(os.path.join(DATA_DIR, 'txt', RESULT_SAVE_DIR, f'{n_clusters}_clusters.txt'), 'w') as f:
            f.write(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")
            f.write('\n')

        for i, c in enumerate(centers):
            dist, ind = tree.query([c], k=25)
            with open(os.path.join(DATA_DIR, 'txt', RESULT_SAVE_DIR, f'{n_clusters}_clusters.txt'), 'a') as f:
                f.write(f"Cluster {i} nearest neighbors are: {[all_keys[word_ind] for word_ind in ind[0]]}")
                f.write('\n')
                print(f"Cluster {i} nearest neighbors are: {[all_keys[word_ind] for word_ind in ind[0]]}")
            new_centers.append(np.mean(plot_data[ind[0]], axis=0))

        words_of_interest = ['dooku', 'doctor', 'hero', 'sith', 'kirk', 'spock', 'ring', 'zulu']
        for word in words_of_interest:
            try:
                embed = cluster_data[all_keys.index(word)]
            except ValueError as e:
                print(f"{word} not found")
                continue
            dist, ind = tree.query([embed], k=25)
            with open(os.path.join(DATA_DIR, 'txt', RESULT_SAVE_DIR, f'{n_clusters}_clusters.txt'), 'a') as f:
                string_print = f"Word {word} nearest neighbors are: {[all_keys[word_ind] for word_ind in ind[0]]}"
                f.write(string_print)
                f.write('\n')
                print(string_print)
            new_centers.append(np.mean(plot_data[ind[0]], axis=0))

        new_centers = np.array(new_centers)
        ax2.scatter(new_centers[:, 0], new_centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        for n_i, n_c in enumerate(new_centers):
            ax2.scatter(n_c[0], n_c[1], marker='$%d$' % n_i, alpha=1,
                        s=50, edgecolor='k')
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(DATA_DIR, 'fig', RESULT_SAVE_DIR, f'{n_clusters}_clusters.png'))
        plt.close()
