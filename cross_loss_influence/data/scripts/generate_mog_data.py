# Created by Andrew Silva on 9/16/20
# Generates mixture of gaussian data
import numpy as np
import torch
import typing as t
from torch.utils.data import Dataset


def generate_normal_data(mean: t.Collection[float], std: t.Collection[float], size: t.Tuple) -> torch.Tensor:
    noise = np.random.normal(mean, std, size)
    noise = torch.from_numpy(noise).type(torch.float)
    return noise


def generate_mixture_normals(means: t.Collection[t.Collection[float]],
                             std: t.Collection[t.Collection[float]],
                             size: t.Tuple,
                             kl_labels: bool = False) -> t.Tuple[torch.Tensor, torch.Tensor]:
    assert len(means) == len(std)
    gaussians = []
    labels = []
    for m, s, l in zip(means, std, np.arange(0, len(means))):
        gaussians.append(generate_normal_data(m, s, size))
        if kl_labels:
            centroid = torch.ones_like(gaussians[-1])*torch.tensor(m)
            dist_from_centroid = torch.sum(torch.abs(gaussians[-1] - centroid), dim=1)
            max_val = torch.max(dist_from_centroid)
            k_labs = torch.ones((gaussians[-1].size(0), len(means)))*(max_val/10.0)
            k_labs[:, l] += torch.abs(dist_from_centroid-max_val*2.0)
            labels.append(torch.nn.functional.softmax(k_labs, dim=1))
        else:
            labels.append(torch.ones(gaussians[-1].size(0)) * l)
    gaussians = torch.cat(gaussians, dim=0)
    labels = torch.cat(labels, dim=0)
    return gaussians, labels


def plot_mog_data(mog_data: torch.Tensor, mog_labels: torch.Tensor, save_fn: str = 'mog_fig',
                  helpful_points: torch.Tensor = None, harmful_points: torch.Tensor = None,
                  central_pt: torch.Tensor = None, show: bool = True):
    import itertools
    import matplotlib.pyplot as plt
    from cross_loss_influence.config import DATA_DIR
    import os
    mog_data = mog_data.cpu().numpy()
    mog_labels = mog_labels.cpu().numpy()
    unique_labs = np.unique(mog_labels)
    plt_hds = []
    colors = itertools.cycle(['red', 'blue', 'black', 'magenta', 'green', 'orange', 'yellow'])
    for cls in unique_labs:
        indices = np.where(mog_labels == cls)
        handle = plt.scatter(mog_data[indices, 0], mog_data[indices, 1], marker="s", s=125, c=next(colors), alpha=0.7,
                             label=cls)
        plt_hds.append(handle)
    plt.title("Mixture of Gaussians")
    if helpful_points is not None:
        plt.scatter(helpful_points[:, 0], helpful_points[:, 1], marker='>', s=200, c='green', label='Helpful Points')
    if harmful_points is not None:
        plt.scatter(harmful_points[:, 0], harmful_points[:, 1], marker='*', s=200, c='orange', label='Harmful Points')
    if central_pt is not None:
        plt.scatter(central_pt[0], central_pt[1], marker='o', s=200, c='cyan', label='Central Point')

    legend_handle = plt.legend(bbox_to_anchor=(0, 1), ncol=2, loc='upper right')
    plt.savefig(os.path.join(DATA_DIR, f'{save_fn}.png'), bbox_extra_artists=(legend_handle,), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


class MOGDataset(Dataset):
    def __init__(self, data_in: t.Tuple[torch.Tensor, torch.Tensor]):
        """
        Args:
            data_in: mixture of gaussian data, as above. In the form of a tuple[data, labels]
        """
        self.dataset = data_in[0]
        self.labels = data_in[1]
        self.kl = False
        if len(self.labels[0].size()) > 0:
            self.kl = True

    def __len__(self):
        return self.dataset.size(0)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


def reset_labels(model_in, dataset_in, initial_means, device='cpu'):
    sam = torch.tensor(initial_means, dtype=torch.float, device=device)
    new_labels = model_in(sam).argmax(dim=1)
    original_labels = np.unique(dataset_in[1])
    new_label_holder = []
    for old_lab, new_lab in zip(original_labels, new_labels):
        ones = torch.ones_like(torch.where(dataset_in[1]==old_lab)[0], device=device)
        ones*=new_lab
        new_label_holder.append(ones)
    new_labels = torch.stack(new_label_holder).view(-1)
    return new_labels

if __name__ == "__main__":
    from cross_loss_influence.config import DATA_DIR
    import os
    m = [[0, 0], [5, 5], [5, -5]]  # , [5, 1], [1, 5]]
    s = [[1.25] * 2] * len(m)  # Old version

    # s = [[1, 0.5], [1, 0.5], [0.5, 1], [0.5, 1]]  # New version

    dataset = generate_mixture_normals(means=m, std=s, size=(50, 2), kl_labels=False)

    torch.save({
        'dataset': dataset,
        'num_clusters': len(m),
        'init_means': m,
        'init_std': s
    }, os.path.join(DATA_DIR, 'mog_data.tar.gz'))