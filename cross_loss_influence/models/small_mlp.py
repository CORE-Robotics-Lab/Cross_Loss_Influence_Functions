"""
Created by anonymous author on 9/16/20

Adapted from https://github.com/vlukiyanov/pt-dec
"""


import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional, Tuple
from cross_loss_influence.data.scripts.generate_mog_data import MOGDataset
from sklearn.cluster import KMeans


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.
        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            # initial_cluster_centers = torch.rand((self.cluster_number, self.embedding_dimension), dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.
        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class DEC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cluster_number: int,
        hidden_dimension: int,
        alpha: float = 1.0,
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.
        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DEC, self).__init__()
        self.embedding_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension)
        )
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha
        )

    def encode(self, data_in: torch.Tensor) -> torch.Tensor:
        return self.embedding_layers(data_in)

    def forward(self, data_in: torch.Tensor) -> torch.Tensor:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.
        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        embedded_data = self.embedding_layers(data_in)
        return self.assignment(embedded_data)


def init_model(model_in: torch.nn.Module,
               dataset_in: Tuple[torch.Tensor, torch.Tensor] = None,
               device: str = 'cpu',
               save: bool = True,
               save_dir=None):
    import os
    from cross_loss_influence.config import MODEL_SAVE_DIR
    if save_dir is not None:
        model_save_path = save_dir
    else:
        model_save_path = MODEL_SAVE_DIR
    model_fn = os.path.join(model_save_path, 'mog_model_init.pth.tar')
    if save:
        ds = MOGDataset(dataset_in)
        num_clusters = len(np.unique(dataset_in[1]))

        kmeans = KMeans(n_clusters=model_in.cluster_number, n_init=num_clusters)
        model_in.train()
        features = []
        actual = []
        # form initial cluster centres
        for data, label in zip(dataset_in[0], dataset_in[1]):
            if ds.kl:
                label = torch.argmax(label)
            actual.append(label.item())
            data = data.to(device)
            features.append(model_in.encode(data).detach().cpu())
        predicted = kmeans.fit_predict(torch.stack(features).numpy())
        cluster_centers = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float, requires_grad=True, device=device
        )
        cluster_centers = cluster_centers + torch.randn(cluster_centers.size()).to(device)/2.0
        with torch.no_grad():
            model_in.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
        torch.save({
            'model':model_in.state_dict(),
        }, model_fn)
    else:
        chk = torch.load(model_fn, map_location='cpu')
        model_in.load_state_dict(chk['model'])
    return model_in


def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.
    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()