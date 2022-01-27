# Created by Andrew Silva
import torch
import time
import datetime
import os

import numpy as np
import copy
import logging
from torch.autograd import grad
import random
from cross_loss_influence.helpers.bolukbasi_prior_work.prior_pca_debiasing import extract_txt_embeddings
from torch.utils.data.dataloader import DataLoader
DEVICE = 'cuda'


def calc_influence_single(model, train_dataset, z_test, t_test, target_dist, nll):
    """Calculates the influences of all training data points on a single
    test dataset image.
    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        embedding_pair: pair of embeddings we want to diff
    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        """

    train_loader = DataLoader(train_dataset, shuffle=False)
    s_test_vec = s_test(z_test, t_test, model, train_loader, target_dist=target_dist, nll=nll)

    # Calculate the influence function
    train_dataset_size = len(train_dataset)
    influences = []

    train_loader = DataLoader(train_dataset, shuffle=False)
    if nll:
        loss_fn = torch.nn.NLLLoss().to(DEVICE)
    else:
        loss_fn = torch.nn.KLDivLoss(reduction='batchmean').to(DEVICE)
    for i, batch_data in enumerate(train_loader):  # instead of random, get all samples of relevance in the dataset.
        data, labels = batch_data
        data = data.to(DEVICE)
        m_out = model(data)

        if nll:
            loss_val = loss_fn(m_out.log(), labels.to(device=DEVICE).type(torch.long))
        else:
            loss_val = loss_fn(m_out.log().view(-1), target_dist[i].view(-1))
        grad_z_vec = list(grad(loss_val, list(model.parameters()), create_graph=True))

        tmp_influence = -sum(
            [
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size
        influences.append([i, tmp_influence.cpu().detach().item()])

    influences = np.array(influences)
    harmful = influences[influences[:, 1].argsort()]
    helpful = harmful[::-1]

    return influences, harmful, helpful


def s_test(z_test, t_test, model, train_loader, damp=0.01, scale=25.0, target_dist=None, nll=False):
    # Damp = 0.01, scale=25.0
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, stochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.
    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        train_loader: torch dataloader, can load the training dataset
        damp: float, dampening factor
        scale: float, scaling factor
    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(z_test, t_test, model)

    h_estimates = v.copy()
    if nll:
        loss_fn = torch.nn.NLLLoss().to(DEVICE)
    else:
        loss_fn = torch.nn.KLDivLoss(reduction='batchmean').to(DEVICE)
    for i, batch_data in enumerate(train_loader):  # instead of random, get all samples of relevance in the dataset.
        data, labels = batch_data
        data = data.to(DEVICE)

        m_out = model(data)
        if nll:
            loss_val = loss_fn(m_out.log(), labels.to(device=DEVICE).type(torch.long))
        else:
            loss_val = loss_fn(m_out.log().view(-1), target_dist[i].view(-1))
        hv = hvp(loss_val, list(model.parameters()), h_estimates)
        # Recursively caclulate h_estimate
        if not hv:
            continue
        # h_estimates = [
        #     _v + (1 - damp) * h_estimate - _hv / scale
        #     for _v, h_estimate, _hv in zip(v, h_estimates, hv)]
        for h_index, bucket in enumerate(zip(v, h_estimates, hv)):
            temp_v, h_est, temp_hv = bucket
            if h_est is not None:
                temp_h_est = temp_v + (1 - damp) * h_est - temp_hv / scale
                # print((h_estimates[h_index] - temp_h_est).abs().sum())
                h_estimates[h_index] = temp_h_est
                # h_estimates[h_index] = temp_v + (1 - damp) * h_est - temp_hv / scale
    return h_estimates


def grad_z(z, t, model):
    """Calculates the gradient z. One grad_z should be computed for each sample.
    Arguments:
        z: torch tensor, data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, data labels
        model: torch NN, model used to evaluate the dataset
    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    # initialize
    z = z.to(device=DEVICE).view(1, -1)
    t = t.to(device=DEVICE).unsqueeze(0).type(torch.long)
    y = model(z)
    y = y.log()
    loss = calc_loss(y, t)
    # Compute sum of gradients from model parameters to loss
    return list(grad(loss, list(model.parameters()), create_graph=True))


def calc_loss(y, t):
    """Calculates the loss
    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)
    Returns:
        loss: scalar, the loss"""
    loss = torch.nn.functional.nll_loss(y, t, reduction='mean')
    return loss


def hvp(ys, xs, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.
    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(xs) != len(v):
        raise(ValueError("xs and v must have the same length."))

    # First backprop
    first_grads = grad(ys, xs, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        if grad_elem is not None and v_elem is not None:
            elemwise_products += torch.sum(grad_elem * v_elem.detach())
    # Second backprop
    if elemwise_products == 0:
        return False
    return_grads = grad(elemwise_products, xs)  # , create_graph=True)

    return return_grads


def write_out_results(influences, harmfuls, helpfuls, dataset, file_prefix='', data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    if not os.path.exists(os.path.join(data_dir, 'mog_data')):
        os.mkdir(os.path.join(data_dir, 'mog_data'))
    f1 = os.path.join(data_dir, 'mog_data', file_prefix+"test_results.txt")
    with open(f1, 'w') as f:
        f.write(f"Influences: {influences}")
        f.write('\n')
        f.write(f"Harmful: {harmfuls}")
        f.write('\n')
        f.write(f"Helpful: {helpfuls}")
        f.write('\n')

    f2 = os.path.join(data_dir, 'mog_data', file_prefix+'harmful_ordered.txt')
    for h in harmfuls:
        bad_coordinates = dataset[int(h[0])].numpy().tolist()
        with open(f2, 'a', encoding='utf-8') as f:
            f.write(f'index: {h[0]} || coordinates: {bad_coordinates}')
            f.write('\n')

    f3 = os.path.join(data_dir, 'mog_data', file_prefix+'helpful_ordered.txt')
    for h in helpfuls:
        good_coordinates = dataset[int(h[0])].numpy().tolist()
        with open(f3, 'a', encoding='utf-8') as f:
            f.write(f'index: {h[0]} || coordinates: {good_coordinates}')
            f.write('\n')


if __name__ == "__main__":
    from cross_loss_influence.models.small_mlp import DEC, target_distribution
    from cross_loss_influence.data.scripts.generate_mog_data import MOGDataset, reset_labels, plot_mog_data
    from cross_loss_influence.config import DATA_DIR, MODEL_SAVE_DIR, PROJECT_NAME

    np.set_printoptions(precision=6, suppress=True)
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = DEVICE

    # Use NLL (match train/test)?
    nll = False

    # Load in the original data
    chkpt = torch.load(os.path.join(DATA_DIR, 'mog_data.tar.gz'))
    num_clust = chkpt['num_clusters']
    data = chkpt['dataset']

    # Load in the model
    unsup_model = DEC(input_dim=2, hidden_dimension=12, cluster_number=num_clust).to(device)
    model_name = os.path.join(MODEL_SAVE_DIR, 'mog_model_final.pth.tar')
    model_chkpt = torch.load(model_name, map_location='cpu')
    unsup_model.load_state_dict(model_chkpt['model'])

    # Reset the labels
    new_labels = reset_labels(unsup_model, data, chkpt['init_means'], device)
    data = (data[0], new_labels)

    # Get the target distribution
    target_in = data[0].to(device)
    output = unsup_model(target_in)
    target = target_distribution(output).detach()

    for index, (data_point, label) in enumerate(zip(data[0], data[1])):
        influence, harmful, helpful = calc_influence_single(model=unsup_model,
                                                            train_dataset=MOGDataset(data),
                                                            z_test=data_point, t_test=label, target_dist=target,
                                                            nll=nll)
        if nll:
            file_prefix = f'mog_influence_matched_{index}'
        else:
            file_prefix = f'mog_influence_cross_loss_{index}'
        write_out_results(influence, harmful, helpful, data[0], file_prefix=file_prefix)
        helpers = torch.stack([data[0][int(i[0])] for i in helpful[:10]]).numpy()
        hurters = torch.stack([data[0][int(i[0])] for i in harmful[:10]]).numpy()
        plot_mog_data(mog_data=data[0], mog_labels=data[1], save_fn=file_prefix,
                      helpful_points=hurters, harmful_points=helpers, central_pt=data_point, show=False)
