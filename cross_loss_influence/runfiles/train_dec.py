"""
Created by anonymous author on 9/17/20

Adapted from https://github.com/vlukiyanov/pt-dec/
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Union
from tqdm import tqdm

from cross_loss_influence.models.small_mlp import target_distribution, cluster_accuracy, DEC, init_model
from cross_loss_influence.data.scripts.generate_mog_data import MOGDataset, plot_mog_data, reset_labels


def train(
    data_in: Tuple[torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    silent: bool = False,
    device: str = 'cpu',
    evaluate_batch_size: int = 1024,
    nll:bool = False
) -> None:
    """
    Train the DEC model given a dataset, a model instance and various configuration parameters.
    :param data_in: input data
    :param model: instance of DEC model to train
    :param epochs: number of training epochs
    :param batch_size: size of the batch to train with
    :param optimizer: instance of optimizer to use
    :param device: device name
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param evaluate_batch_size: batch size for evaluation stage, default 1024
    :param nll: Use NLL loss?
    :return: None
    """
    ds = MOGDataset(data_in)
    train_dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
    )
    accuracy = 0.0
    if nll:
        loss_function = nn.NLLLoss().to(device)
    else:
        loss_function = nn.KLDivLoss(reduction='batchmean').to(device)
    for epoch in range(epochs):
        features = []
        data_iterator = tqdm(
            train_dataloader,
            leave=True,
            unit="batch",
            postfix={
                "epo": epoch,
                "acc": "%.4f" % (accuracy or 0.0),
                "lss": "%.8f" % 0.0,
            },
            disable=silent,
        )
        model.train()
        for index, batch in enumerate(data_iterator):
            batch, target = batch
            batch = batch.to(device)
            output = model(batch)
            if nll:
                target = target.to(device=device).type(torch.long)
            else:
                if ds.kl:
                    target = target.to(device)
                else:
                    target = target_distribution(output).detach()
            # loss = loss_function(output.log(), target) / output.shape[0]
            if nll:
                loss_val = loss_function(output.log(), target)
            else:
                loss_val = loss_function(output.log(), target)
            data_iterator.set_postfix(
                epo=epoch,
                acc="%.4f" % (accuracy or 0.0),
                lss="%.8f" % float(loss_val.item()),
            )
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            features.append(model.encode(batch).detach().cpu())
        if evaluate_batch_size > 0:
            predicted, actual = predict(
                data_in,
                model,
                batch_size=evaluate_batch_size,
                silent=True,
                return_actual=True,
                device=device,
            )
            if ds.kl:
                accuracy = torch.where(predicted==actual)[0].size(0)/actual.size(0)
            else:
                _, accuracy = cluster_accuracy(predicted.cpu().numpy(), actual.cpu().numpy())
            data_iterator.set_postfix(
                epo=epoch,
                acc="%.4f" % (accuracy or 0.0),
                lss="%.8f" % 0.0,
            )


def predict(
    data_in: Tuple[torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
    batch_size: int = 1024,
    device: str = 'cpu',
    silent: bool = False,
    return_actual: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Predict clusters for a dataset given a DEC model instance and various configuration parameters.
    :param data_in: data
    :param model: instance of DEC model to predict
    :param batch_size: size of the batch to predict with, default 1024
    :param device: which device to put the data on
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param return_actual: return actual values, if present in the Dataset
    :return: tuple of prediction and actual if return_actual is True otherwise prediction
    """
    ds = MOGDataset(data_in)

    dataloader = DataLoader(
        ds, batch_size=batch_size, shuffle=False
    )
    data_iterator = tqdm(dataloader, leave=True, unit="batch", disable=silent,)
    features = []
    actual = []
    model.eval()
    for batch in data_iterator:
        batch, value = batch  # unpack if we have a prediction label
        if ds.kl:
            value = torch.argmax(value, dim=1)
        if return_actual:
            actual.append(value)
        batch = batch.to(device)

        features.append(
            model(batch).detach().cpu()
        )  # move to the CPU to prevent out of memory on the GPU
    if return_actual:
        return torch.cat(features).max(1)[1], torch.cat(actual).long()
    else:
        return torch.cat(features).max(1)[1]


if __name__ == "__main__":
    from cross_loss_influence.config import DATA_DIR, MODEL_SAVE_DIR
    import os

    nll = False

    data_checkpoint = torch.load(os.path.join(DATA_DIR, 'mog_data.tar.gz'))
    dataset = data_checkpoint['dataset']
    print(dataset[1])
    num_clust = data_checkpoint['num_clusters']
    dv = 'cuda'
    unsup_model = DEC(input_dim=2, hidden_dimension=12, cluster_number=num_clust).to(dv)
    torch.random.manual_seed(0)
    unsup_model = init_model(unsup_model, dataset, device=dv, save=False)

    optim = torch.optim.Adam(unsup_model.parameters())
    # dataset = (dataset[0][:-3], dataset[1][:-3])
    train(
        data_in=dataset,
        model=unsup_model,
        epochs=2,
        batch_size=500,
        optimizer=optim,
        silent=False,
        device=dv,
        evaluate_batch_size=100,
        nll=nll)

    model_fn = os.path.join(MODEL_SAVE_DIR, 'mog_model_final.pth.tar')
    torch.save({
        'model': unsup_model.state_dict(),
        'optimizer': optim.state_dict()
    }, model_fn)

    predicted, actual = predict(
        dataset,
        unsup_model,
        batch_size=100,
        silent=True,
        return_actual=True,
        device=dv,
    )
    plot_mog_data(mog_data=dataset[0], mog_labels=predicted)
    plot_mog_data(mog_data=dataset[0], mog_labels=actual)
