"""
Created by anonymous author on 9/20/20
"""
import torch
from cross_loss_influence.runfiles.retrain_mogs import nll_from_sample
from cross_loss_influence.data.scripts.generate_mog_data import reset_labels, plot_mog_data
from cross_loss_influence.models.small_mlp import DEC, target_distribution
from cross_loss_influence.runfiles.train_dec import train, init_model
import torch
import numpy as np

def get_class_average_loss_nl(model_in, dataset_in):
    class_holder = {0: [], 1:[], 2:[], 3:[]}
    for idx in range(len(dataset_in[0])):
        nl = nll_from_sample(model_in, dataset_in, idx)
        class_holder[dataset_in[1][idx].item()].append(nl)
    for cls, arr in class_holder.items():
        print(f'for class {cls} average classification loss is {np.mean(arr)} with std {np.std(arr)}')


def get_class_average_loss_kl(model_in, dataset_in, target=None):
    class_holder = {0: [], 1:[], 2:[], 3:[]}
    for idx in range(len(dataset_in[0])):
        sample = dataset_in[0][idx].to(device=DEVICE).view(1, -1)
        y = model_in(sample)
        y = y.log()
        loss_value = torch.nn.functional.kl_div(y, target[idx], reduction='batchmean')
        class_holder[dataset_in[1][idx].item()].append(loss_value.item())
    for cls, arr in class_holder.items():
        print(f'for class {cls} average KL Div loss is {np.mean(arr)} with std {np.std(arr)}')


def get_model_losses(model_in, dataset_in, target):
    get_class_average_loss_nl(model_in=model_in, dataset_in=dataset_in)
    get_class_average_loss_kl(model_in=model_in, dataset_in=dataset_in, target=target)


def get_dataset_stats(dataset_in):
    for cls in np.unique(dataset_in[1]):
        in_sample_points = dataset_in[0][torch.where(dataset_in[1]==cls)]
        dim_one = in_sample_points[:, 0]
        dim_two = in_sample_points[:, 1]
        print(f'for class {cls} dim_one mean is {dim_one.mean()} and std is {dim_one.std()}')
        print(f'for class {cls} dim_two mean is {dim_two.mean()} and std is {dim_two.std()}')


def get_model_means(model_in):
    cluster_centers = model_in.assignment.cluster_centers.clone().cpu().detach().numpy()
    print(cluster_centers)
    return cluster_centers


def get_embedding_means(model_in, dataset_in):
    embed_means = []
    for cls in np.unique(dataset_in[1]):
        in_sample_points = dataset_in[0][torch.where(dataset_in[1]==cls)]
        in_sample_points = in_sample_points.to(DEVICE)
        embeddings = model_in.encode(in_sample_points)
        embed_means.append(embeddings.mean(dim=0).cpu().detach().numpy())
        print(f'for class {cls} average embedding is {embed_means[-1]}')
    return embed_means


def starry_print(str_in: str, num_stars:int):
    print(f'{"".join(["*"]*num_stars)} {str_in} {"".join(["*"]*num_stars)}')


if __name__ == "__main__":
    from cross_loss_influence.config import DATA_DIR, MODEL_SAVE_DIR
    import os
    DEVICE = 'cuda'
    np.set_printoptions(precision=6, suppress=True)
    data_checkpoint = torch.load(os.path.join(DATA_DIR, 'mog_data.tar.gz'))
    dataset = data_checkpoint['dataset']
    num_clust = data_checkpoint['num_clusters']
    dv = 'cuda'
    initial_model = DEC(input_dim=2, hidden_dimension=12, cluster_number=num_clust).to(dv)
    torch.random.manual_seed(0)
    initial_model = init_model(initial_model, dataset, device=dv, save=False)

    finished_model = DEC(input_dim=2, hidden_dimension=12, cluster_number=num_clust).to(dv)

    model_name = os.path.join(MODEL_SAVE_DIR, 'mog_model_final.pth.tar')
    model_chkpt = torch.load(model_name, map_location='cpu')
    finished_model.load_state_dict(model_chkpt['model'])

    new_labels = reset_labels(finished_model, dataset, data_checkpoint['init_means'], dv)
    dataset = (dataset[0], new_labels)
    target_in = dataset[0].to(dv)
    output = finished_model(target_in)
    target = target_distribution(output).detach()

    stars = 15
    starry_print("INITIAL MODEL LOSSES", stars)
    get_model_losses(initial_model, dataset, target)

    starry_print("FINAL MODEL LOSSES", stars)
    get_model_losses(finished_model, dataset, target)

    starry_print("DATASET STATISTICS", stars)
    get_dataset_stats(dataset_in=dataset)

    starry_print("INITIAL MODEL MEANS", stars)
    initial_means = get_model_means(initial_model)

    starry_print("FINAL MODEL MEANS", stars)
    final_means = get_model_means(finished_model)

    starry_print("INITIAL ENCODING MEANS", stars)
    initial_embeddings = get_embedding_means(initial_model, dataset_in=dataset)

    starry_print("FINAL ENCODING MEANS", stars)
    final_embeddings = get_embedding_means(finished_model, dataset_in=dataset)

    starry_print("INITIAL MEAN DISTANCES", stars)
    print(np.abs(initial_embeddings-initial_means).sum(axis=1))

    starry_print("FINAL MEAN DISTANCES", stars)
    print(np.abs(final_embeddings-final_means).sum(axis=1))