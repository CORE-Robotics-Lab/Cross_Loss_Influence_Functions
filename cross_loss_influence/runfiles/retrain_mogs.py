"""
Created by anonymous author on 9/18/20
"""

# This could be more efficient, but it will need to be overhauled a bit.

from cross_loss_influence.data.scripts.generate_mog_data import reset_labels, plot_mog_data
from cross_loss_influence.models.small_mlp import DEC, target_distribution
from cross_loss_influence.runfiles.train_dec import train, init_model
import torch
import numpy as np
DEVICE = 'cuda'


def nll_from_sample(model_in, dataset_in, index_in):
    sample = dataset_in[0][index_in].to(device=DEVICE).view(1, -1)
    t = dataset_in[1][index_in].to(device=DEVICE).unsqueeze(0).type(torch.long)
    y = model_in(sample)
    y = y.log()
    loss = torch.nn.functional.nll_loss(y, t, reduction='mean')
    original_nll = loss.item()
    return original_nll



if __name__ == "__main__":
    # TEST THIS STUFF
    from cross_loss_influence.config import DATA_DIR, MODEL_SAVE_DIR
    import os

    np.set_printoptions(precision=6, suppress=True)
    data_checkpoint = torch.load(os.path.join(DATA_DIR, 'mog_data.tar.gz'))
    dataset = data_checkpoint['dataset']
    num_clust = data_checkpoint['num_clusters']
    dv = 'cuda'
    unsup_model = DEC(input_dim=2, hidden_dimension=12, cluster_number=num_clust).to(dv)
    torch.random.manual_seed(0)
    unsup_model = init_model(unsup_model, dataset, device=dv, save=False)

    # Re-train with NLL instead of KL?
    nll = True

    optim = torch.optim.Adam(unsup_model.parameters())

    finished_model = DEC(input_dim=2, hidden_dimension=12, cluster_number=num_clust).to(dv)

    model_name = os.path.join(MODEL_SAVE_DIR, 'mog_model_final.pth.tar')
    model_chkpt = torch.load(model_name, map_location='cpu')
    finished_model.load_state_dict(model_chkpt['model'])
    finished_model.to(dv)

    new_labels = reset_labels(finished_model, dataset, data_checkpoint['init_means'], dv)
    dataset = (dataset[0], new_labels)
    target_in = dataset[0].to(dv)
    output = finished_model(target_in)
    target = target_distribution(output).detach()


    for index in range(len(dataset[0])):
        base_nll = nll_from_sample(finished_model, dataset, index)
        diff_in_nlls = []
        for other_index in range(len(dataset[0])):
            new_dataset = torch.cat((dataset[0][:other_index], dataset[0][other_index+1:]))
            new_labelset = torch.cat((dataset[1][:other_index], dataset[1][other_index+1:]))
            t_in = torch.cat((target[:other_index], target[other_index+1:]))
            training_dataset = (new_dataset, new_labelset)
            unsup_model = init_model(unsup_model, dataset, device=dv, save=False)
            other_path = os.path.join(MODEL_SAVE_DIR, f'mog_model_final_without_{other_index}.pth.tar')
            if os.path.exists(other_path):
                model_name = os.path.join(MODEL_SAVE_DIR, other_path)
                model_chkpt = torch.load(model_name, map_location='cpu')
                unsup_model.load_state_dict(model_chkpt['model'])
            else:
                train(
                    data_in=training_dataset,
                    model=unsup_model,
                    epochs=2,
                    batch_size=500,
                    optimizer=optim,
                    silent=True,
                    device=dv,
                    evaluate_batch_size=-1,
                    nll=nll
                )
                torch.save({
                    'model': unsup_model.state_dict(),
                    'optimizer': optim.state_dict()
                }, other_path)

            new_nll = nll_from_sample(unsup_model, dataset, index)
            diff_in_nlls.append([other_index, base_nll-new_nll])

        f1 = os.path.join(DATA_DIR, 'mog_data', f'mog_empirical_influence_{index}.txt')
        diff_in_nlls = np.array(diff_in_nlls)
        harmfuls = diff_in_nlls[diff_in_nlls[:, 1].argsort()]
        helpfuls = harmfuls[::-1]
        with open(f1, 'w') as f:
            f.write(f"Influences: {diff_in_nlls}")
            f.write(f"Harmful: {harmfuls}")
            f.write('\n')
            f.write(f"Helpful: {helpfuls}")
            f.write('\n')
        hurters = torch.stack([dataset[0][int(i[0])] for i in helpfuls[:10]]).numpy()
        helpers = torch.stack([dataset[0][int(i[0])] for i in harmfuls[:10]]).numpy()
        plot_mog_data(mog_data=dataset[0], mog_labels=dataset[1], save_fn=f'mog_empirical_influence_{index}',
                      helpful_points=helpers, harmful_points=hurters, central_pt=dataset[0][index], show=False)
