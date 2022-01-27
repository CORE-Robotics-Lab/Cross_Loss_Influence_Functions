"""
Created by Andrew Silva on 01/25/22
"""
from cross_loss_influence.data.scripts.generate_mog_data import reset_labels, \
    plot_mog_data, generate_mixture_normals, MOGDataset
from cross_loss_influence.models.small_mlp import target_distribution, cluster_accuracy, DEC, init_model
from cross_loss_influence.runfiles.train_dec import train
from cross_loss_influence.helpers.mog_influence_function import calc_influence_single, write_out_results
from cross_loss_influence.runfiles.retrain_mogs import nll_from_sample
import torch
import copy
import numpy as np
from scipy.stats import pearsonr, ttest_ind
import matplotlib.pyplot as plt
import os
np.set_printoptions(precision=6, suppress=True)


def load_models(ds_obj, device='cuda'):
    base_model = DEC(input_dim=2, hidden_dimension=12, cluster_number=ds_obj['num_clusters']).to(device)
    base_model = init_model(base_model, ds_obj['dataset'], device=device, save=False)  # Only save once
    non_nll_model = DEC(input_dim=2, hidden_dimension=12, cluster_number=ds_obj['num_clusters']).to(device)
    nll_model = DEC(input_dim=2, hidden_dimension=12, cluster_number=ds_obj['num_clusters']).to(device)
    model_non_chkpt = torch.load(os.path.join(DATA_PATH, 'non_nll.pt'), map_location='cpu')
    non_nll_model.load_state_dict(model_non_chkpt['model'])
    model_chkpt = torch.load(os.path.join(DATA_PATH, 'nll.pt'), map_location='cpu')
    nll_model.load_state_dict(model_chkpt['model'])
    return base_model, non_nll_model, nll_model


def train_two_models(ds_obj, device='cuda'):
    base_model = DEC(input_dim=2, hidden_dimension=12, cluster_number=N_CLUSTERS).to(device)
    base_model = init_model(base_model, ds_obj['dataset'], device=device, save=True)  # Only save once

    torch.random.manual_seed(0)
    non_nll = copy.deepcopy(base_model)
    non_optim = torch.optim.Adam(non_nll.parameters())
    train(
        data_in=ds_obj['dataset'],
        model=non_nll,
        epochs=2,
        batch_size=500,
        optimizer=non_optim,
        silent=False,
        device=device,
        evaluate_batch_size=100,
        nll=False)
    torch.save({
        'model': non_nll.state_dict(),
        'optimizer': non_optim.state_dict()
    }, os.path.join(DATA_PATH, 'non_nll.pt'))

    nll_model = copy.deepcopy(base_model)
    nll_optim = torch.optim.Adam(nll_model.parameters())
    train(
        data_in=ds_obj['dataset'],
        model=nll_model,
        epochs=2,
        batch_size=500,
        optimizer=nll_optim,
        silent=False,
        device=device,
        evaluate_batch_size=100,
        nll=True)
    torch.save({
        'model': nll_model.state_dict(),
        'optimizer': nll_optim.state_dict()
    }, os.path.join(DATA_PATH, 'nll.pt'))
    return base_model, non_nll, nll_model


def dataset_influence(model_in, ds_obj, nll, device='cuda'):
    new_labels = reset_labels(model_in, ds_obj['dataset'], m, device)
    data = (ds_obj['dataset'][0], new_labels)
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # Get the target distribution
    target_in = data[0].to(device)
    output = model_in(target_in)
    target = target_distribution(output).detach()

    for index, (data_point, label) in enumerate(zip(data[0], data[1])):
        influence, harmful, helpful = calc_influence_single(model=model_in,
                                                            train_dataset=MOGDataset(data),
                                                            z_test=data_point, t_test=label, target_dist=target,
                                                            nll=nll)
        if nll:
            file_prefix = f'mog_influence_matched_{index}'
        else:
            file_prefix = f'mog_influence_cross_loss_{index}'
        write_out_results(influence, harmful, helpful, data[0], file_prefix=file_prefix, data_dir=DATA_PATH)


def retrain_influence(initial_model, finished_model, ds_obj, nll=False, device='cuda'):

    optim = torch.optim.Adam(initial_model.parameters())

    new_labels = reset_labels(finished_model, ds_obj['dataset'], ds_obj['init_means'], device)
    dataset = (ds_obj['dataset'][0], new_labels)
    target_in = dataset[0].to(device)
    output = finished_model(target_in)
    target = target_distribution(output).detach()

    for index in range(len(dataset[0])):
        base_nll = nll_from_sample(finished_model, dataset, index)
        diff_in_nlls = []
        for other_index in range(len(dataset[0])):
            new_dataset = torch.cat((dataset[0][:other_index], dataset[0][other_index+1:]))
            new_labelset = torch.cat((dataset[1][:other_index], dataset[1][other_index+1:]))
            training_dataset = (new_dataset, new_labelset)
            initial_model = init_model(initial_model, dataset, device=device, save=False)
            other_path = os.path.join(DATA_PATH, f'mog_model_final_without_{other_index}.pth.tar')
            train(
                    data_in=training_dataset,
                    model=initial_model,
                    epochs=2,
                    batch_size=500,
                    optimizer=optim,
                    silent=True,
                    device=dv,
                    evaluate_batch_size=-1,
                    nll=nll
                )
            torch.save({
                'model': initial_model.state_dict(),
                'optimizer': optim.state_dict()
            }, other_path)
            new_nll = nll_from_sample(initial_model, dataset, index)
            diff_in_nlls.append([other_index, base_nll-new_nll])

        f1 = os.path.join(DATA_PATH, 'mog_data', f'mog_empirical_influence_{index}.txt')
        diff_in_nlls = np.array(diff_in_nlls)
        harmfuls = diff_in_nlls[diff_in_nlls[:, 1].argsort()]
        helpfuls = harmfuls[::-1]
        with open(f1, 'w') as f:
            f.write(f"Influences: {diff_in_nlls}")
            f.write(f"Harmful: {harmfuls}")
            f.write('\n')
            f.write(f"Helpful: {helpfuls}")
            f.write('\n')


def correlate_influences(initial_model, finished_model, ds_obj, device='cuda'):
    if not os.path.exists(os.path.join(DATA_PATH, 'mog_data', 'correlations')):
        os.mkdir(os.path.join(DATA_PATH, 'mog_data', 'correlations'))
    dataset = ds_obj['dataset']
    torch.random.manual_seed(0)

    new_labels = reset_labels(finished_model, dataset, ds_obj['init_means'], device)
    new_labels = new_labels.cpu().numpy()
    dataset = (dataset[0].to(device), new_labels)
    preds = finished_model(dataset[0]).cpu().detach().argmax(dim=1)
    preds = preds.cpu().numpy()
    matched_pearson_values = {0: [], 1: [], 2: [], 3: [], 4: []}
    plot_results = True
    cross_loss_pearson_values = {0: [], 1: [], 2: [], 3: [], 4: []}
    d_size_limit = len(ds_obj['dataset'][0])
    for index in range(d_size_limit):
        empirical_fn = os.path.join(DATA_PATH, 'mog_data', f'mog_empirical_influence_{index}.txt')
        matched_inf_fn = os.path.join(DATA_PATH, 'mog_data', f'mog_influence_matched_{index}test_results.txt')
        cross_loss_inf_fn = os.path.join(DATA_PATH, 'mog_data', f'mog_influence_cross_loss_{index}test_results.txt')
        with open(empirical_fn, 'r') as f:
            emp_infs = f.readlines()
        with open(matched_inf_fn, 'r') as f:
            matched_inf_fn = f.readlines()
        with open(cross_loss_inf_fn, 'r') as f:
            cross_loss_inf_fn = f.readlines()
        first_emp = [float(emp_infs[0][-11:-2])]
        first_m_inf = [float(matched_inf_fn[0][-11:-2])]
        first_c_inf = [float(cross_loss_inf_fn[0][-11:-2])]
        emp_infs = first_emp + [float(line.split('[')[1].split(']')[0][-9:]) for line in emp_infs[1:d_size_limit]]
        matched_inf_fn = first_m_inf + [float(line.split('[')[1].split(']')[0][-9:]) for line in
                                        matched_inf_fn[1:d_size_limit]]
        cross_loss_inf_fn = first_c_inf + [float(line.split('[')[1].split(']')[0][-9:]) for line in
                                           cross_loss_inf_fn[1:d_size_limit]]
        emp_infs = np.array(emp_infs)
        matched_inf_fn = np.array(matched_inf_fn)
        cross_loss_inf_fn = np.array(cross_loss_inf_fn)

        m, b = np.polyfit(emp_infs, matched_inf_fn, 1)
        matched_best_fit = emp_infs * m + b
        matched_r_squared = pearsonr(emp_infs, matched_inf_fn)
        matched_pearson_values[new_labels[index].item()].append(matched_r_squared[0])

        m, b = np.polyfit(emp_infs, cross_loss_inf_fn, 1)
        best_fit = emp_infs * m + b
        r_squared = pearsonr(emp_infs, cross_loss_inf_fn)
        cross_loss_pearson_values[new_labels[index].item()].append(r_squared[0])

        if plot_results:
            if preds[index] == new_labels[index]:
                plt.plot(emp_infs, matched_inf_fn, 'r+')
                plt.plot(emp_infs, cross_loss_inf_fn, 'b+')
            else:
                plt.plot(emp_infs, matched_inf_fn, 'ro')
                plt.plot(emp_infs, cross_loss_inf_fn, 'bo')
            plt.plot(emp_infs, matched_best_fit, c='purple',
                     label=f'Matched Pearson\'s R: {matched_r_squared[0]} \n p = {matched_r_squared[1]}')
            plt.plot(emp_infs, best_fit, c='goldenrod',
                     label=f'Cross-Loss Pearson\'s R: {r_squared[0]} \n p = {r_squared[1]}')
            plt.xlabel('empirical influence')
            plt.ylabel('predicted influence')
            prop = {}
            if r_squared[0] >= 0.9:
                prop['weight'] = 'bold'

            plt.legend(loc='lower left', prop=prop)

            plt.savefig(os.path.join(DATA_PATH, 'mog_data', 'correlations', f'mog_fit_{index}.png'))
            plt.close()
    for key, val in matched_pearson_values.items():
        if len(val) < 1:
            print(f"Skipping {key} for an empty array...")
            continue
        matched_val = np.array(val)
        cross_loss_val = np.array(cross_loss_pearson_values[key])
        print(f'Matched mean pearson for class {key} is {np.mean(matched_val)} and std {np.std(matched_val)}')
        print(f'Percent greater than 0.6: {len(matched_val[matched_val >= 0.6]) / float(len(matched_val))}')
        print(f'Percent greater than 0.8: {len(matched_val[matched_val >= 0.8]) / float(len(matched_val))}')

        print(f'Cross Loss mean pearson for class {key} is {np.mean(cross_loss_val)} and std {np.std(cross_loss_val)}')
        print(f'Percent greater than 0.6: {len(cross_loss_val[cross_loss_val >= 0.6]) / float(len(cross_loss_val))}')
        print(f'Percent greater than 0.8: {len(cross_loss_val[cross_loss_val >= 0.8]) / float(len(cross_loss_val))}')

        print(f'T test between matched and cross loss correlations: {ttest_ind(matched_val, cross_loss_val)}')


if __name__ == "__main__":
    import random
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--data_path", help="where to store everything?", type=str, default='../CLIF_MOG_DATA/')
    parser.add_argument("-r", "--re_use", action="store_true", help="re-use an already-generated MoG result?")
    args = parser.parse_args()
    dv = 'cuda'

    if os.path.exists(args.data_path):
        print('using existing path')
    else:
        os.mkdir(args.data_path)
    DATA_PATH = args.data_path
    if args.re_use:
        print("Loading saved dataset, models, and influences...")
        dataset_object = torch.load(os.path.join(DATA_PATH, 'mog_data.tar.gz'))
        original_model, dec_model, sup_model = load_models(ds_obj=dataset_object, device=dv)
    else:
        # GENERATE DATA:
        m = [[0, 0], [5, 5], [5, -5]]
        s = [[1.25] * 2] * len(m)

        D_SIZE = 50
        N_CLUSTERS = 3
        print("Generating MoG dataset...")
        DATASET = generate_mixture_normals(means=m, std=s, size=(D_SIZE, 2), kl_labels=False)
        dataset_object = {
            'dataset': DATASET,
            'num_clusters': len(m),
            'init_means': m,
            'init_std': s
        }
        torch.save(dataset_object, os.path.join(DATA_PATH, 'mog_data.tar.gz'))
        print("Training DEC and NLL models...")
        original_model, dec_model, sup_model = train_two_models(dataset_object, device=dv)

        print("Computing DEC influence...")
        dataset_influence(model_in=dec_model, ds_obj=dataset_object, nll=False, device=dv)
        print("Computing NLL influence...")
        dataset_influence(model_in=sup_model, ds_obj=dataset_object, nll=True, device=dv)
        print("Computing empirical influence...")
        retrain_influence(initial_model=original_model, finished_model=sup_model, ds_obj=dataset_object, nll=True,
                          device=dv)
    print("Correlating influences...")
    correlate_influences(initial_model=original_model, finished_model=sup_model, ds_obj=dataset_object, device=dv)
