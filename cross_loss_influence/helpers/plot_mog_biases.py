"""
Created by anonymous author on 9/18/20
"""


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from cross_loss_influence.config import DATA_DIR, MODEL_SAVE_DIR
    from cross_loss_influence.data.scripts.generate_mog_data import reset_labels
    from cross_loss_influence.models.small_mlp import DEC
    from cross_loss_influence.runfiles.train_dec import init_model
    import torch
    import os
    from scipy.stats import pearsonr, ttest_ind
    import numpy as np

    DATASET_SIZE = 150


    data_checkpoint = torch.load(os.path.join(DATA_DIR, 'mog_data.tar.gz'))
    dataset = data_checkpoint['dataset']
    num_clust = data_checkpoint['num_clusters']
    dv = 'cuda'
    unsup_model = DEC(input_dim=2, hidden_dimension=12, cluster_number=num_clust).to(dv)
    torch.random.manual_seed(0)
    unsup_model = init_model(unsup_model, dataset, device=dv, save=False)

    optim = torch.optim.Adam(unsup_model.parameters())

    finished_model = DEC(input_dim=2, hidden_dimension=12, cluster_number=num_clust).to(dv)

    model_name = os.path.join(MODEL_SAVE_DIR, 'mog_model_final.pth.tar')
    model_chkpt = torch.load(model_name, map_location='cpu')
    finished_model.load_state_dict(model_chkpt['model'])

    new_labels = reset_labels(finished_model, dataset, data_checkpoint['init_means'], dv)
    new_labels = new_labels.cpu().numpy()
    dataset = (dataset[0].to(dv), new_labels)
    preds = finished_model(dataset[0]).cpu().detach().argmax(dim=1)
    preds = preds.cpu().numpy()
    matched_pearson_values = {0: [], 1:[], 2:[], 3:[], 4:[]}
    plot_results = True
    cross_loss_pearson_values = {0: [], 1:[], 2:[], 3:[], 4:[]}
    for index in range(DATASET_SIZE):
        empirical_fn = os.path.join(DATA_DIR, 'mog_data', f'mog_empirical_influence_{index}.txt')
        matched_inf_fn = os.path.join(DATA_DIR, 'mog_data', f'mog_influence_matched_{index}test_results.txt')
        cross_loss_inf_fn = os.path.join(DATA_DIR, 'mog_data', f'mog_influence_cross_loss_{index}test_results.txt')
        with open(empirical_fn, 'r') as f:
            emp_infs = f.readlines()
        with open(matched_inf_fn, 'r') as f:
            matched_inf_fn = f.readlines()
        with open(cross_loss_inf_fn, 'r') as f:
            cross_loss_inf_fn = f.readlines()
        first_emp = [float(emp_infs[0][-11:-2])]
        first_m_inf = [float(matched_inf_fn[0][-11:-2])]
        first_c_inf = [float(cross_loss_inf_fn[0][-11:-2])]
        emp_infs = first_emp + [float(line.split('[')[1].split(']')[0][-9:]) for line in emp_infs[1:DATASET_SIZE]]
        matched_inf_fn = first_m_inf + [float(line.split('[')[1].split(']')[0][-9:]) for line in matched_inf_fn[1:DATASET_SIZE]]
        cross_loss_inf_fn = first_c_inf + [float(line.split('[')[1].split(']')[0][-9:]) for line in cross_loss_inf_fn[1:DATASET_SIZE]]
        emp_infs = np.array(emp_infs)
        matched_inf_fn = np.array(matched_inf_fn)
        cross_loss_inf_fn = np.array(cross_loss_inf_fn)

        m, b = np.polyfit(emp_infs, matched_inf_fn, 1)
        matched_best_fit = emp_infs*m + b
        matched_r_squared = pearsonr(emp_infs, matched_inf_fn)
        matched_pearson_values[new_labels[index].item()].append(matched_r_squared[0])

        m, b = np.polyfit(emp_infs, cross_loss_inf_fn, 1)
        best_fit = emp_infs*m + b
        r_squared = pearsonr(emp_infs, cross_loss_inf_fn)
        cross_loss_pearson_values[new_labels[index].item()].append(r_squared[0])
        # sst = np.sum((emp_infs - inf_infs)**2)
        # ss_reg = np.sum((best_fit - inf_infs)**2)
        # r_squared = ss_reg / sst

        # emp_infs = np.array(emp_infs)/np.max(np.abs(emp_infs))
        # inf_infs = np.array(inf_infs)/np.max(np.abs(inf_infs))
        if plot_results:
            if preds[index] == new_labels[index]:
                plt.plot(emp_infs, matched_inf_fn, 'r+')
                plt.plot(emp_infs, cross_loss_inf_fn, 'b+')
            else:
                plt.plot(emp_infs, matched_inf_fn, 'ro')
                plt.plot(emp_infs, cross_loss_inf_fn, 'bo')
            plt.plot(emp_infs, matched_best_fit, c='purple', label=f'Matched Pearson\'s R: {matched_r_squared[0]} \n p = {matched_r_squared[1]}')
            plt.plot(emp_infs, best_fit, c='goldenrod', label=f'Cross-Loss Pearson\'s R: {r_squared[0]} \n p = {r_squared[1]}')
            plt.xlabel('empirical influence')
            plt.ylabel('predicted influence')
            prop = {}
            if r_squared[0]>=0.9:
                prop['weight'] = 'bold'

            plt.legend(loc='lower left', prop=prop)

            plt.savefig(os.path.join(DATA_DIR, 'mog_data', 'correlations', f'mog_fit_{index}.png'))
            plt.close()

        # plt.show()
    for key, val in matched_pearson_values.items():
        if len(val) < 1:
            print(f"Skipping {key} for an empty array...")
            continue
        matched_val = np.array(val)
        cross_loss_val = np.array(cross_loss_pearson_values[key])
        print(f'Matched mean pearson for class {key} is {np.mean(matched_val)} and std {np.std(matched_val)}')
        print(f'Percent greater than 0.6: {len(matched_val[matched_val>=0.6])/float(len(matched_val))}')
        print(f'Percent greater than 0.8: {len(matched_val[matched_val>=0.8])/float(len(matched_val))}')

        print(f'Cross Loss mean pearson for class {key} is {np.mean(cross_loss_val)} and std {np.std(cross_loss_val)}')
        print(f'Percent greater than 0.6: {len(cross_loss_val[cross_loss_val>=0.6])/float(len(cross_loss_val))}')
        print(f'Percent greater than 0.8: {len(cross_loss_val[cross_loss_val>=0.8])/float(len(cross_loss_val))}')

        print(f'T test between matched and cross loss correlations: {ttest_ind(matched_val, cross_loss_val)}')
