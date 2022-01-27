# Created by Andrew Silva
import os
from cross_loss_influence.data.scripts.my_pytorch_dataset import CythonMedDataset
from cross_loss_influence.models.skip_gram_word2vec import SkipGramModel
from cross_loss_influence.helpers.influence_function import load_prior_work_embeddings
from torch.utils.data import DataLoader
import time
import itertools
import numpy as np
from cross_loss_influence.config import DATA_DIR, MODEL_SAVE_DIR, PROJECT_NAME
import torch
import argparse


def overdo_skipgram_model(model_name,
                          undo_indices,
                          double_down_indices,
                          dataset_fn,
                          window_size,
                          num_negatives=10,
                          num_repeats=10,
                          cuda=True,
                          test_name='',
                          prior_embeds=None):
    """
    Train skipgram model with given window and negative stuff
    :param model_name: which model to use
    :param undo_indices: which samples to step backwards
    :param num_repeats: how many times to step backwards
    :param dataset_fn: dataset filename all_scripts_numericalized_dataset.pkl or biased_data_numericalized_dataset.pkl
    :param window_size: window size used for the model and dataset (5 or 10 i think)
    :param num_negatives: number of negative samples per pass. 10 for both
    :param cuda: on cuda?
    :return: nothing. saves the model to model_save_directory/project_name/checkpoints/asdfasdf.pth.tar
    """
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    start_time = time.time()
    med_dataset = CythonMedDataset(data_dir=DATA_DIR,
                                   filename=dataset_fn,
                                   window_size=window_size,
                                   num_negs=num_negatives)

    print(f"Dataset created in {time.time()-start_time} seconds")

    dataset_size = len(med_dataset)
    test_dataloader = DataLoader(dataset=med_dataset, batch_size=1, shuffle=False, num_workers=0)
    print("Data loader created")
    most_recent_model = torch.load(os.path.join(MODEL_SAVE_DIR, PROJECT_NAME, 'checkpoints', model_name+'_checkpoint.pth.tar'),
                                   map_location='cpu')

    model = SkipGramModel(vocab_size=len(most_recent_model['model_data']['u_embeddings.weight']),
                          embedding_dim=len( most_recent_model['model_data']['u_embeddings.weight'][0]), sparse=False).to(device=device)  # 15282 scripts size

    optim_adam = torch.optim.Adam(lr=1e-2, params=model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim_adam, .9, last_epoch=-1)
    optim_adam.load_state_dict(most_recent_model['opt_data'])
    lr_scheduler.load_state_dict(most_recent_model['lr_scheduler_data'])
    model.load_state_dict(most_recent_model['model_data'])
    if prior_embeds is not None:
        model = load_prior_work_embeddings(model, prior_embeds)

    # Load saved model and optim stuff. This is hard coded for now...

    for undo_step in range(num_repeats):
        for index in undo_indices:
            undo_data = med_dataset[index]
            words, contexts, negatives = vectorized_non_tensor_loading(undo_data)
            words = torch.autograd.Variable(words).to(device=device)
            contexts = torch.autograd.Variable(contexts).to(device=device)
            negatives = torch.autograd.Variable(negatives).to(device=device)

            optim_adam.zero_grad()
            loss_val = model(words, contexts, negatives)
            loss_val.backward()
            optim_adam.step()
    for undo_step in range(num_repeats):
        for index in double_down_indices:
            undo_data = med_dataset[index]
            words, contexts, negatives = vectorized_non_tensor_loading(undo_data)
            words = torch.autograd.Variable(words).to(device=device)
            contexts = torch.autograd.Variable(contexts).to(device=device)
            negatives = torch.autograd.Variable(negatives).to(device=device)

            optim_adam.zero_grad()
            loss_val = -model(words, contexts, negatives)
            loss_val.backward()
            optim_adam.step()
    model_save_name = model_name.split()
    if prior_embeds is not None:
        model_name = 'bolukbasi_' + model_name
    torch.save({
        'model_data': model.state_dict(),
        'epoch': 99999,
        'opt_data': optim_adam.state_dict(),
        'lr_scheduler_data': lr_scheduler.state_dict()
    }, os.path.join(MODEL_SAVE_DIR, PROJECT_NAME, 'checkpoints', f"{model_name}_{test_name}-overdone_checkpoint.pth.tar"))


def vectorized_non_tensor_loading(batch_data_in):
    np_batch = np.array(batch_data_in)
    all_words = torch.tensor(np_batch[:, 0].tolist()).type(torch.long).view(-1)
    all_contexts = torch.tensor(np_batch[:, 1].tolist()).type(torch.long).view(-1)
    all_negatives = list(itertools.chain.from_iterable(np_batch[:, 2]))
    all_negatives = torch.tensor(all_negatives).type(torch.long)
    all_negatives = all_negatives.view([all_words.size(0), -1])
    return all_words, all_contexts, all_negatives


def mp_overdo_testtype(test='math biased'):
    model_base = test.split()[1]
    test = test.split()[0]
    prior = ''  # 'bolukbasi'
    use_gpu = True
    for harm in ['harm', 'help', 'both']:
        for N in [5, 10, 100, 1000]:
            for K in [5, 10, 100, 1000]:
                u_indices = []
                dd_indices = []
                if harm == 'harm' or harm == 'both':
                    id_txt = os.path.join(DATA_DIR, f'{test}_{model_base}_influence_{prior}harmful_ids.txt')
                    with open(id_txt, 'r') as f:
                        indices = f.readlines()
                    indices = indices[0].split()
                    indices = [int(eval(x)) for x in indices]
                    u_indices = indices[:N]
                elif harm == 'help' or harm == 'both':
                    id_txt = os.path.join(DATA_DIR, f'{test}_{model_base}_influence_{prior}helpful_ids.txt')
                    with open(id_txt, 'r') as f:
                        indices = f.readlines()
                    indices = indices[0].split()
                    indices = [int(eval(x)) for x in indices]
                    dd_indices = indices[:N]

                test_name = f'{test}-N-{N}-K-{K}-'
                if harm == 'harm':
                    test_name += 'harm-'
                elif harm == 'help':
                    test_name += 'help-'
                elif harm == 'both':
                    test_name += 'both'
                tail = 'debiased'
                if test == 'race':
                    tail += '-race'
                if len(prior) > 0:
                    prior_f = f'{prior}_original_DENSE_{model_base}_window-10_negatives-10_60_checkpoint.pth.tar_{tail}.txt'
                    prior_f = os.path.join(DATA_DIR, prior_f)
                else:
                    prior_f = None
                overdo_skipgram_model(
                    model_name=f'DENSE_{model_base}_window-10_negatives-10_60',
                    undo_indices=u_indices,
                    double_down_indices=dd_indices,
                    dataset_fn=f'{model_base}_data_numericalized_dataset.pkl',
                    window_size=10,
                    num_negatives=10,
                    num_repeats=K,
                    cuda=use_gpu,
                    test_name=test_name,
                    prior_embeds=prior_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Knowledge Discovery (because mp failed)')
    parser.add_argument('--disable-cuda', action='store_true', help='disable cuda (default: off/False)')

    args = parser.parse_args()
    use_gpu = not args.disable_cuda and torch.cuda.is_available()

    tests = ['race biased', 'race neutral', 'career biased', 'career neutral',
             'science biased', 'science neutral', 'math biased', 'math neutral']
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    from multiprocessing import Pool
    pool = Pool(8)
    pool.map(mp_overdo_testtype, tests)
