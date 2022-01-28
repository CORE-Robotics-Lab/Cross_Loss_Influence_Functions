# Created by Andrew Silva
import os
from cross_loss_influence.data.scripts.my_pytorch_dataset import MedText, MedTextFiles, CythonMedDataset
from cross_loss_influence.models.skip_gram_word2vec import SkipGramModel
from torch.utils.data import DataLoader
import time
import numpy as np
import itertools
from cross_loss_influence.config import PROJECT_HOME, DATA_DIR, MODEL_SAVE_DIR, PROJECT_NAME
import torch
import argparse


def most_similar(token, skipgram_model, top_k=8):
    """
    Most similar words to a token or word
    :param token: token or word in
    :param skipgram_model: model to use for nearest neighbor prediction
    :param top_k: k nearest neighbors to return
    :return: list of token indices
    """
    if isinstance(token, str):
        from cross_loss_influence.data.scripts.vocab_builder import tokenize_string
        token = tokenize_string(token)[0]
    from cross_loss_influence.data.scripts.vocab_builder import stringify_token
    print(f"Finding most similar word to {stringify_token(token)}")
    token_index = torch.tensor(token, dtype=torch.long).cuda().unsqueeze(0).to(device=skipgram_model.device)
    emb = skipgram_model.predict(token_index)
    sim = torch.mm(emb, skipgram_model.u_embeddings.weight.transpose(0, 1))
    nearest = (-sim[0]).sort()[1][1: top_k + 1]
    top_list = []
    for k in range(top_k):
        nearby_tok = nearest[k].item()
        print(f"{k} nearest is {stringify_token(nearby_tok)}")
        top_list.append(nearby_tok)
    return top_list


def batch_data_to_tensors(batch_data_in):
    all_words = []
    all_contexts = []
    all_negatives = []
    for b in batch_data_in:
        all_words.extend(b[0])
        all_contexts.extend(b[1])
        for sample in range(len(b[2][0])):
            temp_arr = []
            for negative in range(len(b[2])):
                temp_arr.append(b[2][negative][sample])
            all_negatives.append(temp_arr)
    all_words = torch.tensor(all_words, dtype=torch.long)
    all_contexts = torch.tensor(all_contexts, dtype=torch.long)
    all_negatives = torch.tensor(all_negatives, dtype=torch.long)
    return all_words, all_contexts, all_negatives


def vectorized_batch_data_to_tensors(batch_data_in):
    np_batch = np.array(batch_data_in)
    all_words = torch.stack(np_batch[:, 0].tolist()).type(torch.long).view(-1)
    all_contexts = torch.stack(np_batch[:, 1].tolist()).type(torch.long).view(-1)
    all_negatives = list(itertools.chain.from_iterable(np_batch[:, 2]))
    all_negatives = torch.stack(all_negatives).type(torch.long)
    all_negatives = all_negatives.view([all_words.size(0), -1])
    return all_words, all_contexts, all_negatives


def training_log(start_time, index, batch_size, dataset_size, loss_val):
    elapsed_time = time.time() - start_time
    est_remaining = elapsed_time / (index * batch_size / dataset_size) - elapsed_time
    print(f"Index {index} || {dataset_size - index * batch_size} Remaining || Loss: {loss_val} \n"
          f"Elapsed Time: {elapsed_time} || Est. Time Remaining {est_remaining}")


def train_skipgram_model(window_size,
                         number_negatives,
                         batch_size=1000,
                         number_epochs=200,
                         num_workers=30,
                         learning_rate=1e-2,
                         cuda=True,
                         sparse=True):
    """
    Train skipgram model with given window and negative stuff
    :param window_size: context window size
    :param number_negatives: number of negatives per sample
    :param batch_size: batch size
    :param number_epochs: number of epochs to train
    :param num_workers: number of workers for the data loader
    :param cuda: on cuda?
    :return: nothing. saves the model to model_save_directory/project_name/checkpoints/asdfasdf.pth.tar
    """
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model_data_name = 'biased'

    model_name = f'DENSE_{model_data_name}_window-{window_size}_negatives-{number_negatives}'
    print(f"Batch size = {batch_size}")
    print(f"Num workers = {num_workers}")
    print(f"Model name is {model_name}")
    start_time = time.time()
    med_dataset = CythonMedDataset(data_dir=DATA_DIR,
                                   filename=f'{model_data_name}_data_numericalized_dataset.pkl',
                                   window_size=window_size,
                                   num_negs=number_negatives)

    print(f"Dataset created in {time.time()-start_time} seconds")

    log_every = max(len(med_dataset)//batch_size//10, 1)
    dataset_size = len(med_dataset)
    print(f"Logging every {log_every}")

    test_dataloader = DataLoader(dataset=med_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers)
    print("Data loader created")

    model = SkipGramModel(vocab_size=25355, embedding_dim=100, sparse=sparse).to(device=device)  # 15282 scripts size

    optim_adam = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim_adam, .9, last_epoch=-1)
    base_epoch = 0
    if os.path.exists(os.path.join(MODEL_SAVE_DIR, PROJECT_NAME, 'checkpoints', f'{model_name}_last_checkpoint.pth.tar')):
        checkpoint = torch.load(os.path.join(MODEL_SAVE_DIR, PROJECT_NAME, 'checkpoints', f'{model_name}_last_checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['model_data'])
        optim_adam.load_state_dict(checkpoint['opt_data'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_data'])
        base_epoch = checkpoint['epoch']
    else:
        torch.save({
            'model_data': model.state_dict(),
            'epoch': 0,
            'opt_data': optim_adam.state_dict(),
            'lr_scheduler_data': lr_scheduler.state_dict()
        }, os.path.join(MODEL_SAVE_DIR, PROJECT_NAME, 'checkpoints', f"{model_name}_init_checkpoint.pth.tar"))


    for epoch in range(1, number_epochs+1-base_epoch):
        epoch += base_epoch
        start_time = time.time()
        for index, batch_data in enumerate(test_dataloader, 1):
            words, contexts, negatives = vectorized_batch_data_to_tensors(batch_data)
            words = torch.autograd.Variable(words).to(device=device)
            contexts = torch.autograd.Variable(contexts).to(device=device)
            negatives = torch.autograd.Variable(negatives).to(device=device)

            optim_adam.zero_grad()
            loss_val = model(words, contexts, negatives)
            loss_val.backward()
            optim_adam.step()
            if index % log_every == 0:
                training_log(start_time, index, batch_size, dataset_size, loss_val.item())
        if epoch % 10 == 0:
            torch.save({
                'model_data': model.state_dict(),
                'epoch': epoch,
                'opt_data': optim_adam.state_dict(),
                'lr_scheduler_data': lr_scheduler.state_dict()
            }, os.path.join(MODEL_SAVE_DIR, PROJECT_NAME, 'checkpoints', f"{model_name}_{epoch}_checkpoint.pth.tar"))
        else:
            torch.save({
                'model_data': model.state_dict(),
                'epoch': epoch,
                'opt_data': optim_adam.state_dict(),
                'lr_scheduler_data': lr_scheduler.state_dict()
            }, os.path.join(MODEL_SAVE_DIR, PROJECT_NAME, 'checkpoints', f"{model_name}_last_checkpoint.pth.tar"))
        lr_scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--context-window', type=int, default=8, help='context window size (default: 4)')
    parser.add_argument('-n', '--number-negatives', type=int, default=20, help='number of negative samples (default: 15')
    parser.add_argument('-b', '--batch-size', type=int, default=1000, help='batch size (default: 1000)')
    parser.add_argument('-e', '--number-epochs', type=int, default=35, help='number of epochs to train (default: 100')
    parser.add_argument('-w', '--num-workers', type=int, default=10, help='number of workers for dataloader (default: 10')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-2, help='learning rate (default: 1e-4)')
    parser.add_argument('--disable-cuda', action='store_true', help='disable cuda (default: off/False)')
    parser.add_argument('--sparse', action='store_true', help='sparse tensors (default:False)')

    args = parser.parse_args()
    use_gpu = not args.disable_cuda and torch.cuda.is_available()

    train_skipgram_model(window_size=args.context_window,
                         number_negatives=args.number_negatives,
                         batch_size=args.batch_size,
                         number_epochs=args.number_epochs,
                         num_workers=args.num_workers,
                         learning_rate=args.learning_rate,
                         cuda=use_gpu,
                         sparse=args.sparse)
