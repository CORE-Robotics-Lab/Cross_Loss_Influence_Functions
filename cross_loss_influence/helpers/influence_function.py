# Created by Andrew Silva
# Extensions to https://github.com/nimarb/pytorch_influence_functions
import torch
import time
import datetime

import numpy as np
import copy
import logging
from torch.autograd import grad
import random
from cross_loss_influence.helpers.bolukbasi_prior_work.prior_pca_debiasing import extract_txt_embeddings
from torch.utils.data.dataloader import DataLoader
DEVICE = 'cuda'


def calc_influence_single(model, train_dataset, z_test, t_test, recursion_depth, r, test_indices, scifi=True):
    """Calculates the influences of all training data points on a single
    test dataset image.
    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        embedding_pair: pair of embeddings we want to diff
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        """

    train_loader = DataLoader(train_dataset, shuffle=True)
    s_test_vec = calc_s_test_single(model,
                                    z_test,
                                    t_test,
                                    train_loader,
                                    recursion_depth=recursion_depth,
                                    r=r,
                                    test_indices=test_indices,
                                    scifi=scifi)

    # Calculate the influence function
    train_dataset_size = len(train_dataset)
    influences = []

    train_loader = DataLoader(train_dataset, shuffle=False)

    for index, batch_data in enumerate(train_loader):
        good_enough = False  # Is a word of interest in this sample?
        words, contexts = vectorized_influence_data_to_tensors(batch_data)
        for v_index in test_indices:
            if v_index in words.cpu():
                good_enough = True
        if not good_enough:
            continue
        words = torch.autograd.Variable(words).to(device=DEVICE)
        contexts = torch.autograd.Variable(contexts).to(device=DEVICE)
        loss_val = model.forward_no_negatives(words, contexts)
        grad_z_vec = list(grad(loss_val, list(model.parameters()), create_graph=True))
        # For sparse:
        if recursion_depth <= 1:
            tmp_influence = 0
            for k, j in zip(grad_z_vec, s_test_vec):
                if (k * j).indices().size(1) > 0:
                    tmp_influence -= (k * j).values().sum()/train_dataset_size
        # For dense
        else:
            tmp_influence = -sum(
                [
                    ####################
                    # TODO: potential bottle neck, takes 17% execution time
                    # torch.sum(k * j).data.cpu().numpy()
                    ####################
                    torch.sum(k * j).data
                    for k, j in zip(grad_z_vec, s_test_vec)
                ]) / train_dataset_size
        influences.append([index, tmp_influence.cpu()])

    influences = np.array(influences)
    harmful = influences[influences[:, 1].argsort()]
    helpful = harmful[::-1]
    influences = influences[:, 1]

    return influences, harmful.tolist(), helpful.tolist()


def calc_s_test_single(model, z_test, t_test, train_loader,
                       damp=0.01, scale=25, recursion_depth=5000, r=1, test_indices=[], scifi=True):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))
    Arguments:
        model: pytorch model, for which s_test should be calculated
        z_test: test image
        t_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""
    s_test_vec_list = []
    # For my sparse approach, 1 pass is all we need and we go through the entire dataset to get samples of import.

    for i in range(r):
        print("Beginning another round of estimation")
        s_test_vec_list.append(s_test(z_test, t_test, model, train_loader, damp=damp, scale=scale,
                                      recursion_depth=recursion_depth, test_indices=test_indices, scifi=scifi))

    s_test_vec = s_test_vec_list[0]
    for i in range(1, r):
        s_test_vec += s_test_vec_list[i]

    s_test_vec = [i / r for i in s_test_vec if i is not None]

    return s_test_vec


def s_test(z_test, t_test, model, train_loader, damp=0.01, scale=25.0,
           recursion_depth=5000, test_indices=[], scifi=True):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.
    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        train_loader: torch dataloader, can load the training dataset
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.
    Returns:
        h_estimate: list of torch tensors, s_test"""
    # v = grad_z(z_test, t_test, model)
    if scifi:
        v = calc_loss(z_test, t_test)  # Change this to bias estimation
    else:
        v = calc_bias(z_test, t_test, model)
    v = list(grad(v, list(model.parameters()), create_graph=True, allow_unused=True))  # A bit sketched by this
    # v[1] = v[0]
    h_estimates = v.copy()
    if recursion_depth <= 1:  # If we're sparse
        success_limit = 5000
    else:
        success_limit = recursion_depth

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    successes = 0
    for i, batch_data in enumerate(train_loader):  # instead of random, get all samples of relevance in the dataset.
        good_enough = False  # Is a word of interest in this sample?
        words, contexts = vectorized_influence_data_to_tensors(batch_data)
        for v_index in test_indices:
            if v_index in words.cpu():
                good_enough = True
        if not good_enough:
            continue
        words = torch.autograd.Variable(words).to(device=DEVICE)
        contexts = torch.autograd.Variable(contexts).to(device=DEVICE)
        loss_val = model.forward_no_negatives(words, contexts)
        hv = hvp(loss_val, list(model.parameters()), h_estimates, sparse=recursion_depth == 1)
        # Recursively caclulate h_estimate
        if not hv:
            continue
        successes += 1
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

        if successes >= success_limit:
            break
    return h_estimates


# def grad_z(z, t, model):
#     """Calculates the gradient z. One grad_z should be computed for each
#     training sample.
#     Arguments:
#         z: torch tensor, training data points
#             e.g. an image sample (batch_size, 3, 256, 256)
#         t: torch tensor, training data labels
#         model: torch NN, model used to evaluate the dataset
#     Returns:
#         grad_z: list of torch tensor, containing the gradients
#             from model parameters to loss"""
#     model.eval()
#     # initialize
#     z = z.to(device=DEVICE)
#     t = t.to(device=DEVICE)
#     y = model(z)
#     loss = calc_loss(y, t)
#     # Compute sum of gradients from model parameters to loss
#     return list(grad(loss, list(model.parameters()), create_graph=True))

def calc_bias(target_set, attribute_set, model):
    targets_one = target_set[0]
    targets_two = target_set[1]
    attribute_one = attribute_set[0]
    attribute_two = attribute_set[1]
    mean_one = torch.zeros(len(targets_one))
    mean_two = torch.zeros(len(targets_two))
    std_all = torch.zeros(len(targets_one)+len(targets_two))
    ind=0
    for x, y in zip(targets_one, targets_two):
        m1 = similarity_diff(x, attribute_one, attribute_two, model)
        m2 = similarity_diff(y, attribute_one, attribute_two, model)
        mean_one[ind] = m1
        mean_two[ind] = m2
        std_all[ind*2] = m1
        std_all[ind*2 + 1] = m2
        ind += 1
    return (mean_one.mean() - mean_two.mean()) / std_all.std()

def similarity_diff(word, attrs_A, attrs_B, model):
    cos_attr_one = torch.zeros(len(attrs_A), requires_grad=True)
    cos_attr_two = torch.zeros(len(attrs_B), requires_grad=True)
    ind = 0
    for a_A, a_B in zip(attrs_A, attrs_B):
        cos_attr_one[ind] = cos_diff(word, a_A, model)
        cos_attr_two[ind] = cos_diff(word, a_B, model)
        ind += 1
    return cos_attr_one.mean() - cos_attr_two.mean()


def cos_diff(x, y, model):
    return torch.nn.functional.cosine_similarity(model.predict(x), model.predict(y))


def calc_loss(y, t):
    """Calculates the loss
    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)
    Returns:
        loss: scalar, the loss"""
    loss = torch.nn.functional.mse_loss(y, t, reduction='mean')  # TODO: Test cosine loss... but clustering doesn't use that
    return loss


def hvp(ys, xs, v, sparse=False):
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
    first_grads = grad(ys, xs, create_graph=True)  # , retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        if not sparse:
            if grad_elem is not None and v_elem is not None:
                elemwise_products += torch.sum(grad_elem * v_elem.detach())
        else:
            if (grad_elem*v_elem).indices().size(1) > 0:
                elemwise_products += (grad_elem * v_elem).values().sum()
    # Second backprop
    if elemwise_products == 0:
        return False
    return_grads = grad(elemwise_products, xs)  # , create_graph=True)

    return return_grads


def vectorized_influence_data_to_tensors(batch_data_in):
    # np_batch = np.array(batch_data_in)
    # all_words = torch.tensor(np_batch[:, 0].tolist(), dtype=torch.long, device=DEVICE).view(-1)
    # all_contexts = torch.tensor(np_batch[:, 1].tolist(), dtype=torch.long, device=DEVICE).view(-1)
    all_words = torch.tensor([x[0] for x in batch_data_in], dtype=torch.long, device=DEVICE).view(-1)
    all_contexts = torch.tensor([x[1] for x in batch_data_in], dtype=torch.long, device=DEVICE).view(-1)
    return all_words, all_contexts


def load_skipgram(fn, device):
    checkpoint = torch.load(os.path.join(MODEL_SAVE_DIR, PROJECT_NAME, 'checkpoints', fn), map_location='cpu')
    vocab_size = checkpoint['model_data']['u_embeddings.weight'].size(0)
    embed_dim = checkpoint['model_data']['u_embeddings.weight'].size(1)
    sparse = checkpoint['model_data']['u_embeddings.weight'].is_sparse
    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embed_dim, sparse=sparse).to(device=device)
    model.load_state_dict(checkpoint['model_data'])
    return model


def load_prior_work_embeddings(model, fn):
    embeds, words = extract_txt_embeddings(fn)
    model.u_embeddings.weight.data = torch.from_numpy(embeds).float().to(DEVICE)
    return model


def explain_embed_pos(word='mandalorians',
                      model_name='DENSE_scifi_window-3_negatives-5',
                      data_fn='all_scripts_numericalized_dataset.pkl',
                      vocab_fn='all_scripts_stoi.pkl',
                      window_size=5,
                      r=1,
                      recursion_depth=500):
    dataset = InfluenceMedDataset(data_dir=DATA_DIR,
                                  filename=data_fn,
                                  window_size=window_size)
    stoi = pickle.load(open(os.path.join(DATA_DIR, vocab_fn), 'rb'))
    all_keys = []
    all_values = []
    for key, val in stoi.items():
        all_keys.append(key)
        all_values.append(val)
    mando_ind = torch.tensor([all_values[all_keys.index(word)]], dtype=torch.long).to(device=DEVICE)

    # checkpoint = torch.load(os.path.join(MODEL_SAVE_DIR, PROJECT_NAME, 'checkpoints',
    #                                      'DENSE_medkd_window-5_negatives-10_last_checkpoint.pth.tar'), map_location='cpu')
    model = load_skipgram(model_name+'_init_checkpoint.pth.tar', device)
    model.eval()
    mando_init = model.predict(mando_ind)

    model = load_skipgram(model_name+'_last_checkpoint.pth.tar', device)
    model.eval()
    mando_final = model.predict(mando_ind)
    test_index = mando_ind.item()

    # If in sparse, r = 1. else r = 120 ??
    # If in sparse, recursion_depth = 1, else recursion_depth = 10000 ??
    influence, harmful, helpful = calc_influence_single(model=model,
                                                        train_dataset=dataset,
                                                        z_test=mando_final, t_test=mando_init,
                                                        recursion_depth=recursion_depth,
                                                        r=r,
                                                        test_indices=[test_index],
                                                        scifi=True)
    return influence, harmful, helpful, dataset, all_keys


def explain_bias(test='math',
                 model_name='DENSE_biased_window-10_negatives-10_60_checkpoint.pth.tar',
                 data_fn='biased_data_numericalized_dataset.pkl',
                 vocab_fn='biased_data_stoi.pkl',
                 window_size=10,
                 recursion_depth=5000,
                 r=10,
                 other_embeds=None):

    dataset = InfluenceMedDataset(data_dir=DATA_DIR,
                                  filename=data_fn,
                                  window_size=window_size)

    stoi = pickle.load(open(os.path.join(DATA_DIR, vocab_fn), 'rb'))
    all_keys = []
    for key, val in stoi.items():
        all_keys.append(key)
    if test == 'math':
        A = ['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition']
        B = ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture']
        X = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
        Y = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
    elif test == 'science':
        A = ['science', 'technology', 'physics', 'chemistry', 'einstein', 'nasa', 'experiment', 'astronomy']
        B = ['poetry', 'art', 'shakespeare', 'dance', 'literature', 'novel', 'symphony', 'drama']
        X = ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him']
        Y = ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'her', 'hers']
    elif test == 'career':
        X = ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill']
        Y = ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna']
        A = ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career']
        B = ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']
    elif test == 'race':
        A = ['freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond',
             'gentle', 'honest', 'lucky', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family',
             'happy', 'laughter', 'vacation']
        B = ['crash', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink',
             'assault', 'disaster', 'hatred', 'tragedy', 'bomb', 'divorce', 'jail', 'poverty', 'ugly',
             'cancer', 'evil', 'kill', 'rotten', 'vomit']
        X = ['josh', 'alan', 'ryan', 'andrew', 'jack', 'greg', 'amanda', 'katie', 'nancy', 'ellen']
        Y = ['theo', 'jerome', 'leroy', 'lamar', 'lionel', 'malik', 'tyrone', 'ebony', 'jasmine', 'tia', ]
    x_embed = [torch.tensor([all_keys.index(x)], dtype=torch.long).to(device='cpu') for x in X]
    y_embed = [torch.tensor([all_keys.index(x)], dtype=torch.long).to(device='cpu') for x in Y]
    a_embed = [torch.tensor([all_keys.index(x)], dtype=torch.long).to(device='cpu') for x in A]
    b_embed = [torch.tensor([all_keys.index(x)], dtype=torch.long).to(device='cpu') for x in B]
    test_indices = np.unique(x_embed+y_embed+a_embed+b_embed).tolist()
    model = load_skipgram(model_name, device)
    x_embed = [x.to(device=DEVICE) for x in x_embed]
    y_embed = [x.to(device=DEVICE) for x in y_embed]
    a_embed = [x.to(device=DEVICE) for x in a_embed]
    b_embed = [x.to(device=DEVICE) for x in b_embed]
    z_test = [x_embed, y_embed]
    t_test = [a_embed, b_embed]

    if other_embeds is not None:
        model = load_prior_work_embeddings(model, other_embeds)

    influence, harmful, helpful = calc_influence_single(model=model,
                                                        train_dataset=dataset,
                                                        z_test=z_test, t_test=t_test,
                                                        recursion_depth=recursion_depth,
                                                        r=r,
                                                        test_indices=test_indices,
                                                        scifi=False)
    return influence, harmful, helpful, dataset, all_keys


def write_out_results(influences, harmfuls, helpfuls, dataset, keys, file_prefix=''):
    f1 = os.path.join(DATA_DIR, file_prefix+"test_results.txt")
    with open(f1, 'w') as f:
        f.write(f"Influences: {influences}")
        f.write('\n')
        f.write(f"Harmful: {harmfuls}")
        f.write('\n')
        f.write(f"Helpful: {helpfuls}")
        f.write('\n')

    f2 = os.path.join(DATA_DIR, file_prefix+'harmful_ordered.txt')
    for h in harmfuls:
        bad_indices = np.array(dataset.get_raw_sample(int(h[0])))
        b_ws = ''
        for index in range(len(bad_indices)):
            try:
                b_ws += keys[bad_indices[index]]
                b_ws += ' '
            except:
                continue
        with open(f2, 'a', encoding='utf-8') as f:
            f.write(str(b_ws))
            f.write('\n')

    f3 = os.path.join(DATA_DIR, file_prefix+'helpful_ordered.txt')
    for h in helpfuls:
        good_indices = np.array(dataset.get_raw_sample(int(h[0])))
        g_ws = ''
        for index in range(len(good_indices)):
            try:
                g_ws += keys[good_indices[index]]
                g_ws += ' '
            except:
                continue
        with open(f3, 'a', encoding='utf-8') as f:
            f.write(str(g_ws))
            f.write('\n')


if __name__ == "__main__":
    # TEST THIS STUFF
    from cross_loss_influence.models.skip_gram_word2vec import SkipGramModel
    from cross_loss_influence.data.scripts.my_pytorch_dataset import InfluenceMedDataset
    from cross_loss_influence.config import DATA_DIR, MODEL_SAVE_DIR, PROJECT_NAME
    import pickle
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", help="biased or neutral", type=str, default='biased')
    parser.add_argument("-t", "--test", help="test? ['career', 'science', 'math', 'race']", type=str, default='career')
    parser.add_argument("-m", "--model_name", help="Model name to investigate", type=str, required=True)
    parser.add_argument("-w", "--word", help="Word for the scifi tests", type=str)
    args = parser.parse_args()
    original = args.base
    test_name = args.test
    model_name = args.model_name
    word = args.word
    # start_time = time.time()

    w_s = 3
    device = DEVICE
    rec = 10  # 1
    r_depth = 5000 # 1
    if test_name == 'scifi':
        rec = 1
        r_depth = 5000
        assert len(word) > 0, "-w or --word must be passed in"
        i_out, harm_out, help_out, dset, keyset = explain_embed_pos(word=word,
                                                                    model_name=model_name,
                                                                    data_fn='all_scripts_numericalized_dataset.pkl',
                                                                    vocab_fn='all_scripts_stoi.pkl',
                                                                    window_size=w_s,
                                                                    r=rec,
                                                                    recursion_depth=r_depth)
        write_out_results(i_out, harm_out, help_out, dset, keyset,
                          file_prefix=f'{test_name}_{original}_influence_scifi')
    else:
        prior_work_fn = None
        if 'bolukbasi' in model_name:
            prior_work_fn = model_name
            model_name = model_name.split('original_')[-1].split('_debiased')[0]

        # prior_work_fn = os.path.join(DATA_DIR, f'bolukbasi_original_{model_name}_debiased.txt')

        print(f"Beginning influence estimation on {original} embeddings with {test_name} WEAT")
        i_out, harm_out, help_out, dset, keyset = explain_bias(test=test_name,
                                                               model_name=model_name,
                                                               data_fn=f'{original}_data_numericalized_dataset.pkl',
                                                               vocab_fn='biased_data_stoi.pkl',
                                                               window_size=10,
                                                               recursion_depth=r_depth,
                                                               r=rec,
                                                               other_embeds=prior_work_fn)

        write_out_results(i_out, harm_out, help_out, dset, keyset,
                          file_prefix=f'{test_name}_{original}_{model_name}_influence_')
