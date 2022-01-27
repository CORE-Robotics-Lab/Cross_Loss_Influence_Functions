# Influence Functions
This repo houses the code to identify influential examples in embedding models.

### Setup:
This a pip install of itself, as well as datasets and external libraries. You probably also want to edit your config.json:
```
{
    "data_directory": "DATA SAVE DIRECTORY",
    "project_home": "/home/user/interimstuff/cross_loss_influence/",
    "model_save_directory": "/media/user/models",
    "project_name": "inf_we",
}
```
This file is explicitly ignored in the `.gitignore`.

Requirements are listed within `requirements.txt`. For only the MoG experiments, you will likely only need the usual scientific computing packages (numpy, scipy, pytorch, etc). For the WEAT/Sci-Fi experiments, you may need mat2vec which you can get [here](https://github.com/materialsintelligence/mat2vec). Follow the entire installation guide before coming back here and running the remainder of your pip install.

For the pip install, move the `setup.py` to this project's parent directory, navigate to the parent directory of this and run:
```
pip install -e .
```
You _also_ need to modify `config.py` in the main directory (here) to reflect your machine. So mine says `/home/myname/Documents/Cross_Loss_Influence_Functions/cross_loss_influence/config.json`, yours should point to your machine.

Finally, the Cython needs to be built. Navigate to `cross_loss_influence/data/scripts` and run:
```
python setup.py build_ext --inplace
```
You will hopefully see a successful build of the Cython files. If they aren't being loaded properly, you'll need to fix that particular import statement in `data/scripts/my_pytorch_dataset.py`.


### Mixture-of-Gaussians experiments:

There is a single demo script to automate all of the below. Simply run:

```
cd runfiles
python mog_demo.py
```
There are 2 command line args for this:
* `-p` or `--data_path`: Where would you like results/data to be stored?
* '-r' or `--re_use`: Should we re-use an already-generated MoG result? (only add this if you've already run the script at least once)

To manually step through the entire script, follow the steps in `MOG_README.md`

### Data:
#### WNC:
For the [WNC data](https://arxiv.org/abs/1911.09709v1), Our dataset is available here: https://drive.google.com/file/d/1KQLxwU5Yh-Ora7ri_G09iYFIplyrsq-G/view?usp=sharing . Just put it in this directory.

I have used their set of tokens and created a string-to-index (stoi -- `biased_data_stoi.pkl`) and index-to-string (itos -- `biased_data_itos.pkl`) file. These are pickle files, when you open them up you'll just see a bunch of strings on newlines. The index of the newline is the index of that string in the dataset. The raw strings have all been converted into their appropriate numericalized counterparts in the `biased_data_numericalized_dataset.pkl` file. For the neutral half of the corpus, the data is numericalized in `neutral_data_numericalized_dataset.pkl`, though notably both use the same itos/stoi files. If you need the raw data, you can get it [here](https://github.com/rpryzant/neutralizing-bias#data).
#### Scifi:
For the [SciFi dataset](https://www.aclweb.org/anthology/W19-3405/), the dataset I used is here: https://drive.google.com/file/d/1i4HOu6eilYBOUYVhkMbl6krj0a0d7sHz/view?usp=sharing . For the original, contact the original authors.
### Running things:

Training scripts are located in `runfiles`. In there, run:
```
python train_skipgram.py -c 4 -n 15 -b 1000 -e 35 -w 0 -l 1e-2
```
To train a Word2Vec model on the biased dataset with a **c**ontext window of 4, 15 **n**egative samples, a **b**atch size of 1000, 35 **e**pochs, 0 parallelized **w**orkers, and a **l**earning rate of 1e-2.

Also this will try to load a previously saved model to continue training if the `model_init` file doesn't exist.

Once you have a trained model, you'll want to identify the influential samples for different things. That is done in `helpers/influence_function.py`.

You specify the test that you want to run in line 474, options are `career`, `math`, `science`, or `race`. In the paper, all results for `art` are gathered by running the `science` test and negating the results (changing terminology basically). 
In line 476 you specify the model you want to test (which checkpoint or final model), then in 477 you choose the dataset (neutral_numericalized or biased_numericalized). The vocab file can stay the same for both (as they share a vocab).

To run the MSE test, run with `--test scifi`. This will swap out the model file, data file, vocab file, and write-out files automatically. To change the word you want to test, you'll need to manually edit the `word='dooku'` argument in line 492.

That writes out results to different text files, and you then need to order them, extract the indices, and chop the files down to manageable size. There is a script for this: `helpers/write_results_given_inf.py`. There is also a notebook for this, in which you must exchange the filename at the top of the notebook and just re-run all cells. The notebook is `extract_indices_from_influence_txt.pynb` in the `notebooks` directory. It's also a bit misleading in that `harmful` means things which _reduce_ bias, and helpful means things which _amplify_ bias. Mostly because it's harmful to the loss value (lowers it) or helpful to the loss value (grows it).

With ordered sets in hand, you can undo their effects. This is done in `runfiles/undo_influence.py` There is an `undo_bias=True` parameter setting on line 165, and that's what you change (to be `False`) if you want to do the scifi undoing. Otherwise just leave it as is. The bias model which is being undone is in line 135, and this simultaneously does the undoing, redoing, and doing both. And sweeps over number-samples={5, 10, 100, 1000} and num_iterations={5, 10, 100, 1000}. Saving all models along the way.

Finally, you can evaluate finished models. with `helpers/weat.py`. This will run WEAT scores over a given model and for prior-work debiasing.

Basically:
1. Train your model (`train_skipgram.py`). Saves model to model_save_dir
1. Identify samples of influence on model (`influence_function.py`). Loads models from model_save_dir and writes inluential sample text results to data_dir.
1. Convert those saved samples into a usable format (`helpers/write_results_given_inf.py`, modify lines 9-10). Uses influence text files saved into data_dir.
1. Undo the effects of those samples across a variety of hyperparameter settings (`undo_influence.py` multiprocessing code). Uses saved model in model_save_dir and the influential sample text files in data_dir.
1. Run the WEAT over your model (`weat.py`). Saves results to data_dir.
1. Print out comparisons across tests (`plot_bias_movement.py`). Uses data_dir results.

### Prior Work
Within `helpers/` there is a directory called `bolukbasi_prior_work` that has files cloned from the bolukbasi prior work directory, which I use to debias my word embeddings. That is done in the `prior_pca_debiasing.py` file, which saves the debiased embeddings to the data directory specified in the config.json. These can then be loaded into a saved model with a function in the `influence_fuction.py` script. 


