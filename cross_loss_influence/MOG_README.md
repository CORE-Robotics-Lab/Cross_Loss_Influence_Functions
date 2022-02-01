First, generate the MoG Data:
`data/scripts: generate_mog_data.py`.
Make sure that the DATA_DIR and MODEL_SAVE_DIR are emptied out to begin with.

Next, train a Deep Embedded Clusering model: `runfiles/train_dec.py`. This will create the initial and final models. Make sure you create a new initial model before other experiments here. (save=True)

Finally, you can estimate the influence and calculate the empirical influence. To estimate influence, use `helpers/mog_influence_function.py` and to calculate empirical influence, use: `runfiles/retrain_mogs.py`.

Once you've calculated the influences and your baseline, you can plot correlations and get results with: `helpers/plot_mog_biases.py`.

Be aware that training/testing with matched influence functions requires manually setting a few flags in the above files. In `runfiles/train_dec.py`, you must set `nll=True` for matched (telling the network to use the negative log-likelihood loss). Then, in `helpers/mog_influence_function.py`, again set `nll=True` to use the same objective for influence calculation. Finally, calculating the empirical influence, set `nll=True` in the `runfiles/retrain_mogs.py` script.
