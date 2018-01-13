# Unsupervised-Neural-Machine-Translation

## Update: Jan. 13, 2018
1. **Change epoch to iteration**: In each iteration, perform L1 denoising, L2 denoising, L1-L2 back-translation, and L2-L1 back-translation on **one batch of data**.
2. **Change sentence MAX_LENGTH to 50**: To speed up the training process. Also, the paper use 50 instead of 60. But be sure the clean the old `vocab_{en,fr}.pkl` and `emb_{en,fr}` before training. **DO NOT USE THE OLD SAVED FILES**.
3. **Disable subword embedding**: According to the paper, using  subword embedding does not improve the performance, so don't use subword for the sake of speed.
4. **Add l2_de.train() and l1_de.train() in `train.py`**: Bug fix.

