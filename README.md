# VirtualSign

### Dataset format:
Datasets consist of 15 columns. The first one determines the hand configuration. The rest correspond to the data gloves sensors input.  Each hand configuration is measured 10 times.

### TODO list:
- [x] review architecture (add multiple filters)
- [x] Visualise w/ tf.tensorboard
- [x] create inferring script
- [x] transfer model building function to utils.py
- [x] k-fold cross validation test
- [x] create a model with an intermediate numper of parameters to 25k - 275k
- [x] rounding script for datasets
- [x] name columns appropriately
- [x] implement exploration/visualization scripts
- [x] batch evaluation script
- [x] use argparse to get inputs in all scripts
- [x] modeling name scheme (model name, val loss, epoch)
- [x] change script to work with directories or files
- [x] Prompt the user to calibrate and then turn off auto-calibration
- [x] Insert new dataset
- [x] Batch normalization
- [ ] Add evaluation in the end of the training script
- [ ] Test L2 regularization
- [ ] Check unsupervised pre-training
- [ ] Check decision trees and gradient boosting
- [ ] Try SVMs
- [ ] consider separating the knuckle inputs with the finger inputs
- [ ] train, train-val, val, test
- [ ] decide stopping strategy (try early stopping)
- [ ] hyper-parameter search script (check hyperas)
- [ ] upload a graph of the model
- [ ] integrate in Virtual Sign

