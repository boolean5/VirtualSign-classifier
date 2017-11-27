# VirtualSign

### Dataset format:
Datasets consist of 18 columns. The first one determines the hand configuration. The rest correspond to the data gloves sensors input.  Each hand configuration is measured 10 times.

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
- [x] Test L2 regularization
- [x] Check decision trees and gradient boosting
- [x] Add evaluation in the end of the training script
- [x] train, train-val, val, test
- [x] Try L1 regulization (though the sparsity of the features is not certain)
- [x] decide stopping strategy (try early stopping)
- [x] explore how linearly separable is our data and try SVMs
- [x] consider separating the knuckle inputs with the finger inputs
- [x] integrate in Virtual Sign
- [x] Check unsupervised pre-training
- [ ] hyper-parameter search script (check hyperas)
- [ ] upload graph of the models
- [ ] add the SVM classifier to Virtual Sign
- [ ] use the new model of gloves: add pitch, roll and yaw inputs, adapt calibration process, output 57 instead of 42 classes
- [ ] capture new dataset

