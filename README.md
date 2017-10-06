# VirtualSign

### Dataset format:
Datasets consist of 15 columns. The first 14 correspond to the data gloves sensors input. The last one determines the hand configuration. Each hand configuration is measured 10 times.

### TODO list:
- [x] review architecture (add multiple filters)
- [x] Visualise w/ tf.tensorboard
- [x] create inferring script
- [x] transfer model building function to utils.py
- [ ] k-fold cross validation test
- [ ] insert new dataset
- [ ] decide stopping strategy (try early stopping)
- [ ] train-val-test
- [ ] hyper-parameter search script
- [ ] evaluation script
- [ ] upload a graph of the model
- [ ] integrate in Virtual Sign
