# VirtualSign

### Dataset format:
Datasets consist of 15 columns. The first 14 correspond to the data gloves sensors input. The last one determines the hand configuration. Each hand configuration is measured 10 times.

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
- [ ] hyper-parameter search script
- [ ] insert new dataset
- [ ] decide stopping strategy (try early stopping)
- [ ] train, train-val, val, test
- [ ] consider separating the knuckle inputs with the finger inputs
- [ ] upload a graph of the model
- [ ] integrate in Virtual Sign
