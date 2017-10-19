import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import create_dataset

# Parsing from terminal
parser = argparse.ArgumentParser(description='Dataset exploration: Range & standard deviation heatmaps, '
                                             'hand configuration box plots')
parser.add_argument('dataset_path', help='Path of datasets folder or file')
args = parser.parse_args()
dataset_path = args.dataset_path

dataset = create_dataset(dataset_path, raw=True, deletedups=False)

grouped = dataset.groupby(['id'], as_index=True)
ranges = grouped.apply(lambda g: g.max() - g.min())
stds = grouped.agg(np.std)

sns.set()
# sns.palplot(sns.color_palette("BuGn_r"))
# sns.palplot(sns.color_palette("Blues"))

sns.heatmap(ranges, annot=True, fmt='.2f', linewidths=.1, linecolor='black', cmap="YlGnBu")
plt.title('Range plot')
plt.show()

sns.heatmap(stds, annot=True, fmt='.2f', linewidths=.1, linecolor='black', cmap="YlGnBu")
plt.title('Standard deviation plot')
plt.show()

for name, groups in dataset.groupby(['id']):
    groups.drop('id', axis=1).boxplot()
    plt.title(name)
    plt.show()
