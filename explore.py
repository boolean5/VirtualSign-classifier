import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import create_dataset

dataset = create_dataset('datasets/higher_precision_datasets')

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
