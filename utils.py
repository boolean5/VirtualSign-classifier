def correct_dict():
    replacements = {'Ã£': 'ã', 'Ã¡': 'á', 'Ã¢': 'â', 'Ã§': 'ç', 'Ãª': 'ê', 'Ã©': 'é',
                    'Ã³': 'ó', 'Ã­': 'í', 'Ãº': 'ú', 'Ã': 'Á', 'Ãµ': 'õ'}

    with open('data/datasetLeft.txt') as infile, open('data/dictLeft-corrected', 'w') as outfile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(src, target)
            outfile.write(line)


def duplicates(dataframe):
    import numpy as np

    num_dups = np.flatnonzero(dataframe.duplicated())
    return len(num_dups)


def create_dataset(path, deletedups=False, randomize=True):
    import pandas as pd
    import os

    sensors = ['col'+str(i) for i in range(14)] + ['id']
    frames = []

    for root, dirs, files in os.walk(path):
        for filename in files:
            with open(os.path.join(root, filename)) as infile:
                df = pd.read_csv(infile, delim_whitespace=True, names=sensors, usecols=sensors, header=None)

            # Check for misplaced class-label column, properly swap columns/shift contents if so
            if df.iloc[:, 0].dtype == 'int64':
                print('Rearranging data in ' + filename)

                # Reorder columns, then change back to original names
                cols = df.columns.tolist()
                cols = cols[1:] + cols[:1]
                df = df[cols]
                df.columns = sensors
            frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)

    if deletedups:
        dataset = dataset.drop_duplicates().reset_index(drop=True)

    if randomize:
        dataset = dataset.sample(frac=1).reset_index(drop=True)

    return dataset
