import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def expand_and_smooth_dataset(df):
    expanded_subsets = []
    for start in range(0, len(df), 7):
        end = start + 7
        subset = df.iloc[start:end]
        x_original = np.arange(0, 7)
        x_new = np.linspace(0, 6, 140)
        expanded_data = []

        for col in subset.columns:
            f = interp1d(x_original, subset[col], kind='quadratic', fill_value="extrapolate")
            interpolated_values = f(x_new)
            window_size = 19
            poly_order = 3
            if len(interpolated_values) > window_size:
                smoothed_values = savgol_filter(interpolated_values, window_size, poly_order, mode='nearest')
            else:
                smoothed_values = interpolated_values
            expanded_data.append(smoothed_values)
        expanded_subset = pd.DataFrame(expanded_data).transpose()
        expanded_subsets.append(expanded_subset)
    final_expanded_df = pd.concat(expanded_subsets, ignore_index=True)
    return final_expanded_df

train_df = pd.read_csv('./trainset_bigc_7_053024.csv', index_col=0)
train_df = train_df.transpose()
expanded_train_df = expand_and_smooth_dataset(train_df)
expanded_train_df.to_csv('./trainset140_bigc_quad_7_053024.csv', index = False, header = False)

trainset = expanded_train_df
trainset = trainset.to_numpy()
num_rows = trainset.shape[0]
num_samples = 20000
random_indices = np.random.choice(num_rows, num_samples, replace=False)
final_matrix = trainset[random_indices, :]
df = pd.DataFrame(final_matrix.transpose())
df.to_csv('./Monocyte/trainset_Seurat.csv', header=False, index_label=False)