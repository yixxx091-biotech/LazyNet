import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

df = pd.read_csv('./metabolomics.csv')
df = df[df.index.notnull() & (df.index != '')]
df = df.loc[~(df == 0).all(axis=1)]

final_df = pd.DataFrame()
window_size = 14
poly_order = 3

original_time = df.iloc[:, 0]

for start in range(0, len(df), 14):
    end = start + 14
    segment = df.iloc[start:end]
    time_interp = np.linspace(original_time.iloc[start], original_time.iloc[end - 1], 500)
    expanded_segment = pd.DataFrame(index=time_interp)

    for col in segment.columns:
        y = segment[col].interpolate()
        y_smooth = savgol_filter(y, window_size, poly_order, mode='nearest')
        f = interp1d(original_time[start:end], y_smooth, kind='quadratic', fill_value='extrapolate')
        expanded_segment[col] = f(time_interp)

    final_df = pd.concat([final_df, expanded_segment])

    trainset_df = final_df.iloc[:4000, :]
    testset_df = final_df.iloc[4000:, :]

trainset_df.fillna(0, inplace=True)
testset_df.fillna(0, inplace=True)
trainset_df.to_csv('./trainset070224.csv', index = False, header = False)
testset_df.to_csv('./testset070224.csv', index = False, header = False)
