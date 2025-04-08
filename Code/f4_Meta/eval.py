import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

bsize = 500
dat1 = np.genfromtxt('testset070224.csv', delimiter=',', dtype=None)
dat1 = np.vstack(dat1)
nodenumber = np.shape(dat1)[1]
runcount = 'e1'
width = 1
depth = 1
param1 = './20_LazyNet1_trainset_1X1Xe200bsize5@0.0001dk0.0001dv0.1_nature_cuda1.pth'

class LazyNet1(nn.Module):
    def __init__(self, width, depth):
        super(LazyNet1, self).__init__()
        self.nodenumber = nodenumber
        self.depth = depth
        self.denselog_layers = nn.ModuleList([nn.Linear(nodenumber, nodenumber * width) if i == 0 else nn.Linear(nodenumber * width, nodenumber * width) for i in range(depth)])
        self.denseexp_layers = nn.ModuleList([nn.Linear(nodenumber * width, nodenumber * width) for i in range(depth)])
        self.linear1 = nn.Linear(nodenumber * width, nodenumber)
        self.output = nn.Linear(nodenumber, nodenumber)

    def forward(self, x):
        layer_output = x
        for i in range(self.depth):
            dense_log_output = self.denselog_layers[i](layer_output)
            log_layer = torch.log(dense_log_output.clamp(min=1e-6))
            dense_exp_output = self.denseexp_layers[i](log_layer)
            exp_layer = torch.exp(dense_exp_output)
            layer_output = exp_layer
        dense_no_activation = self.linear1(layer_output)
        addition_layer = x + dense_no_activation
        output = addition_layer
        return output

model = LazyNet1(width, depth)
model.load_state_dict(torch.load(str(param1)))
print(model)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < len(self.data):
            input_row = self.data[idx]
            input_tensor = torch.tensor(input_row, dtype=torch.float32)
            return input_tensor
        else:
            raise IndexError("Index out of range")
test_dataset = MyDataset(dat1)
test_loader = DataLoader(test_dataset, batch_size=bsize)

individual_losses = []
all_labels = []
all_predictions = []
criterion = nn.HuberLoss()

model.eval()
with torch.no_grad():
    for data in test_loader:
        initial_input = data[0].unsqueeze(0)
        targets = data[1:]
        current_input = initial_input
        predictions = []
        for _ in range(targets.size(0)):
            output = model(current_input)
            predictions.append(output.squeeze(0))
            current_input = output
        predictions = torch.stack(predictions)
        all_labels.append(targets)
        all_predictions.append(predictions)
        batch_loss = criterion(predictions, targets.unsqueeze(0))
        individual_losses.append(batch_loss.item())

loss_array = np.array(individual_losses)
median_loss = np.nanmedian(loss_array)
avg_loss = np.mean(loss_array)

all_labels = torch.cat(all_labels).cpu().numpy()
all_predictions = torch.cat(all_predictions).cpu().numpy()

num_genes = all_predictions.shape[1]
num_genes_to_plot = num_genes  # Assuming this is 80
num_time_points = all_predictions.shape[0]
time = np.arange(1, num_time_points + 1)
df = pd.read_csv('./metabolomics.csv')
df = df[df.index.notnull() & (df.index != '')]
df = df.loc[~(df == 0).all(axis=1)]
column_names = df.columns

molecules_to_plot = [1, 2, 3, 6, 20, 21, 22, 23, 24, 25, 27, 28, 30, 34,
                     35, 38, 44, 45, 46, 47, 48, 49, 51, 52, 53, 56, 57, 59,
                     60, 62, 68, 69, 77]  # Indices of molecules to plot

num_selected = len(molecules_to_plot)
fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(80, 40))  # Increase figure size for clarity
axes = axes.flatten()  # Flatten the 2D array to a 1D array

for idx, molecule_idx in enumerate(molecules_to_plot):
    ax = axes[idx]
    ax.plot(time, all_labels[:, molecule_idx], label='True Answer', color='blue')
    ax.plot(time, all_predictions[:, molecule_idx], '--', label='Prediction', color='red', linewidth=2)
    ax.set_title(f'{column_names[molecule_idx]}', fontsize=10)
    ax.set_xlabel('Time Points', fontsize=8)
    ax.set_ylabel('Values', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='x', labelrotation=45)

# Turn off any unused subplots (if there are any left over)
for j in range(idx + 1, len(axes)):
    axes[j].axis('off')

# Add a legend to the first subplot
axes[0].legend()
plt.tight_layout(pad=1)
plt.show()


# For the RMSE bar plot, restrict to the selected molecules
# 'rmse_per_gene' should have been computed previously (see your code)
selected_rmse = rmse_per_gene[molecules_to_plot]
selected_names = [column_names[i] for i in molecules_to_plot]

plt.figure(figsize=(10, 6))
plt.bar(selected_names, selected_rmse, color='green')
plt.xlabel('Molecules')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error for Selected Molecules')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

scaler = MinMaxScaler()
all_labels = np.concatenate(all_labels)
all_predictions = np.concatenate(all_predictions)
labels_scaled = scaler.fit_transform(all_labels.reshape(-1, 1)).flatten()
predictions_scaled = scaler.transform(all_predictions.reshape(-1, 1)).flatten()
rmse = np.sqrt(np.mean((predictions_scaled - labels_scaled) ** 2))
print("Overall RMSE:", rmse)

selected_all_labels = all_labels[:, molecules_to_plot]
selected_all_predictions = all_predictions[:, molecules_to_plot]

# --- Compute Normalized RMSD ---
# Here, we use MinMax scaling to bring the values to [0,1] before computing RMSE
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Flatten the selected data for scaling
selected_labels_flat = selected_all_labels.flatten()
selected_predictions_flat = selected_all_predictions.flatten()

labels_scaled = scaler.fit_transform(selected_labels_flat.reshape(-1, 1)).flatten()
predictions_scaled = scaler.transform(selected_predictions_flat.reshape(-1, 1)).flatten()

selected_norm_rmsd = np.sqrt(np.mean((predictions_scaled - labels_scaled) ** 2))
print("Normalized RMSD for selected molecules:", selected_norm_rmsd)


# --- Compute AUC ---
from sklearn.metrics import roc_curve, auc

# Define a threshold to binarize the continuous outputs (adjust as needed)
threshold = 0.5
binary_labels_selected = (selected_all_labels > threshold).astype(int)
binary_predictions_selected = (selected_all_predictions > threshold).astype(int)

# Compute ROC curve and AUC by flattening the arrays
fpr_sel, tpr_sel, _ = roc_curve(binary_labels_selected.ravel(), binary_predictions_selected.ravel())
selected_auc = auc(fpr_sel, tpr_sel)
print("AUC for selected molecules:", selected_auc)


# (Optional) Print total number of trainable parameters (this is model-wide)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable parameters:", total_params)

from scipy.stats import pearsonr
corr, _ = pearsonr(selected_all_labels.flatten(), selected_all_predictions.flatten())
print("Pearson Correlation:", corr)