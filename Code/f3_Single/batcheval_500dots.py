import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

bsize = 500
nodenumber = 8
width = 2
depth = 1
param1 = './82_LazyNet1_trainset062624_1_2X1Xe16bsize20@0.0001dk0.0001_nature_cuda1.pth'
model_name = param1.split('/')[-1].split('.')[0]
output_folder = f'./500dots/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_row = self.data[idx]
        input_tensor = torch.tensor(input_row, dtype=torch.float32)
        return input_tensor

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

num_files = 50
for file_num in range(1, num_files + 1):
    file_path = f'./testset063024_1/testset063024_{file_num}.txt'
    dat = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=None)
    dat = np.vstack(dat)
    dat = dat[:bsize,1:]

    test_dataset = MyDataset(dat)
    test_loader = DataLoader(test_dataset, batch_size=bsize)

    all_labels = []
    all_predictions = []

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

    all_labels = torch.cat(all_labels).cpu().numpy()
    all_predictions = torch.cat(all_predictions).cpu().numpy()

    num_time_points = all_predictions.shape[0]
    time = np.arange(1, num_time_points + 1)

    num_equations = 8
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 15))
    axes = axes.flatten()
    for i in range(num_equations):
        ax = axes[i]
        ax.plot(time, all_labels[:, i], label='True Answer', color='blue')
        ax.plot(time, all_predictions[:, i], '--', label='Prediction', color='red', linewidth=2)
        ax.set_title(f'Equation {i + 1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Values')
        ax.legend()

    plt.tight_layout(pad=0.2)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    fig.savefig(f'{output_folder}/model_predictions_{file_num}.png')
    plt.close()

# --- Compute Normalized RMSD ---
# Here, we use MinMax scaling to bring the values to [0,1] before computing RMSE
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Flatten the selected data for scaling
all_labels_flat = all_labels.flatten()
all_predictions_flat = all_predictions.flatten()

labels_scaled = scaler.fit_transform(all_labels_flat.reshape(-1, 1)).flatten()
predictions_scaled = scaler.transform(all_predictions_flat.reshape(-1, 1)).flatten()

all_norm_rmsd = np.sqrt(np.mean((predictions_scaled - labels_scaled) ** 2))
print("Normalized RMSD for selected molecules:", all_norm_rmsd)


# --- Compute AUC ---
from sklearn.metrics import roc_curve, auc

# Define a threshold to binarize the continuous outputs (adjust as needed)
threshold = 0.5
binary_labels_selected = (all_labels > threshold).astype(int)
binary_predictions_selected = (all_predictions > threshold).astype(int)

# Compute ROC curve and AUC by flattening the arrays
fpr_sel, tpr_sel, _ = roc_curve(binary_labels_selected.ravel(), binary_predictions_selected.ravel())
all_auc = auc(fpr_sel, tpr_sel)
print("AUC for selected molecules:", all_auc)


# (Optional) Print total number of trainable parameters (this is model-wide)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable parameters:", total_params)

from scipy.stats import pearsonr
corr, _ = pearsonr(all_labels.flatten(), all_predictions.flatten())
print("Pearson Correlation:", corr)