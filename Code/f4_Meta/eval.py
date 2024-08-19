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
num_genes_to_plot = num_genes
num_time_points = all_predictions.shape[0]
time = np.arange(1, num_time_points + 1)
df = pd.read_csv('./metabolomics.csv')
df = df[df.index.notnull() & (df.index != '')]
df = df.loc[~(df == 0).all(axis=1)]
column_names = df.columns
fig, axes = plt.subplots(nrows=10, ncols=8, figsize=(40, 30))
axes = axes.flatten()
for i in range(num_genes_to_plot):
    ax = axes[i]
    ax.plot(time, all_labels[:, i], label='True Answer', color='blue')
    ax.plot(time, all_predictions[:, i], '--', label='Prediction', color='red', linewidth=2)
    ax.set_title(f'{column_names[i]}', fontsize=10)
    ax.set_xlabel('Time Points', fontsize=5)
    ax.set_ylabel('Values', fontsize=5)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.tick_params(axis='x', labelrotation=45)

axes[0].legend()
plt.tight_layout(pad=1)
plt.show()

scaler = StandardScaler()
labels_scaled = scaler.fit_transform(all_labels)
predictions_scaled = scaler.fit_transform(all_predictions)
rmse_per_gene = np.sqrt(np.mean((predictions_scaled - labels_scaled)**2, axis=0))
plt.figure(figsize=(10, 6))
plt.bar(column_names, rmse_per_gene, color='green')
plt.xlabel('Molecules')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error for Each Molecule Across All Time Points')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

scaler = MinMaxScaler()
all_labels = np.concatenate(all_labels)
all_predictions = np.concatenate(all_predictions)
labels_scaled = scaler.fit_transform(all_labels.reshape(-1, 1)).flatten()
predictions_scaled = scaler.transform(all_predictions.reshape(-1, 1)).flatten()
rmse = np.sqrt(np.mean((predictions_scaled - labels_scaled) ** 2))
print("Overall RMSE:", rmse)