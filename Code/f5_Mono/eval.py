import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

bsize = 700
testset = pd.read_csv('./trainset140_bigc_quad_7_053024.csv', header=None)
testset = testset.to_numpy()
dat1 = testset
nodenumber = np.shape(dat1)[1]

runcount = 'e1'
width = 1
depth = 2
param1 = './6_LazyNet1_bigquad_1X2Xe8bsize2@1e-06_nature_cuda1.pth'

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
all_predictions = []

model.eval()
with torch.no_grad():
    for data in test_loader:
        initial_input = data[0].unsqueeze(0)
        current_input = initial_input
        predictions = []
        for _ in range(bsize):
            output = model(current_input)
            predictions.append(output.squeeze(0))
            current_input = output
        predictions = torch.stack(predictions)
        all_predictions.append(predictions)

cache_matrix = np.vstack(all_predictions)
num_rows = cache_matrix.shape[0]
indices = np.arange(0, num_rows, 20)
final_matrix = cache_matrix[indices, :]

print(final_matrix.shape)
df = pd.DataFrame(final_matrix.transpose())
df.to_csv('./080324pub_bigc_trainset140_053024_quad_pred_700dp.csv', header=False, index_label=False)