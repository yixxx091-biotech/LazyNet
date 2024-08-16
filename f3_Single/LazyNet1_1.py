import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

runcount = 82
width = 2
depth = 1
LRN = 1e-4
enumber = 16
bsize = 20
dcay = 1e-4

dat1 = np.genfromtxt('./trainset062624_1.txt', delimiter=',', dtype=float)
dat1 = np.vstack(dat1)
dat1 = dat1[:, 1:]
nodenumber = np.shape(dat1)[1]

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < len(self.data):
            row = self.data[idx]
            tensor = torch.tensor(row, dtype=torch.float32)
            return tensor
        else:
            raise IndexError("Index out of range")

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

device = torch.device("cuda")
gnum = (torch.cuda.device_count())
print(device)

dat1 = MyDataset(dat1)
train_loader = DataLoader(dat1, batch_size=bsize)
model = LazyNet1(width, depth).to(device)
print(model)

criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=LRN, weight_decay=dcay)
loss_values = []
average_loss_values = []

nan_epoch_count = 0
best_loss = float('inf')
epochs_no_improve = 0
for epoch in range(enumber):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for inputs in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        current_input = inputs[0].unsqueeze(0)
        all_predictions = [current_input.squeeze(0)]
        for t in range(1, inputs.size(0)):
            output = model(current_input)
            all_predictions.append(output.squeeze(0))
            current_input = output
        all_predictions = torch.stack(all_predictions)
        loss = criterion(all_predictions, inputs)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        running_loss += loss.item()
        num_batches += 1
    average_epoch_loss = running_loss / num_batches*10000
    if torch.isnan(torch.tensor(average_epoch_loss)):
        nan_epoch_count += 1
        if nan_epoch_count > 3:
            print("Terminating training due to NaN average loss in more than 3 consecutive epochs.")
            break
    else:
        nan_epoch_count = 0
    if average_epoch_loss < best_loss:
        best_loss = average_epoch_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= 5:
            print(f"No improvement in the last 5 epochs. Best Loss: {best_loss:.4f}")
            break
    print(f'Epoch [{epoch + 1}/{enumber}], Loss: {average_epoch_loss:.4f}')
    with open('epoch_loss_record.txt', 'a') as file:
        file.write(f'1_Epoch [{epoch + 1}/{enumber}], Loss: {average_epoch_loss:.4f}\n')

modelname = str(runcount)+'_LazyNet1_trainset062624_1_'+str(width)+'X'+str(depth)+'Xe'+str(enumber)+'bsize'+str(bsize)+'@'+str(LRN)+'dk'+str(dcay)+'dval'+str(dval)+'_nature_cuda'+str(gnum)+".pth"
torch.save(model.state_dict(), modelname)