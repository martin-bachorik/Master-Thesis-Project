# Standard libraries
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torch_optimizer as opt

# Local libraries
from SourceCode.simulation.nn.nn_models import *
from SourceCode.simulation.nn.preprocessing import data_preparation, split_recurrent_sequence
from SourceCode.simulation.templates import templates as tmp


def loader(X_data, y_data):
    data = torch.utils.data.TensorDataset(X_data, y_data)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    return data_loader


# Hyper Parameters
batch_size = 256
num_epochs = 10

seq_length = 5
input_dim = 2
output_dim = 1

hidden_dim = 10
layer_dim = 2


inx = input_dim - output_dim  # Inputs to the network except the input-outputs

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)  # instantiate model class

criterion = nn.modules.loss.MSELoss()  # Mean Squared Error loss functions
# optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

model_parameters = sum(p.numel() for p in model.parameters())
print("lstm: {}".format(model_parameters))

scaled_data = data_preparation("../../../data/MultiStep/T10s/gauss_closed_loop_data.csv", scale="Standard")
sc1 = scaled_data['scale']

all_time = scaled_data['t']
all_data = scaled_data['u_y']

h = int(len(all_time) / 2)

# Training data
time1 = all_time[h:]
train_data = all_data[h:]
# Testing data
time2 = all_time[-h:]
test_data = all_data[-h:]
uncaled_test_data = sc1.inverse_transform(test_data)

# Train data
X, y = split_recurrent_sequence(train_data, seq_length, output_dim)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
# Test data
X2, y2 = split_recurrent_sequence(test_data, seq_length, output_dim)
X2 = torch.tensor(X2, dtype=torch.float32)
y2 = torch.tensor(y2, dtype=torch.float32)
# Loading batches
train_loader = loader(X, y)
test_loader = loader(X2, y2)

j = 0
save_train_loss = list()
save_test_loss = list()


for epoch in range(num_epochs):
    for in_x, out_y in train_loader:
        inputs_xb = torch.clone(in_x.view(-1, seq_length, input_dim))
        targets_yb = torch.clone(out_y)
        optimizer.zero_grad()  # Clear the gradients (partial derivatives)
        model.train()  # training mode
        outputs = model.forward(inputs_xb)

        loss = criterion(outputs, targets_yb)

        loss.backward()  # Update the weights and biases
        optimizer.step()
        j += 1
    # Validation
    for xb_test, yb_test in test_loader:
        xb_intest = torch.clone(xb_test.view(-1, seq_length, input_dim))
        yb_targtest = torch.clone(yb_test)
        outputs_test = model.forward(xb_intest)
        model.eval()
        loss_test = criterion(outputs_test, yb_targtest)

    save_train_loss.append(loss.cpu().detach().numpy())
    save_test_loss.append(loss_test.detach().numpy())

    if j % j / num_epochs == 0:
        print('Epoch: {}. Iteration: {}. Loss: {}'.format(epoch + 1, j, loss.data))
ts = time2[1] - time2[0]
save_model = True
if save_model is True:
    torch.save({'whole_model': model,
                'delta_t': ts,
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'output_dim': output_dim,
                'seq_len': seq_length,
                'hidden_dim': hidden_dim,
                'layer_dim': layer_dim,
                'optimizer_state_dict': optimizer.state_dict(),
                'batch_size': batch_size,
                'epoch': num_epochs,
                'scale_factor': sc1
                }, 'LSTM_model.pt')

    """ Save the model as a dictionary instead, with suffix .pt
    torch.save({'whole_model': model,
                'state_dict': model_state_dict(),
                'input_dim': input_dim,
                'output_dim': output_dim,
                'seq_len': seq_length,
                'hidden_dim': hidden_dim,
                'layer_dim': layer_dim,
                'optimizer': optimizer_state_dict(),
                'epoch': num_epochs,
    }, 'model.pt')
    """

# PREDICT MODEl
predicted = (model(X2.view(-1, seq_length, input_dim))).data.numpy()
# Inverse transform of the predicted and test data part
predicted = np.hstack((np.zeros((len(predicted), inx)), predicted))
predicted = sc1.inverse_transform(predicted)[:, inx:]

y2 = np.hstack((np.zeros((len(predicted), inx)), y2))
y2 = sc1.inverse_transform(y2)[:, inx:]

# Output value
# Output data are shorter by an initial sequence sequence
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size_long())
axes.plot(time2[seq_length:], y2[:, 0], 'g-')
axes.plot(time2[seq_length:], predicted[:, 0], 'b-')
fig.legend(['Raw Data', 'Predicted Data'],
           loc='upper center', fancybox=None, ncol=6, borderaxespad=0.1, edgecolor='black', fontsize=8,
           bbox_to_anchor=(0.5, 1), borderpad=0.3)
axes.set(xlabel=r"$t$ $[s]$")
axes.set(ylabel="pH")
axes.grid(True)
# Control input
fig1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size_long())
axes1.plot(time2[seq_length:], uncaled_test_data[seq_length:, 0], 'b-')
axes1.set(xlabel=r"$t$ $[s]$")
axes1.set(ylabel=r"$F_2$ $[mL.s^{-1}]$")
axes1.grid(True)
# Graph for MSE
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size_long())
axes.plot(save_train_loss, 'b-')
axes.plot(save_test_loss, 'r-')
fig.legend(['Train data', 'Validation data'],
           loc='upper center', fancybox=None, ncol=6, borderaxespad=0.1, edgecolor='black', fontsize=8,
           bbox_to_anchor=(0.5, 1), borderpad=0.3)
axes.set(xlabel=r"Loss")
axes.set(ylabel=r"epochs")

plt.show()
