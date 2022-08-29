import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPoolLayer(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(MaxPoolLayer, self).__init__()
        self.maxPool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.maxPool(x)
        return x.permute(0, 2, 1)


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class ConvLayer2D(nn.Module):
    """2-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """
    def __init__(self, n_features, kernel_size=5):
        super(ConvLayer2D, self).__init__()
        #self.padding = nn.ConstantPad2d(((kernel_size - 1) // 2, (kernel_size - 1) // 2), 0.0)
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_features, kernel_size=kernel_size, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        #print('x:', x.shape)
        x = torch.unsqueeze(x, dim=1)
        #x = self.padding(x)
        #print('padding:', x.shape)
        x = self.relu(self.conv(x))
        #print('conv2:', x.shape)
        x = x.permute(0, 2, 3, 1)
        #print('permute:', x.shape)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        return x #x.permute(0, 2, 1)  # Permute back


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)

