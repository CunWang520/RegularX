import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import ModuleList
from torch_geometric.nn import BatchNorm, ARMAConv
import math
import warnings
from collections import OrderedDict
from torch.nn import Linear as Lin
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, act=nn.Tanh()):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(OrderedDict([
                ('lin1', Lin(in_channels, hidden_channels)),
                ('act', act),
                ('lin2', Lin(hidden_channels, out_channels))
                ]))
    def forward(self, x):
        return self.mlp(x)

class EdgeMaskNet(torch.nn.Module):

    def __init__(self,
                 n_in_channels,
                 e_in_channels,
                 hid=72, n_layers=3):
        super(EdgeMaskNet, self).__init__()

        self.node_lin = Lin(n_in_channels, hid)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(n_layers):
            conv = ARMAConv(in_channels=hid, out_channels=hid)   #TODO
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hid))

        self.mlp1 = MLP(2 * hid, hid, hid)
        self.mlp2 = MLP(e_in_channels, hid, hid)
        self.mlp = MLP(2 * hid, hid, 1)

    def forward(self, x, edge_index, edge_attr):

        x = torch.flatten(x, 1, -1)
        x = F.relu(self.node_lin(x))
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index))
            x = batch_norm(x)

        e = torch.cat([x[edge_index[0, :]], x[edge_index[1, :]]], dim=1)
        e1 = self.mlp1(e)

        e2 = self.mlp2(edge_attr)
        e = torch.cat([e1, e2], dim=1)  # connection

        return self.mlp(e)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


