# ==============================================================================
# models/backbones.py - Configurable GNN Encoder Backbones (GCN, GAT, SAGE, GIN)
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GATv2Conv
except ImportError:
    GCNConv = None
    GATConv = None
    SAGEConv = None
    GINConv = None
    GATv2Conv = None


def get_activation(name: str):
    name = name.lower()
    if name == 'relu':
        return F.relu
    elif name == 'elu':
        return F.elu
    elif name == 'gelu':
        return F.gelu
    elif name == 'leaky_relu':
        return F.leaky_relu
    else:
        raise ValueError(f"Unsupported activation: {name}")


class GNNEncoder(nn.Module):
    """
    Flexible GNN Encoder supporting:
      - Architecture: 'gcn', 'gat', 'sage', 'gin', 'gatv2'
      - Layers: 1, 2, or 3 layers
      - Attention heads (for GAT/GATv2)
      - Activation: 'relu', 'elu', 'gelu'
      - Dropout and Residual Connections
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        gnn_type: str = 'gcn',
        num_layers: int = 1,
        heads: int = 1,
        dropout: float = 0.0,
        activation: str = 'relu',
        residual: bool = False
    ):
        super(GNNEncoder, self).__init__()
        self.gnn_type = gnn_type.lower()
        self.num_layers = num_layers
        self.dropout = dropout
        self.act_fn = get_activation(activation)
        self.residual = residual

        self.convs = nn.ModuleList()

        dims = [in_channels]
        if num_layers == 1:
            dims.append(out_channels)
        elif num_layers == 2:
            dims.extend([hidden_channels, out_channels])
        elif num_layers >= 3:
            dims.extend([hidden_channels] * (num_layers - 1))
            dims.append(out_channels)

        for l in range(num_layers):
            c_in = dims[l]
            c_out = dims[l + 1]
            is_last = (l == num_layers - 1)

            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(c_in, c_out))
            elif self.gnn_type in ['gat', 'gatv2']:
                conv_cls = GATv2Conv if self.gnn_type == 'gatv2' else GATConv
                h = 1 if is_last else heads
                # If multi-head and intermediate, output is c_out * h, so adjust input to next layer
                out_per_head = c_out if is_last else max(1, c_out // heads)
                concat = not is_last
                self.convs.append(conv_cls(c_in, out_per_head, heads=h, concat=concat, dropout=dropout))
            elif self.gnn_type == 'sage':
                self.convs.append(SAGEConv(c_in, c_out))
            elif self.gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(c_in, c_out),
                    nn.ReLU(),
                    nn.Linear(c_out, c_out)
                )
                self.convs.append(GINConv(mlp))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}. Choose from 'gcn', 'gat', 'sage', 'gin', 'gatv2'.")

        if self.residual:
            self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None):
        identity = x
        for i, conv in enumerate(self.convs):
            is_last = (i == len(self.convs) - 1)

            if self.gnn_type in ['gcn']:
                x = conv(x, edge_index, edge_weight)
            else:
                # GAT, SAGE, GIN take edge_index
                x = conv(x, edge_index)

            if not is_last:
                x = self.act_fn(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)

        if self.residual:
            x = x + self.res_proj(identity)

        return x

