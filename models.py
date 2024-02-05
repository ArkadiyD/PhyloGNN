import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv, GINConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, mul, sum as sparsesum


class GINConvModule(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(output_dim, output_dim),
            )
        )

    def forward(self, x, edge_index, edge_features=None):
        return self.conv(x, edge_index)


class GATConvModule(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = GATConv(input_dim, output_dim)

    def forward(self, x, edge_index, edge_features=None):
        return self.conv(x, edge_index, edge_attr=edge_features)

#This code is adopted from https://github.com/emalgorithm/directed-graph-neural-network
def directed_norm(adj):
    r"""
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj

#This code is adopted from https://github.com/emalgorithm/directed-graph-neural-network
def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")


class GCNConvModule(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv = GCNConv(input_dim, output_dim, add_self_loops=False)

    def forward(self, x, edge_index, edge_features=None):
        return self.conv(x, edge_index)

#This code is adopted from https://github.com/emalgorithm/directed-graph-neural-network
class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = torch.nn.Linear(input_dim, output_dim)
        self.lin_dst_to_src = torch.nn.Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index, edge_features=None):
        row, col = edge_index
        num_nodes = x.shape[0]

        adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        self.adj_norm = get_norm_adj(adj, norm="dir")

        adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
        self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (
            1 - self.alpha
        ) * self.lin_dst_to_src(self.adj_t_norm @ x)

#This code is adopted from https://github.com/emalgorithm/directed-graph-neural-network
class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output = output_dim

        self.conv_src_to_dst = GATConv(input_dim, output_dim)
        self.conv_dst_to_src = GATConv(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index, edge_features=None):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)
        return (1 - self.alpha) * self.conv_src_to_dst(
            x, edge_index
        ) + self.alpha * self.conv_dst_to_src(x, edge_index_t)


class DirGINConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output = output_dim

        self.conv_src_to_dst = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(output_dim, output_dim),
            ),
            flow="source_to_target",
        )

        self.conv_dst_to_src = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(output_dim, output_dim),
            ),
            flow="target_to_source",
        )

        self.alpha = alpha

    def forward(self, x, edge_index, edge_features=None):
        return (1 - self.alpha) * self.conv_src_to_dst(
            x, edge_index
        ) + self.alpha * self.conv_dst_to_src(x, edge_index)


class MultiscaleEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_sizes, conv, conv_dropout):
        super().__init__()

        self.layer_num = len(hidden_sizes)
        self.conv_dropout = conv_dropout
        self.convs = torch.nn.ModuleList()
        for l in range(self.layer_num):
            if l == 0:
                input_dim = num_features
            else:
                input_dim = hidden_sizes[l - 1]
            output_dim = hidden_sizes[l]

            if conv == "GIN":
                self.convs.append(GINConvModule(input_dim, output_dim))
            elif conv == "GCN":
                self.convs.append(GCNConvModule(input_dim, output_dim))
            elif conv == "GAT":
                self.convs.append(GATConvModule(input_dim, output_dim))
            elif conv == "DirGAT":
                self.convs.append(DirGATConv(input_dim, output_dim))
            elif conv == "DirGCN":
                self.convs.append(DirGCNConv(input_dim, output_dim))
            elif conv == "DirGIN":
                self.convs.append(DirGINConv(input_dim, output_dim))
            
            else:
                raise ValueError("conv should be one of GIN, GCN, GAT, DirGAT, DirGCN, DirGIN, got:", conv)

    def forward(self, x, edge_index, edge_features):
        outs = []
        for i in range(self.layer_num):
            x = self.convs[i](x, edge_index, edge_features)
            if self.conv_dropout > 0:
                x = torch.nn.functional.dropout(
                    x, p=self.conv_dropout, training=self.training
                )
            x = torch.nn.functional.relu(x)
            outs.append(x)

        return outs


class NetworkCombineGraphs(torch.nn.Module):
    def __init__(
        self,
        num_features,
        hidden_sizes=[128, 128, 128, 128, 128],
        use_edge_features=2,
        pool="mean",
        conv="gin",
        conv_dropout=0.0,
        mlp_dropout=0.0,
        print_info=True,
    ):
        super().__init__()
        self.num_gnn_layers = len(hidden_sizes)
        self.pool_type = str(pool)

        self.features_embedding = nn.Linear(num_features, num_features)

        if self.pool_type == "mean":
            self.pool = torch_geometric.nn.global_mean_pool
        elif self.pool_type == "add":
            self.pool = torch_geometric.nn.global_add_pool
        elif self.pool_type == "max":
            self.pool = torch_geometric.nn.global_max_pool
        else:
            raise ValueError("pool should be one of mean, add, max, got:", pool)

        if print_info:      
            print("hidden_sizes", hidden_sizes)
            print("conv:", conv)
            print("pool:", pool)
            print("conv_dropout:", conv_dropout)
            print("mlp_dropout:", mlp_dropout)

        self.encoder = MultiscaleEncoder(
            num_features, hidden_sizes, str(conv), conv_dropout
        )
        if mlp_dropout > 0:
            self.post = torch.nn.Sequential(
                torch.nn.Dropout(p=mlp_dropout),
                torch.nn.Linear(num_features + np.sum(hidden_sizes), hidden_sizes[-1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=mlp_dropout),
                torch.nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
            )
        else:
            self.post = torch.nn.Sequential(
                torch.nn.Linear(num_features + np.sum(hidden_sizes), hidden_sizes[-1]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
            )

        self.final = torch.nn.Linear(hidden_sizes[-1], 1)

    def forward(self, data, additional_features=None):
        x = data.x
        edge_index = data.edge_index
        edge_features = data.edge_attr

        x = self.features_embedding(x)

        nodewise_scores_list = [x] + self.encoder(
            x, edge_index, edge_features=edge_features
        )
        nodewise_scores = torch.cat([emb for emb in nodewise_scores_list], dim=1)
        out = self.pool(nodewise_scores, data.batch)
        out = self.post(out)
        out = self.final(out)
        
        out = torch.sigmoid(out)
        return out

class BaselineSiameseGNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        hidden_sizes=[128, 128, 128, 128, 128],
        conv="gin",
        pool="add",
        mlp_dropout=0.2,
    ):
        super().__init__()
        self.num_gnn_layers = len(hidden_sizes)
        self.n_additional_features = 0
        print("num_features", num_features)
        self.features_embedding = nn.Linear(num_features, num_features)
        self.pool_type = pool
        if pool == "add":
            self.pool = torch_geometric.nn.global_add_pool
        elif pool == "max":
            self.pool = torch_geometric.nn.global_max_pool
        elif pool == "mean":
            self.pool = torch_geometric.nn.global_mean_pool
        else:
            raise ValueError("pool should be one of add, max, mean, got:", pool)

        self.encoder = MultiscaleEncoder(
            num_features, hidden_sizes, conv, conv_dropout=0.0
        )
        if mlp_dropout > 0:
            self.post = torch.nn.Sequential(
                torch.nn.Dropout(p=mlp_dropout),
                torch.nn.Linear(num_features + np.sum(hidden_sizes), hidden_sizes[-1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=mlp_dropout),
                torch.nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
            )

            self.mlp_model = torch.nn.Sequential(
                torch.nn.Dropout(p=mlp_dropout),
                torch.nn.Linear(2 * hidden_sizes[-1], hidden_sizes[-1]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_sizes[-1], 1),
            )

        else:
            self.post = torch.nn.Sequential(
                torch.nn.Linear(num_features + np.sum(hidden_sizes), hidden_sizes[-1]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
            )

            self.mlp_model = torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_sizes[-1], hidden_sizes[-1]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_sizes[-1], 1),
            )


    def forward(self, data, data_tree, additional_features=None):
        x = data.x
        edge_index = data.edge_index
        
        x_tree = data_tree.x  # [:,:-1]
        edge_index_tree = data_tree.edge_index
        
        x = self.features_embedding(x)
        x_tree = self.features_embedding(x_tree)

        nodewise_embeddings_list = [x] + self.encoder(
            x, edge_index, data.edge_attr
        )
        nodewise_embeddings_list_tree = [x_tree] + self.encoder(
            x_tree, edge_index_tree, data_tree.edge_attr
        )

        leaf_diff = 0

        nodewise_embeddings = torch.cat([emb for emb in nodewise_embeddings_list], dim=1)
        nodewise_embeddings_tree = torch.cat(
            [emb for emb in nodewise_embeddings_list_tree], dim=1
        )

        x = self.pool(nodewise_embeddings, data.batch)
        tree_x = self.pool(nodewise_embeddings_tree, data_tree.batch)
    
        x = self.post(x)
        tree_x = self.post(tree_x)
        
        out = self.mlp_model(torch.cat([x, tree_x], dim=1))
        out = torch.sigmoid(out)
        return out