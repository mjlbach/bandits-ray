import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
# from torch_geometric.nn import HANConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

from torch_geometric.nn import MLP


def weighted_pool(x, weight, batch, size=None):
    size = int(batch.max().item() + 1) if size is None else size
    x = x * weight
    return scatter(x, batch, dim=0, dim_size=size, reduce="sum")

def graph_sum(x, batch, size=None):
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce="sum")


class HSAM(torch.nn.Module):
    def __init__(self, in_features, out_features=256, metadata=None, num_heads=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # The scene graph encoding is a 3 layer deep graph convolution,
        # Equivalent to 3 MLP layers
        self.conv1 = HGTConv(in_features, 128, metadata, num_heads)
        self.conv2 = HGTConv(128, 128, metadata, num_heads)
        self.conv3 = HGTConv(128, 128, metadata, num_heads)

        # The attention mask is a 3 layer deep graph convolution, equivalent to 3 MLP
        # layers
        self.a_conv1 = HGTConv(in_features, 64, metadata, num_heads)
        self.a_conv2 = HGTConv(64, 32, metadata, num_heads)
        self.a_conv3 = HGTConv(32, num_heads, metadata, num_heads)

        self.weight_mlp = MLP([6, 32, 1], batch_norm=False)

        self.mlp = MLP([128, 128, out_features], batch_norm=False)

    def forward(self, data):
        # Extract the features
        x_dict_init = data.x_dict
        edge_index_dict = data.edge_index_dict
        batch = data.batch_dict


        # First round of graph convolution
        x_dict = self.conv1(x_dict_init, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        # Second round of graph convolution
        # Compute the per node attention for edge
        # weight_dict = self.a_conv1(x_dict_init, edge_index_dict)
        # weight_dict = {key: weight.relu() for key, weight in weight_dict.items()}
        # weight_dict = self.a_conv2(weight_dict, edge_index_dict)
        # weight_dict = {key: weight.relu() for key, weight in weight_dict.items()}
        # weight_dict = self.a_conv3(weight_dict, edge_index_dict)
        # weight_dict = {key: weight.relu() for key, weight in weight_dict.items()}

        # Softmax of the weight 
        # weight = softmax(weight_dict['node'], batch['node'])
        # weight = weight.mean(-1, keepdim=True)
        # x_encoded = x_dict_init['node']
        x_encoded = F.one_hot(torch.squeeze(x_dict_init['node'], 1).long(), num_classes=6).float()
        weight = self.weight_mlp(x_encoded)
        # weight = self.weight_mlp(x_dict['node'])
        weight = softmax(weight, batch['node'])

        # print("node type & weights:", list(zip(torch.flatten(x_dict_init['node']).tolist(), 
        #                                       torch.flatten(weight).tolist())))
        # weight = ((x_dict_init['node'] == 1) | (x_dict_init['node'] == 2)).float()
        # weight[weight==0] = 2

        # print("weight:", weight)
        # print("weight shape", weight.shape)
        # print("weight sum", weight.sum())
        
        x = graph_sum(x_dict['node'] * weight, batch['node'])

        # Final encoding layers
        x = self.mlp(x)

        return x

    def get_weights(self, data):
        # Extract the features

        x_dict_init = data.x_dict
        edge_index_dict = data.edge_index_dict
        batch = data.batch_dict


        with torch.no_grad():
            # Second round of graph convolution
            # Compute the per node attention for edge
            # weight_dict = self.a_conv1(x_dict_init, edge_index_dict)
            # weight_dict = {key: weight.relu() for key, weight in weight_dict.items()}
            # weight_dict = self.a_conv2(weight_dict, edge_index_dict)
            # weight_dict = {key: weight.relu() for key, weight in weight_dict.items()}
            # weight_dict = self.a_conv3(weight_dict, edge_index_dict)
            # weight_dict = {key: weight.relu() for key, weight in weight_dict.items()}

            # weight = scatter_softmax(weight_dict['node'], batch['node'], dim=0)
            # weight = weight.mean(-1, keepdim=True)

            # First round of graph convolution
            # x_dict = self.conv1(x_dict_init, edge_index_dict)
            # x_dict = {key: x.relu() for key, x in x_dict.items()}
            # x_dict = self.conv2(x_dict, edge_index_dict)
            # x_dict = {key: x.relu() for key, x in x_dict.items()}
            # x_dict = self.conv3(x_dict, edge_index_dict)
            # x_dict = {key: x.relu() for key, x in x_dict.items()}

            # Second round of graph convolution
            # Compute the per node attention for edge
            # weight_dict = self.a_conv1(x_dict_init, edge_index_dict)
            # weight_dict = {key: weight.relu() for key, weight in weight_dict.items()}
            # weight_dict = self.a_conv2(weight_dict, edge_index_dict)
            # weight_dict = {key: weight.relu() for key, weight in weight_dict.items()}
            # weight_dict = self.a_conv3(weight_dict, edge_index_dict)
            # weight_dict = {key: weight.relu() for key, weight in weight_dict.items()}

            # Softmax of the weight 
            # weight = softmax(weight_dict['node'], batch['node'])
            # weight = weight.mean(-1, keepdim=True)

            # x_encoded = x_dict_init['node']
            x_encoded = F.one_hot(torch.squeeze(x_dict_init['node'], 1).long(), num_classes=6).float()
            weight = self.weight_mlp(x_encoded)
            # weight = self.weight_mlp(x_dict['node'])
            weight = softmax(weight, batch['node'])

            

        return weight
