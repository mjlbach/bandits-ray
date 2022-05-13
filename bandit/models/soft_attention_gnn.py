import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
# from torch_geometric.nn import HANConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter

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
        # self.a_conv1 = HGTConv(in_features, 64, metadata, num_heads)
        # self.a_conv2 = HGTConv(64, 32, metadata, num_heads)
        # self.a_conv3 = HGTConv(32, num_heads, metadata, num_heads)

        self.weight_mlp = MLP([in_features, 32, 1], batch_norm=False)

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

        # import pdb; pdb.set_trace()

        # x_encoded = x_dict_init['node']

        weight = self.weight_mlp(x_dict_init['node'])
        weight = F.softmax(weight, dim=0)
        
        print("weight:", weight)
        print("weight shape", weight.shape)

        # weight = softmax(weight_dict['node'], batch['node'])
        # x = graph_sum(x_dict['node'] * weight.mean(-1, keepdim=True), batch['node'])
        # weight = weight.mean(-1, keepdim=True)
        # print("node type & weights:", list(zip(torch.flatten(x_dict_init['node']).tolist(), 
        #                                       torch.flatten(weight).tolist())))

        # weight = ((x_dict_init['node'] == 1) | (x_dict_init['node'] == 2)).float()
        # weight[weight==0] = 2

        x = graph_sum(x_dict['node'] * weight, batch['node'])
        # x = graph_sum(x_dict['node'] * float(1.0/3.0), batch['node'])

        # Final encoding layers
        x = self.mlp(x)

        return x

    def get_weights(self, data):
        # Extract the features

        x_dict_init = data.x_dict
        edge_index_dict = data.edge_index_dict
        batch = data.batch_dict

        # # Second round of graph convolution
        # # Compute the per node attention for edge
        # weight_dict = self.a_conv1(x_dict_init, edge_index_dict)
        # weight_dict = {key: weight.relu() for key, weight in weight_dict.items()}
        # weight_dict = self.a_conv2(weight_dict, edge_index_dict)
        # weight_dict = {key: weight.relu() for key, weight in weight_dict.items()}
        # weight_dict = self.a_conv3(weight_dict, edge_index_dict)
        # weight_dict = {key: weight.relu() for key, weight in weight_dict.items()}

        # weight = softmax(weight_dict['node'], batch['node'])
        # # x = graph_sum(x_dict['node'] * weight.mean(-1, keepdim=True), batch['node'])
        # weight = weight.mean(-1, keepdim=True)

        with torch.no_grad():
            weight = self.weight_mlp(x_dict_init['node'])
            weight = F.softmax(weight, dim=0)

        return weight
