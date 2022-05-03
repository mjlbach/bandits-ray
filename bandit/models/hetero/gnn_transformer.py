import torch
from torch_geometric.nn import HGTConv

from torch_geometric.nn import MLP
import torch_geometric as pyg
import math

from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Transformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        # 3 layer embedding network for obj feature vectors
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, d_model),
        )

        # The CLS Token used as input on the first element of each sequence
        self.cls = nn.parameter.Parameter(
            data=torch.nn.init.xavier_uniform_(torch.zeros(1, num_features), gain=1.0)
        )

        # Transformer encoder layer which oeprates on node embeddings
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Embedding dimension of model
        self.d_model = d_model

        # Out features after decoding
        self.out_features = ntoken

        # Decoder which reads out CLS token
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, lengths: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        # Make mask based on repeated sequence length, padding with 1 for CLS token (appended at start of sequence)
        # Mask == True means that the elements per each sequence will *not* be included in the attention mask
        mask = torch.arange(src.shape[1] + 1).repeat((src.shape[0], 1)).to(
            lengths.device
        ) >= (lengths[:, None] + 1)

        # Append CLS token to start of sequence
        cls = torch.broadcast_to(
            self.cls, (src.shape[0], self.cls.shape[0], self.cls.shape[1])
        )
        src = torch.cat([cls, src], dim=1)

        # Run the model
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output = self.decoder(output)

        # Read out only CLS token
        output = output[:, 0, :]

        return output

class HGNN(torch.nn.Module):
    def __init__(self, in_features, out_features=256, metadata=None, num_heads=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Embedding network
        self.mlp = MLP([in_features, 128, 128], batch_norm=False)

        # The scene graph encoding is a 3 layer deep graph convolution,
        # Equivalent to 3 MLP layers
        self.conv1 = HGTConv(128, 128, metadata, num_heads)
        self.conv2 = HGTConv(128, 128, metadata, num_heads)
        self.conv3 = HGTConv(128, out_features, metadata, num_heads)

    def forward(self, data):
        # Extract the features
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        # batch = data.batch_dict

        # Embed features
        x_dict = {key: self.mlp(x) for key, x in x_dict.items()}
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        # First round of graph convolution
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict['node'] # batch * node, features

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_features, metadata):
        super().__init__()

        self.gnn = HGNN(in_features, out_features=128, metadata=metadata)
        self.transformer = Transformer(
            num_features=self.gnn.out_features,
            ntoken=128,
            d_model=256,
            d_hid=200,
            nhead=8,
            dropout=0.2,
            nlayers=2,
        )
        self.out_features = self.transformer.out_features

    def forward(self, data, lengths, goal=None):
        x = self.gnn(data)
        x, _ = pyg.utils.to_dense_batch(x, data.to_homogeneous().batch, max_num_nodes=50)
        return self.transformer(x, lengths)

class GoalEmbeddingGraphTransformer(torch.nn.Module):
    def __init__(self, in_features, goal_features, metadata):
        super().__init__()

        # Embedding network
        self.goal_embedding_mlp = MLP([goal_features, 128, 128], batch_norm=False)

        self.gnn = HGNN(in_features, out_features=128, metadata=metadata)
        self.transformer = Transformer(
            num_features=self.gnn.out_features,
            ntoken=128,
            d_model=256,
            d_hid=200,
            nhead=8,
            dropout=0.2,
            nlayers=2,
        )
        self.out_features = self.transformer.out_features

    def forward(self, data, lengths, goal):
        x = self.gnn(data)
        x, _ = pyg.utils.to_dense_batch(x, data.to_homogeneous().batch, max_num_nodes=50)
        embedded_goal = self.goal_embedding_mlp(goal)
        embedded_goal = embedded_goal.reshape(-1, 1, 128)
        x = torch.cat([embedded_goal, x], dim=1)
        return self.transformer(x, lengths)

def get_fake_data(size):
    from torch_geometric.data import HeteroData
    data = HeteroData()
    obs = {
        "scene_graph": {
            "nodes": torch.ones((size, 8), dtype=torch.float),
        }
    }
    obs["scene_graph"]["onTop"] = torch.zeros((size, 2), dtype=torch.long)
    obs["scene_graph"]["onTop"][1,0] = 1

    for key in obs["scene_graph"]:
        if key == 'nodes':
            data['node'].x = torch.tensor(obs['scene_graph']['nodes'], dtype=torch.float)
        else:
            data['node', key, 'node'].edge_index = torch.tensor(obs['scene_graph'][key], dtype=torch.long).T
    return data

def main():
    from torch_geometric.data import Batch
    data_1 = get_fake_data(5)
    data_2 = get_fake_data(7)
    batch = Batch.from_data_list([data_1, data_2])

    with torch.no_grad():
        model = GraphTransformer(in_features=8, metadata = (['node'], [('node', 'onTop', 'node')]))
    out = model(batch)



if __name__ == "__main__":
    main()

