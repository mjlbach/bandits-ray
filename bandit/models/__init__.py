from .gnn import GNN
from .soft_attention_gnn import SAM
from .hetero.gnn import HGNN
from .hetero.gnn_transformer import GraphTransformer

REGISTERED_MODELS = {
    "GNN": GNN,
    "SAM": SAM,
    "HGNN": HGNN,
    "HGT": GraphTransformer
}
