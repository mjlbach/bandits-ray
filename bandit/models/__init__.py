from .gnn import GNN
from .soft_attention_gnn import HSAM
from .hetero.gnn import HGNN
from .hetero.gnn_transformer import GraphTransformer

REGISTERED_MODELS = {
    "GNN": GNN,
    "HSAM": HSAM,
    "HGNN": HGNN,
    "HGT": GraphTransformer
}
