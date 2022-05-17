from .gnn import GNN
from .soft_attention_gnn import SAM
from .hetero.gnn import HGNN
from .hetero.soft_attention_gnn import HSAM
from .hetero.fixed_attention_gnn import HFAM
from .hetero.gnn_transformer import GraphTransformer

REGISTERED_MODELS = {
    "GNN": GNN,
    "SAM": SAM,
    "HGNN": HGNN,
    "HSAM": HSAM,
    "HFAM": HFAM,
    "HGT": GraphTransformer
}
