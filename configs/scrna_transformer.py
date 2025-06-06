from dataclasses import dataclass


@dataclass
class ScRNATransformerConfig:
    """
    Configuration class for ScRNATransformer.
    """

    embed_dim: int = 16
    num_heads: int = 4
    ff_dim: int = 64
    num_layers: int = 4
    max_seq_len: int = 512
    dropout: float = 0.1
