from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.
    """

    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    l1_lambda: float = 0.0001
