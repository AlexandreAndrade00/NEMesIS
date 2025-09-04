from dataclasses import dataclass


from torch import optim


@dataclass
class LearningParams:
    epochs: int
    torch_optimiser: optim.Optimizer

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LearningParams):
            return self.torch_optimiser.__dict__ == other.torch_optimiser.__dict__ and self.epochs == other.epochs
        return False
