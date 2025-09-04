from typing import Any, Dict, List

from nemesis.misc.fitness_metrics import Fitness


class EvaluationMetrics:
    def __init__(
        self,
        is_valid_solution: bool,
        fitness: Fitness,
        n_trainable_parameters: int,
        n_layers: int,
        training_time_spent: float,
        train_losses: List[float],
        n_epochs: int,
        total_epochs_trained: int,
        max_epochs_reached: bool,
    ) -> None:
        self.is_valid_solution: bool = is_valid_solution
        self.fitness: Fitness = fitness
        self.n_trainable_parameters: int = n_trainable_parameters
        self.n_layers: int = n_layers
        self.training_time_spent: float = training_time_spent
        self.train_losses: List[float] = train_losses
        self.n_epochs: int = n_epochs
        self.total_epochs_trained: int = total_epochs_trained
        self.max_epochs_reached: bool = max_epochs_reached

    @classmethod
    def default(cls, fitness: Fitness) -> "EvaluationMetrics":
        return cls(
            is_valid_solution=False,
            fitness=fitness,
            n_trainable_parameters=-1,
            n_layers=-1,
            training_time_spent=0.0,
            train_losses=[],
            n_epochs=0,
            total_epochs_trained=0,
            max_epochs_reached=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid_solution": self.is_valid_solution,
            "fitness": self.fitness,
            "n_trainable_parameters": self.n_trainable_parameters,
            "n_layers": self.n_layers,
            "training_time_spent": self.training_time_spent,
            "train_losses": self.train_losses,
            "n_epochs": self.n_epochs,
            "total_epochs_trained": self.total_epochs_trained,
            "max_epochs_reached": self.max_epochs_reached,
        }

    def __str__(self) -> str:
        return (
            "EvaluationMetrics("
            + f"is_valid_solution: {self.is_valid_solution},  "
            + f"n_trainable_parameters: {self.n_trainable_parameters},  "
            + f"n_layers: {self.n_layers},  "
            + f"training_time_spent: {self.training_time_spent},  "
            + f"n_epochs: {self.n_epochs},  "
            + f"total_epochs_trained: {self.total_epochs_trained},  "
            + f"fitness: {self.fitness},  "
            + f"train_losses: {self.train_losses},  "
            + f"max_epochs_reached: {self.max_epochs_reached})"
        )
