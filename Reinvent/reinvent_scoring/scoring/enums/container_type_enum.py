from dataclasses import dataclass


@dataclass(frozen=True)
class ContainerType:
    SCIKIT_CONTAINER = "scikit_container"
    OPTUNA_CONTAINER = "optuna_container"
