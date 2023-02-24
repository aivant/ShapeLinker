from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnNamesEnum:
    STEP: str = "Step"
    SCAFFOLD: str = "Scaffold"
    SMILES: str = "SMILES"
    METADATA: str = "Metadata"