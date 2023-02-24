from dataclasses import dataclass


@dataclass(frozen=True)
class DiversityFilterEnum:
    IDENTICAL_TOPOLOGICAL_SCAFFOLD = "IdenticalTopologicalScaffold"
    IDENTICAL_MURCKO_SCAFFOLD = "IdenticalMurckoScaffold"
    SCAFFOLD_SIMILARITY = "ScaffoldSimilarity"
    NO_FILTER = "NoFilter"
