from dataclasses import dataclass


@dataclass(frozen=True)
class ROCSInputFileTypesEnum:
    SHAPE_QUERY = "shape_query"
    SDF_QUERY = "sdf"
