from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeAlignmentSpecificParametersEnum:
    QUERY_TYPE = "query_type"
    QUERY = "query"
    MODEL_PATH = "model_path"
    ALIGNMENT_ENV = "alignment_env"
    NUM_CONFORMERS = "num_conformers"
    POSES_FOLDER = "poses_folder"
    ES_WEIGHT = "es_weight"
    GET_EXT_LINKER = "get_ext_linker"
    CORRECT_FLIPPING = "correct_flipping"