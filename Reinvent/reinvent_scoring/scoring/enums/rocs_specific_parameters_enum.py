from dataclasses import dataclass


@dataclass(frozen=True)
class ROCSSpecificParametersEnum():
    ROCS_INPUT = "rocs_input"
    INPUT_TYPE = "input_type"
    SHAPE_WEIGHT = "shape_weight"
    COLOR_WEIGHT = "color_weight"
    SIM_MEASURE = "similarity_measure"
    MAX_CPUS = "max_num_cpus"
    CUSTOM_CFF = "custom_cff"
    SAVE_ROCS_OVERLAYS = "save_rocs_overlays"
    ROCS_OVERLAYS_DIR = "rocs_overlays_dir"
    ROCS_OVERLAYS_PREFIX = "rocs_overlays_prefix"
    ENUM_STEREO = "enumerate_stereo"
    MAX_STEREO = "max_stereocenters"
    NEGATIVE_VOLUME = "negative_volume"
    PROTEIN_NEG_VOL_FILE = "protein_neg_vol_file"
    LIGAND_NEG_VOL_FILE = "ligand_neg_vol_file"
    MAX_CONFS = "max_confs"
    EWINDOW = "ewindow"
