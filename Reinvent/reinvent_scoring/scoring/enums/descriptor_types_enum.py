from dataclasses import dataclass


@dataclass(frozen=True)
class DescriptorTypesEnum():
    ECFP = "ecfp"
    ECFP_COUNTS = "ecfp_counts"
    MACCS_KEYS = "maccs_keys"
    AVALON = "avalon"
