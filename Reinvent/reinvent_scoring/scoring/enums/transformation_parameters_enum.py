from dataclasses import dataclass


@dataclass(frozen=True)
class TransformationParameters:
    TRANSFORMATION_TYPE = "transformation_type"
    LOW = "low"
    HIGH = "high"
    K = "k"
    COEF_DIV = "coef_div"
    COEF_SI = "coef_si"
    COEF_SE = "coef_se"
    TRUNCATE_LEFT = "truncate_left"
    TRUNCATE_RIGHT = "truncate_right"
    INTERPOLATION_MAP = "interpolation_map"


TransformationParametersEnum = TransformationParameters()
