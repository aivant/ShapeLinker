from dataclasses import dataclass


@dataclass(frozen=True)
class TransformationTypeEnum:
    DOUBLE_SIGMOID = "double_sigmoid"
    SIGMOID = "sigmoid"
    REVERSE_SIGMOID = "reverse_sigmoid"
    RIGHT_STEP = "right_step"
    LEFT_STEP = "left_step"
    STEP = "step"
    CUSTOM_INTERPOLATION = "custom_interpolation"
    NO_TRANSFORMATION = "no_transformation"
