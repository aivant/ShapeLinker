from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringFunctionNameEnum:
    CUSTOM_PRODUCT = "custom_product"
    CUSTOM_SUM = "custom_sum"
