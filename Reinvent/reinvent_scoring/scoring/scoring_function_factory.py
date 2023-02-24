from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring import CustomProduct, CustomSum
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters
from reinvent_scoring.scoring.enums import ScoringFunctionNameEnum


class ScoringFunctionFactory:

    def __new__(cls, sf_parameters: ScoringFunctionParameters) -> BaseScoringFunction:
        enum = ScoringFunctionNameEnum()
        scoring_function_registry = {
            enum.CUSTOM_PRODUCT: CustomProduct,
            enum.CUSTOM_SUM: CustomSum
        }
        return cls.create_scoring_function_instance(sf_parameters, scoring_function_registry)

    @staticmethod
    def create_scoring_function_instance(sf_parameters: ScoringFunctionParameters,
                                         scoring_function_registry: dict) -> BaseScoringFunction:
        """Returns a scoring function instance"""
        scoring_function = scoring_function_registry[sf_parameters.name]
        parameters = [ComponentParameters(**p) for p in sf_parameters.parameters]

        return scoring_function(parameters, sf_parameters.parallel)
