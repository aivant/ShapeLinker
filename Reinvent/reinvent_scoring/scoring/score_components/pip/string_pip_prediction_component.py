import numpy as np
from typing import List
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.pip.base_pip_model_component import BasePiPModelComponent


class StringPiPPredictionComponent(BasePiPModelComponent):
    """
    This class is to be used with pip models that return non-numeric (string mostly)
    values as predictions - casts raw value to float immediately.
    """

    def _parse_single_compound(self, compound):
        mapping = self.parameters.specific_parameters[self.component_specific_parameters.VALUE_MAPPING]
        value = compound["prediction"]

        return mapping.get(value, 0.0)
