from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.pip.base_pip_model_component import BasePiPModelComponent


class PiPPredictionComponent(BasePiPModelComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def _parse_single_compound(self, compound):
        return float(compound["prediction"])
