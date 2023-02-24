from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.pip.base_pip_model_component import BasePiPModelComponent


class RatPKPiP(BasePiPModelComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._rat_pk_property = self.parameters.specific_parameters[self.component_specific_parameters.RAT_PK_PROPERTY]


    def _parse_single_compound(self, compound):
        return float(compound[self._rat_pk_property])
