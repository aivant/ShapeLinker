from abc import abstractmethod
from typing import List

import numpy as np

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary


class BaseConsoleInvokedComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def calculate_score_for_step(self, molecules: List, step=-1) -> ComponentSummary:
        return self.calculate_score(molecules, step)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        valid_smiles = self._chemistry.mols_to_smiles(molecules)
        score, raw_score = self._calculate_score(valid_smiles, step)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _get_step_string(self, step) -> str:
        if step == -1:
            return "\"\""
        return "".join(["\"e", str(step).zfill(4), "_\""])

    @abstractmethod
    def _calculate_score(self, smiles: List[str], step) -> np.array:
        raise NotImplementedError("_calculate_score method is not implemented")

    @abstractmethod
    def _create_command(self, step, input_json_path: str, output_json_path: str):
        raise NotImplementedError("_create_command method is not implemented")

