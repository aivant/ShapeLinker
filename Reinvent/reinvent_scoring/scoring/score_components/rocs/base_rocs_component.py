import glob
import os
import re
from abc import abstractmethod
from typing import List

import numpy as np

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary


class BaseROCSComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def calculate_score_for_step(self, molecules: List, step=-1) -> ComponentSummary:
        return self.calculate_score(molecules, step)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        # NOTE: valid_idxs are determined with RDKit not with Open Eye
        valid_smiles = self._chemistry.mols_to_smiles(molecules)
        score = self._calculate_omega_score(valid_smiles, step)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        score = self._transformation_function(score, transform_params)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    @abstractmethod
    def _calculate_omega_score(self, smiles, step) -> np.array:
        raise NotImplementedError("_calculate_omega_score method is not implemented")
