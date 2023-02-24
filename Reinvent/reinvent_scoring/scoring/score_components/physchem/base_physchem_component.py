from abc import abstractmethod
from typing import List

import numpy as np
from reinvent_chemistry.phys_chem_descriptors import PhysChemDescriptors

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary


class BasePhysChemComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._phys_chem_descriptors = PhysChemDescriptors()

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        score, raw_score = self._calculate_score(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _calculate_score(self, query_mols) -> np.array:
        scores = []
        for mol in query_mols:
            try:
                score = self._calculate_phys_chem_property(mol)
            except ValueError:
                score = 0.0
            scores.append(score)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)
        return np.array(transformed_scores, dtype=np.float32), np.array(scores, dtype=np.float32)

    @abstractmethod
    def _calculate_phys_chem_property(self, mol):
        raise NotImplementedError("_calculate_phys_chem_property method is not implemented")

