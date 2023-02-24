from abc import abstractmethod
from typing import List

import numpy as np
from reinvent_chemistry.link_invent.linker_descriptors import LinkerDescriptors

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary


class BaseLinkInventComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._linker_descriptor = LinkerDescriptors()

    def calculate_score(self, labeled_molecules: List, step=-1) -> ComponentSummary:
        score, raw_score = self._calculate_score(labeled_molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _calculate_score(self, query_labeled_mols) -> np.array:
        scores = []
        for mol in query_labeled_mols:
            try:
                score = self._calculate_linker_property(mol)
            except ValueError:
                score = 0.0
            scores.append(score)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)
        return np.array(transformed_scores, dtype=np.float32), np.array(scores, dtype=np.float32)

    @abstractmethod
    def _calculate_linker_property(self, labeled_mol):
        raise NotImplementedError("_calculate_linker_property method is not implemented")