from typing import List

from reinvent_chemistry.similarity import Similarity
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary


class TanimotoSimilarity(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._similarity = Similarity()
        self._radius = self.parameters.specific_parameters.get("radius", 3)
        self._use_counts = self.parameters.specific_parameters.get("use_counts", True)
        self._use_features = self.parameters.specific_parameters.get("use_features", True)
        smiles = self.parameters.specific_parameters.get(self.component_specific_parameters.SMILES, [])
        self._ref_fingerprints = self._chemistry.smiles_to_fingerprints(smiles, radius=self._radius,
                                                                        use_counts=self._use_counts,
                                                                        use_features=self._use_features)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        query_fps = self._chemistry.mols_to_fingerprints(molecules, self._radius, self._use_counts, self._use_features)
        score = self._similarity.calculate_tanimoto(query_fps, self._ref_fingerprints)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

