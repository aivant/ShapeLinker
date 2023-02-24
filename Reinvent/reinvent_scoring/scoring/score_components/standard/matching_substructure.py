from typing import List

import numpy as np
from rdkit import Chem

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary


class MatchingSubstructure(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.target_smarts = self.parameters.specific_parameters.get(self.component_specific_parameters.SMILES, [])
        self._validate_inputs(self.target_smarts)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = self._substructure_match(molecules, self.target_smarts)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def _substructure_match(self, query_mols, list_of_SMARTS):
        if len(list_of_SMARTS) == 0:
            return np.ones(len(query_mols), dtype=np.float32)

        match = [any([mol.HasSubstructMatch(Chem.MolFromSmarts(subst)) for subst in list_of_SMARTS
                      if Chem.MolFromSmarts(subst)]) for mol in query_mols]
        return 0.5 * (1 + np.array(match))

    def _validate_inputs(self, smiles):
        for smart in smiles:
            if Chem.MolFromSmarts(smart) is None:
                raise IOError(f"Invalid smarts pattern provided as a matching substructure: {smart}")
