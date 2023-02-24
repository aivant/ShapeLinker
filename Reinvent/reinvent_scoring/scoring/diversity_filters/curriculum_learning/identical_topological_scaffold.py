from copy import deepcopy

import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold

from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.memory_record_dto import MemoryRecordDTO
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO


class IdenticalTopologicalScaffold(BaseDiversityFilter):
    """Penalizes compounds based on exact Topological Scaffolds previously generated."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, dto: UpdateDiversityFilterDTO) -> np.array:
        score_summary = deepcopy(dto.score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles

        for i in score_summary.valid_idxs:
            smile = self._chemistry.convert_to_rdkit_smiles(smiles[i])
            scaffold = self._calculate_scaffold(smile)
            scores[i] = 0 if self._smiles_exists(smile) else scores[i]

            if scores[i] >= self.parameters.minscore:
                loggable_data = self._compose_loggable_data(dto.loggable_data[i]) if dto.loggable_data else ''
                memory_dto = MemoryRecordDTO(i, dto.step, scores[i], smile, scaffold, loggable_data,
                                             score_summary.scaffold_log)
                self._add_to_memory(memory_dto)
                scores[i] = self._penalize_score(scaffold, scores[i])

        return scores

    def _calculate_scaffold(self, smile):
        mol = self._chemistry.smile_to_mol(smile)
        if mol:
            try:
                scaffold = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
                scaffold_smiles = self._chemistry.mol_to_smiles(scaffold)
            except ValueError:
                scaffold_smiles = ''
        else:
            scaffold_smiles = ''
        return scaffold_smiles
