from copy import deepcopy
import numpy as np

from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.memory_record_dto import MemoryRecordDTO
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO


class NoFilterWithPenalty(BaseDiversityFilter):
    """Penalize previously generated compounds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, dto: UpdateDiversityFilterDTO) -> np.array:
        score_summary = deepcopy(dto.score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles

        for i in score_summary.valid_idxs:
            smiles[i] = self._chemistry.convert_to_rdkit_smiles(smiles[i])
            scores[i] = self.parameters.penalty_multiplier * scores[i] if self._smiles_exists(smiles[i]) else scores[i]

        for i in score_summary.valid_idxs:
            if scores[i] >= self.parameters.minscore:
                loggable_data = self._compose_loggable_data(dto.loggable_data[i])
                memory_dto = MemoryRecordDTO(i, dto.step, scores[i], smiles[i], smiles[i], loggable_data,
                                             score_summary.scaffold_log)
                self._add_to_memory(memory_dto)
        return scores
