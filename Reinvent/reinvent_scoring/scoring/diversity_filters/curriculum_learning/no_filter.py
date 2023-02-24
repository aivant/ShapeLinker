from copy import deepcopy

import numpy as np

from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.memory_record_dto import MemoryRecordDTO
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO


class NoFilter(BaseDiversityFilter):
    """Doesn't penalize compounds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, dto: UpdateDiversityFilterDTO) -> np.array:
        score_summary = deepcopy(dto.score_summary)
        scores = score_summary.total_score
        for i in score_summary.valid_idxs:
            if scores[i] >= self.parameters.minscore:
                #TODO: perhaps no validation is needed
                # smile = score_summary.scored_smiles[i]
                smile = self._chemistry.convert_to_rdkit_smiles(score_summary.scored_smiles[i])
                loggable_data = self._compose_loggable_data(dto.loggable_data[i]) if dto.loggable_data else ''
                memory_dto = MemoryRecordDTO(i, dto.step, scores[i], smile, smile, loggable_data,
                                             score_summary.scaffold_log)
                self._add_to_memory(memory_dto)
        return scores

