import abc

import numpy as np
import pandas as pd
from reinvent_chemistry.conversions import Conversions

from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters, \
    DiversityFilterMemory
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.loggable_data_dto import UpdateLoggableDataDTO
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.memory_record_dto import MemoryRecordDTO
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO


class BaseDiversityFilter(abc.ABC):

    @abc.abstractmethod
    def __init__(self, parameters: DiversityFilterParameters):
        self.parameters = parameters
        self._diversity_filter_memory = DiversityFilterMemory()
        self._chemistry = Conversions()

    @abc.abstractmethod
    def update_score(self, update_dto: UpdateDiversityFilterDTO) -> np.array:
        raise NotImplementedError("The method 'update_score' is not implemented!")

    def get_memory_as_dataframe(self) -> pd.DataFrame:
        return self._diversity_filter_memory.get_memory()

    def set_memory_from_dataframe(self, memory: pd.DataFrame):
        self._diversity_filter_memory.set_memory(memory)

    def number_of_smiles_in_memory(self) -> int:
        return self._diversity_filter_memory.number_of_smiles()

    def number_of_scaffold_in_memory(self) -> int:
        return self._diversity_filter_memory.number_of_scaffolds()

    def update_bucket_size(self, bucket_size: int):
        self.parameters.bucket_size = bucket_size

    def _calculate_scaffold(self, smile):
        raise NotImplementedError

    def _smiles_exists(self, smile):
        return self._diversity_filter_memory.smiles_exists(smile)

    def _add_to_memory(self, memory_dto: MemoryRecordDTO):
        self._diversity_filter_memory.update(memory_dto)

    def _penalize_score(self, scaffold, score):
        """Penalizes the score if the scaffold bucket is full"""
        if self._diversity_filter_memory.scaffold_instances_count(scaffold) > self.parameters.bucket_size:
            score = 0.
        return score

    def _compose_loggable_data(self, dto: UpdateLoggableDataDTO):
        prior_likelihood = f'{dto.prior_likelihood}|' if dto.prior_likelihood else ''
        likelihood =  f'{dto.likelihood}|' if dto.likelihood else ''
        input =  f'{dto.input}|' if dto.input else ''
        output = f'{dto.output}' if dto.output else ''
        loggable_data = f'{prior_likelihood}{likelihood}{input}{output}'
        return loggable_data