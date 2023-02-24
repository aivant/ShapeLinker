from typing import List, Dict

import pandas as pd

from reinvent_scoring.scoring.diversity_filters.curriculum_learning.column_names_enum import ColumnNamesEnum
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.memory_record_dto import MemoryRecordDTO
from reinvent_scoring.scoring.score_summary import ComponentSummary
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class DiversityFilterMemory:

    def __init__(self):
        self._sf_component_name = ScoringFunctionComponentNameEnum()
        self._column_name = ColumnNamesEnum()
        df_dict = {self._column_name.STEP: [], self._column_name.SCAFFOLD: [], self._column_name.SMILES: [],
                   self._column_name.METADATA: []}
        self._memory_dataframe = pd.DataFrame(df_dict)

    def update(self, dto: MemoryRecordDTO):
        component_scores = {c.parameters.name: float(c.total_score[dto.id]) for c in dto.components}
        component_scores = self._include_raw_score(dto.id, component_scores, dto.components)
        component_scores[self._sf_component_name.TOTAL_SCORE] = float(dto.score)
        if not self.smiles_exists(dto.smile): self._add_to_memory_dataframe(dto, component_scores)

    def _add_to_memory_dataframe(self, dto: MemoryRecordDTO, component_scores: Dict):
        data = []
        headers = []
        for name, score in component_scores.items():
            headers.append(name)
            data.append(score)
        headers.append(self._column_name.STEP)
        data.append(dto.step)
        headers.append(self._column_name.SCAFFOLD)
        data.append(dto.scaffold)
        headers.append(self._column_name.SMILES)
        data.append(dto.smile)
        headers.append(self._column_name.METADATA)
        data.append(dto.loggable_data)
        new_data = pd.DataFrame([data], columns=headers)
        self._memory_dataframe = pd.concat([self._memory_dataframe, new_data], ignore_index=True, sort=False)

    def get_memory(self) -> pd.DataFrame:
        return self._memory_dataframe

    def set_memory(self, memory: pd.DataFrame):
        self._memory_dataframe = memory

    def smiles_exists(self, smiles: str):
        if len(self._memory_dataframe) == 0:
            return False
        return smiles in self._memory_dataframe[self._column_name.SMILES].values

    def scaffold_instances_count(self, scaffold: str):
        return (self._memory_dataframe[self._column_name.SCAFFOLD].values == scaffold).sum()

    def number_of_scaffolds(self):
        return len(set(self._memory_dataframe[self._column_name.SCAFFOLD].values))

    def number_of_smiles(self):
        return len(set(self._memory_dataframe[self._column_name.SMILES].values))

    def _include_raw_score(self, indx: int, component_scores: dict, components: List[ComponentSummary]):
        raw_scores = {f'raw_{c.parameters.name}': float(c.raw_score[indx]) for c in components if
                      c.raw_score is not None}
        all_scores = {**component_scores, **raw_scores}
        return all_scores
