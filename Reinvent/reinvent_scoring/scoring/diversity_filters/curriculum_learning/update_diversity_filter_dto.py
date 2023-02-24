from dataclasses import dataclass
from typing import List

from reinvent_scoring.scoring.diversity_filters.curriculum_learning.loggable_data_dto import UpdateLoggableDataDTO
from reinvent_scoring.scoring.score_summary import FinalSummary


@dataclass
class UpdateDiversityFilterDTO:
    score_summary: FinalSummary
    loggable_data: List[UpdateLoggableDataDTO]
    step: int = 0
