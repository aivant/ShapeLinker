from dataclasses import dataclass
from typing import List

from reinvent_scoring.scoring.score_summary import ComponentSummary


@dataclass
class MemoryRecordDTO:
    id: int
    step: int
    score: float
    smile: str
    scaffold: str
    loggable_data: str
    components: List[ComponentSummary]
