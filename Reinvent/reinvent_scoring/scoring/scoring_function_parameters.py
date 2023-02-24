from dataclasses import dataclass
from typing import List


@dataclass
class ScoringFunctionParameters:
    name: str
    parameters: List[dict]
    parallel: bool = False