from dataclasses import dataclass
from typing import List

@dataclass
class LinkInventSamplingConfiguration:
    model_path: str
    output_path: str
    warheads: List[str]
    num_samples: int = 1024
    batch_size: int = 128
    randomize_warheads: bool = False
    with_likelihood: bool = False
    temperature: float = 1.0
    