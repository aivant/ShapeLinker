from dataclasses import dataclass

import torch


@dataclass
class LinkInventBatchDTO:
    input: torch.Tensor
    output: torch.Tensor
