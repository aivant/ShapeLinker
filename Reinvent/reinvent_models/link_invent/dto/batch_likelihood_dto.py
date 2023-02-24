from dataclasses import dataclass
from typing import Union
import torch

from reinvent_models.link_invent.dto.linkinvent_batch_dto import LinkInventBatchDTO


@dataclass
class BatchLikelihoodDTO:
    batch: Union[LinkInventBatchDTO]
    likelihood: torch.Tensor