from abc import ABC, abstractmethod

from typing import Dict, List


class BaseModelContainer(ABC):

    @abstractmethod
    def predict(self, molecules: List, parameters: Dict):
        raise NotImplementedError("'predict' method is not implemented !")
