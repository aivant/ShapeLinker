from typing import List, Dict, Any

import numpy as np
from reinvent_chemistry.conversions import Conversions

from reinvent_scoring.scoring.predictive_model.base_model_container import BaseModelContainer


class OptunaModelContainer(BaseModelContainer):
    def __init__(self, activity_model):
        """
        :type activity_model: scikit-learn object
        """
        self._activity_model = activity_model
        self._conversions = Conversions()

    def predict(self, molecules: List[Any], parameters: Dict) -> np.array:
        """
        Takes a list of smiles as input an predicts activities.
        :param molecules:
        :param parameters:
        :return:
        """

        if len(molecules) == 0:
            return np.empty([])

        smiles = [self._conversions.mol_to_smiles(mol) for mol in molecules]
        activity = self._activity_model.predict_from_smiles(smiles)

        return activity


