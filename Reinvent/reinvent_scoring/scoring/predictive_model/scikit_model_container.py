from typing import List, Dict

import numpy as np

from reinvent_chemistry.descriptors import Descriptors
from reinvent_scoring.scoring.predictive_model.base_model_container import BaseModelContainer


class ScikitModelContainer(BaseModelContainer):
    def __init__(self, activity_model, model_type: str, specific_parameters: {}):
        """
        :type activity_model: scikit-learn type of model object
        :type model_type: can be "classification" or "regression"
        """
        self._activity_model = activity_model
        self._model_type = model_type
        self._molecules_to_descriptors = self._load_descriptor(specific_parameters)

    def predict(self, molecules: List, parameters: Dict) -> np.array:
        """
        Takes as input RDKit molecules and uses a pickled scikit-learn model to predict activities.
        :param molecules: This is a list of rdkit.Chem.Mol objects
        :param parameters: Those are descriptor-specific parameters.
        :return: numpy.array with activity predictions
        """
        return self.predict_from_mols(molecules, parameters)

    def predict_from_mols(self, molecules: List, parameters: dict):
        if len(molecules) == 0:
            return np.empty([])
        fps = self._molecules_to_descriptors(molecules, parameters)
        activity = self.predict_from_fingerprints(fps)
        return activity

    def predict_from_fingerprints(self, fps):
        if self._model_type == "regression":
            activity = self._activity_model.predict(fps)
        else:
            predictions = self._activity_model.predict_proba(fps)
            activity = predictions[:, 1]

        return activity

    def _load_descriptor(self, parameters: {}):
        descriptors = Descriptors()
        descriptor = descriptors.load_descriptor(parameters)
        return descriptor