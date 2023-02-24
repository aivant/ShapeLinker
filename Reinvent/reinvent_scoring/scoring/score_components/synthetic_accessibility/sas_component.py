import pickle
from typing import List

import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.Descriptors import ExactMolWt

from reinvent_chemistry import Descriptors

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_components.synthetic_accessibility.sascorer import calculateScore
from reinvent_scoring.scoring.score_summary import ComponentSummary


class SASComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.activity_model = self._load_model(parameters)
        self._descriptors = Descriptors()
        self.fp_parameters = dict(
            radius=3,
            size=4096,  # Descriptors class calls this parameter "size", RDKit calls it "nBits".
            use_features=False,  # RDKit has False as default, Descriptors class has True.
        )

    def calculate_score(self, molecules: List[Mol], step=-1) -> ComponentSummary:
        score = self.predict_from_molecules(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def predict_from_molecules(self, molecules: List[Mol]) -> np.ndarray:
        if len(molecules) == 0:
            return np.array([])

        descriptors = self._calculate_descriptors(molecules)

        # Normally, predict_proba takes a 2d array, one row per observation,
        # but a list of 1d arrays works too.
        sas_predictions = self.activity_model.predict_proba(descriptors)

        return sas_predictions[:, 1]

    def _load_model(self, parameters: ComponentParameters):
        try:
            # TODO: in the future should use self.component_specific_parameters.MODEL_PATH
            # model_path = self.parameters.specific_parameters.get(self.component_specific_parameters.MODEL_PATH, "")
            model_path = self.parameters.specific_parameters.get("saz_model_path", "")
            activity_model = self._load_scikit_model(model_path)
        except:
            # model_path = self.parameters.specific_parameters.get(self.component_specific_parameters.MODEL_PATH, "")
            model_path = self.parameters.specific_parameters.get("saz_model_path", "")
            raise Exception(f"The loaded file `{model_path}` isn't a valid scikit-learn model")
        return activity_model

    def _load_scikit_model(self, model_path: str):
        with open(model_path, "rb") as f:
            scikit_model = pickle.load(f)
        return scikit_model

    def _calculate_descriptors(self, molecules: List[Mol]) -> List[np.ndarray]:
        descriptors = [self._sas_descriptor(mol) for mol in molecules]
        return descriptors

    def _sas_descriptor(self, mol: Mol) -> np.ndarray:
        """Returns SAS descriptor for a molecule, to be used as input to SAS model.

        SAS descriptor consists of three parts:
            1. SA score by Ertl and Schuffenhauer (Novartis), part of RDKit, copied to this repo.
            2. Molecular weight.
            3. Morgan fingerprint, with counts (ECFP6).

        The three parts are concatenated into one 1d numpy array.
        """

        sascore = calculateScore(mol)
        molwt = ExactMolWt(mol)
        fp = self._fingerprint(mol)

        descriptor = np.concatenate([[sascore], [molwt], fp])

        return descriptor

    def _fingerprint(self, mol: Mol) -> np.ndarray:
        fps = self._descriptors.molecules_to_count_fingerprints([mol], parameters=self.fp_parameters)
        return fps[0]
