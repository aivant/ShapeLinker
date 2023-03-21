from typing import List
import numpy as np

from reinvent_chemistry.link_invent.bond_breaker import BondBreaker
from reinvent_chemistry.link_invent.attachment_point_modifier import AttachmentPointModifier

from reinvent_scoring.scoring.score_components.shape_scoring.shape_alignment import ShapeAlignment
from reinvent_scoring.scoring.enums.shape_alignment_specific_parameters_enum import ShapeAlignmentSpecificParametersEnum
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.link_invent.base_link_invent_component import BaseLinkInventComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary

VALID_QUERY_TYPES = ["sdf", "mol2", "smiles"]

class LinkerShapeScoring(BaseLinkInventComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._alignment = ShapeAlignment()
        self._shape_specific_params = ShapeAlignmentSpecificParametersEnum()
        self._bond_breaker = BondBreaker()
        self._attachment_point_modifier = AttachmentPointModifier()

        self.query_type = self.parameters.specific_parameters.get(self._shape_specific_params.QUERY_TYPE, "smiles")
        if self.query_type not in VALID_QUERY_TYPES:
            raise ValueError("Query type not recognized. Valid query types are: {}".format(VALID_QUERY_TYPES))

        self.model_path = self.parameters.specific_parameters.get(self.component_specific_parameters.MODEL_PATH, "")
        self.query = self.parameters.specific_parameters.get(self._shape_specific_params.QUERY, "")
        self.alignment_env = self.parameters.specific_parameters.get(self._shape_specific_params.ALIGNMENT_ENV, "")
        self.num_conformers = self.parameters.specific_parameters.get(self._shape_specific_params.NUM_CONFORMERS, 4)
        self.poses_folder = self.parameters.specific_parameters.get(self._shape_specific_params.POSES_FOLDER, "")
        self.es_weight = self.parameters.specific_parameters.get(self._shape_specific_params.ES_WEIGHT, 0.0)
        self.get_ext_linker = self.parameters.specific_parameters.get(self._shape_specific_params.GET_EXT_LINKER, False)
        self.correct_flipping = self.parameters.specific_parameters.get(self._shape_specific_params.CORRECT_FLIPPING, False)
        if not self.get_ext_linker: 
            # only works with extended linker due to availability of substructure matches to query
            self.correct_flipping = False

    def calculate_score(self, labeled_molecules: List, step=-1) -> ComponentSummary:
        linker_mols = [self._get_linker_mol(mol, self.get_ext_linker) for mol in labeled_molecules]
        # get index of None
        invalid_idx = [i for i, x in enumerate(linker_mols) if x is None]
        # remove None
        linker_mols = [x for x in linker_mols if x is not None]
        valid_smiles = self._chemistry.mols_to_smiles(linker_mols)
        raw_scores = self._alignment.calculate_alignment_score(self.query, valid_smiles, self.model_path, 
                                                    self.query_type, self.alignment_env, self.num_conformers,
                                                    self.poses_folder, self.es_weight, step, mode = 'linkinvent',
                                                    correct_flipping = self.correct_flipping)
        for i in invalid_idx:
            penalty = 99
            raw_scores.insert(i, penalty)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        scores = self._transformation_function(raw_scores, transform_params)
        score_summary = ComponentSummary(total_score=np.array(scores, dtype=np.float32), parameters=self.parameters, raw_score=np.array(raw_scores, dtype=np.float32))
        return score_summary
    
    def calculate_score_for_step(self, molecules: List, step=-1) -> ComponentSummary:
        return self.calculate_score(molecules, step)
    
    def _get_linker_mol(self, labeled_mol, get_ext_linker = False):
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol, get_input_frags=get_ext_linker)

        if get_ext_linker:
            input_frags = []
            for frag in linker_mol:
                input_frags.append(self._attachment_point_modifier.cap_linker_with_hydrogen(frag))
            linker_mol = self._bond_breaker.generate_ext_linker(labeled_mol, input_frags[0], input_frags[1])
            if linker_mol is None:
                return None
        else:
            linker_mol = self._attachment_point_modifier.cap_linker_with_hydrogen(linker_mol)
        return linker_mol
    
    def _calculate_linker_property(self, labeled_mol):
        raise NotImplementedError("_calculate_linker_property method is not implemented")