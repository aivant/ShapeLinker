from typing import List

import numpy as np

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.base_score_component import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary
from aizynthfinder.aizynthfinder import AiZynthExpander


class BuildingBlockAvailabilityComponent(BaseScoreComponent):
    """AiZynth one-step synthesis building block availability.

    Score is the ratio between
    the number of reactants in stock
    and the number of all reactants.

    If a molecule can be synthesized using different reactions,
    with different sets of reactants,
    the maximum ratio is used.

    This scoring component uses AiZynthFinder Expansion interface:
    https://molecularai.github.io/aizynthfinder/python_interface.html#expansion-interface
    """

    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

        configfile = self.parameters.specific_parameters[
            self.component_specific_parameters.AIZYNTH_CONFIG_FILE_PATH
        ]
        self._expander = self._set_up_expander(configfile)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        valid_smiles = self._chemistry.mols_to_smiles(molecules)
        score = self._score_smiles(valid_smiles)  # This is the main calculation.
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=score)

        return score_summary

    def _score_one_smi(self, smi: str) -> float:
        stock = self._expander.config.stock

        reactions = self._expander.do_expansion(smi)

        ratios = []  # Collect all, in case there are alternative reactions.
        for reaction_tuple in reactions:
            precursors = reaction_tuple[0].reactants[0]
            if len(precursors) == 0:
                # Corner case - no reactants.
                # It implies that the template was not applicable on the query molecule,
                # or there was an error, and it was not possible to produce reactants.
                ratios.append(0)  # Assign the lowest possible score.
            else:
                in_stock = [mol in stock for mol in precursors]
                ratio_in_stock = sum(in_stock) / len(in_stock)
                ratios.append(ratio_in_stock)

        if len(ratios) > 0:
            max_ratio = max(ratios)  # Take the best.
        else:
            max_ratio = 0  # No building blocks, return the lowest possible score.

        return max_ratio

    def _score_smiles(self, smiles: List[str]) -> np.ndarray:
        results = [self._score_one_smi(smi) for smi in smiles]
        return np.array(results)

    def _set_up_expander(self, configfile: str) -> AiZynthExpander:
        expander = AiZynthExpander(configfile=configfile)
        expander.expansion_policy.select_first()
        expander.filter_policy.select_first()
        expander.config.stock.select_first()
        return expander
