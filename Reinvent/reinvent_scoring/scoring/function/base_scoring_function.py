from abc import ABC, abstractmethod
from typing import List

import numpy as np
from pathos.multiprocessing import ProcessPool

from reinvent_chemistry.conversions import Conversions
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.score_component_factory import ScoreComponentFactory
from reinvent_scoring.scoring.score_summary import ComponentSummary, FinalSummary
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum


def _update_raw_score(summary: ComponentSummary, query_length: int, valid_indices: List[int]) -> ComponentSummary:
    if summary.raw_score is not None:
        raw_score = np.full(query_length, 0, dtype=np.float32)
        assert len(valid_indices) == len(summary.raw_score)
        for idx, value in zip(valid_indices, summary.raw_score):
            raw_score[idx] = value
        summary.raw_score = raw_score
    return summary


def _update_total_score(summary: ComponentSummary, query_length: int, valid_indices: List[int]) -> ComponentSummary:
    total_score = np.full(query_length, 0, dtype=np.float32)
    assert len(valid_indices) == len(summary.total_score)
    for idx, value in zip(valid_indices, summary.total_score):
        total_score[idx] = value
    summary.total_score = total_score
    summary = _update_raw_score(summary, query_length, valid_indices)
    return summary


def parallel_run(component_smiles_pair):
    component = component_smiles_pair[0]
    molecules = component_smiles_pair[1]
    valid_indices = component_smiles_pair[2]
    smiles = component_smiles_pair[3]
    scores = component.calculate_score(molecules)
    scores = _update_total_score(scores, len(smiles), valid_indices)
    return scores


def parallel_run_for_step(component_smiles_pair):
    component = component_smiles_pair[0]
    molecules = component_smiles_pair[1]
    valid_indices = component_smiles_pair[2]
    smiles = component_smiles_pair[3]
    step = component_smiles_pair[4]
    scores = component.calculate_score_for_step(molecules, step)
    scores = _update_total_score(scores, len(smiles), valid_indices)
    return scores


class BaseScoringFunction(ABC):
    def __init__(self, parameters: List[ComponentParameters], parallel=False):
        self.component_enum = ScoringFunctionComponentNameEnum()
        self.component_specific_parameters = ComponentSpecificParametersEnum()
        self._chemistry = Conversions()
        factory = ScoreComponentFactory(parameters)
        self.scoring_components = factory.create_score_components()
        if parallel:
            self.get_final_score = self._parallel_final_score
            self.get_final_score_for_step = self._parallel_final_score_for_step

    def get_final_score_for_step(self, smiles: List[str], step: int) -> FinalSummary:
        molecules, valid_indices = self._chemistry.smiles_to_mols_and_indices(smiles)
        query_size = len(smiles)
        summaries = [_update_total_score(sc.calculate_score_for_step(molecules, step), query_size, valid_indices) for sc
                     in self.scoring_components]
        return self._score_summary(summaries, smiles, valid_indices)

    def get_final_score(self, smiles: List[str]) -> FinalSummary:
        molecules, valid_indices = self._chemistry.smiles_to_mols_and_indices(smiles)
        query_size = len(smiles)
        summaries = [_update_total_score(sc.calculate_score(molecules), query_size, valid_indices) for sc
                     in self.scoring_components]
        return self._score_summary(summaries, smiles, valid_indices)

    def _score_summary(self, summaries: List[ComponentSummary], smiles: List[str],
                       valid_indices: List[int]) -> FinalSummary:

        penalty = self._compute_penalty_components(summaries, smiles)
        non_penlaty = self._compute_non_penalty_components(summaries, smiles)
        product = penalty * non_penlaty
        final_summary = self._create_final_summary(product, summaries, smiles, valid_indices)

        return final_summary

    def _create_final_summary(self, final_score, summaries: List[ComponentSummary], smiles: List[str],
                              valid_indices: List[int]) -> FinalSummary:

        return FinalSummary(total_score=np.array(final_score, dtype=np.float32),
                            scored_smiles=smiles,
                            valid_idxs=valid_indices,
                            scaffold_log_summary=summaries)

    def _compute_penalty_components(self, summaries: List[ComponentSummary], smiles: List[str]):
        penalty = np.full(len(smiles), 1, dtype=np.float32)

        for summary in summaries:
            if self._component_is_penalty(summary):
                penalty = penalty * summary.total_score

        return penalty

    @abstractmethod
    def _compute_non_penalty_components(self, summaries: List[ComponentSummary], smiles: List[str]):
        raise NotImplementedError("_score_summary method is not implemented")

    def _component_is_penalty(self, summary: ComponentSummary) -> bool:
        return (summary.parameters.component_type == self.component_enum.MATCHING_SUBSTRUCTURE) or (
                summary.parameters.component_type == self.component_enum.CUSTOM_ALERTS)

    def _parallel_final_score(self, smiles: List[str]) -> FinalSummary:
        molecules, valid_indices = self._chemistry.smiles_to_mols_and_indices(smiles)
        component_smiles_pairs = [[component, molecules, valid_indices, smiles] for component in
                                  self.scoring_components]
        pool = ProcessPool(nodes=len(self.scoring_components))
        mapped_pool = pool.map(parallel_run, component_smiles_pairs)
        pool.clear()
        return self._score_summary(mapped_pool, smiles, valid_indices)

    def _parallel_final_score_for_step(self, smiles: List[str], step: int) -> FinalSummary:
        molecules, valid_indices = self._chemistry.smiles_to_mols_and_indices(smiles)
        component_smiles_pairs = [[component, molecules, valid_indices, smiles, step] for component in
                                  self.scoring_components]
        pool = ProcessPool(nodes=len(self.scoring_components))
        mapped_pool = pool.map(parallel_run_for_step, component_smiles_pairs)
        pool.clear()
        return self._score_summary(mapped_pool, smiles, valid_indices)
