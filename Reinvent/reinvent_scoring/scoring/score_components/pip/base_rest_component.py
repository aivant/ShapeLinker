from abc import abstractmethod
from typing import List

import numpy as np

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary


class BaseRESTComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._request_url = self._create_url(self.parameters.component_type)
        self._request_header = self._create_header()

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        valid_smiles = self._chemistry.mols_to_smiles(molecules)
        score, raw_score = self._score_smiles(valid_smiles)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)

        return score_summary

    def _score_smiles(self, smiles: List[str]) -> np.array:
        response = self._post_request(self._request_url, smiles, self._request_header)
        results_raw = self._parse_response(response, len(smiles))
        results = self._apply_score_transformation(results_raw)

        return results, results_raw

    def _post_request(self, url, smiles, header):
        data = self._format_data(smiles)
        result = self._execute_request(url, data, header)

        return result

    @abstractmethod
    def _format_data(self, smiles: List[str]) -> dict:
        raise NotImplementedError("_format_data method is not implemented")

    @abstractmethod
    def _execute_request(self, request_url, data, header) -> dict:
        raise NotImplementedError("_execute_request method is not implemented")

    @abstractmethod
    def _parse_response(self, response_json: dict, data_size: int) -> np.array:
        raise NotImplementedError("_parse_response method is not implemented")

    def _apply_score_transformation(self, results_raw: np.array) -> np.array:
        """Returns np.array with non-NaN elements transformed by transformation function, and all NaN elements
        transformed into 0. """
        valid_mask = ~np.isnan(results_raw)
        results_raw_valid = results_raw[valid_mask]
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        results_transformed = self._transformation_function(results_raw_valid, transform_params)
        results = np.zeros(len(results_raw), dtype=np.float32)
        results[valid_mask] = results_transformed

        return results

    @abstractmethod
    def _create_url(self, component_name) -> str:
        raise NotImplementedError("_create_url method is not implemented")

    @abstractmethod
    def _create_header(self) -> dict:
        raise NotImplementedError("_create_header method is not implemented")
