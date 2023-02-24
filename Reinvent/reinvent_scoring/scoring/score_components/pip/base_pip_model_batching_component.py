import json
import os
from abc import abstractmethod
from typing import List

import numpy as np

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.enums import EnvironmentalVariablesEnum
from reinvent_scoring.scoring.score_components.pip.base_rest_component import BaseRESTComponent
from reinvent_scoring.configs.config import reinvent_scoring_config
import requests
import time

class BasePiPModelBatchingComponent(BaseRESTComponent):
    def __init__(self, parameters: ComponentParameters):
        self._environment_keys = EnvironmentalVariablesEnum()
        super().__init__(parameters)
        self._get_header = self._create_get_header()

    def _execute_request(self, request_url, data, header) -> dict:
        request = requests.post(request_url, json=data, headers=header)
        if request.status_code != 202:
            raise ValueError(
                f" Status: {request.status_code} Reason: ({request.reason})."
                f"Response content: {request.content}\n"
                f"Response content: {request.text}"
            )
        response = self._get_results(request)

        return response.json()

    def _parse_response(self, response_json: dict, data_size: int) -> np.array:
        compounds = response_json['jsonData']['data']
        results_raw = np.empty(data_size, dtype=np.float32)
        results_raw[:] = np.nan

        try:
            for compound in compounds:
                try:
                    index = int(compound["id"])
                    results_raw[index] = self._parse_single_compound(compound)

                except (ValueError, TypeError, KeyError):
                    pass  # If parsing failed, keep value NaN for this compound and continue.
        finally:
            return results_raw

    @abstractmethod
    def _parse_single_compound(self, compound):
        raise NotImplementedError("_parse_compound method is not implemented")

    def _format_data(self, smiles: List[str]) -> dict:
        molecules = [{"molData": smi, "id": f"{i}"} for i, smi in enumerate(smiles)]
        data = {
            "jsonData": {
                "data": molecules,
                "metadata": {
                    "molFormat":
                        "smiles"
                },
                "parameters": {

                }
            }
        }
        return data

    def _create_url(self, async_path: str) -> str:
        pip_url = self._get_enviornment_variable(self._environment_keys.PIP_URL)
        request_url = pip_url.format(async_path)
        return request_url

    def _create_get_url(self, component_name) -> str:
        pip_url = self._get_enviornment_variable(self._environment_keys.PIP_GET_RESULTS)
        request_url = pip_url.format(component_name)
        return request_url

    def _create_header(self) -> dict:
        pip_key = self._get_enviornment_variable(self._environment_keys.PIP_KEY)

        header = {
            'Content-Type': 'application/vnd.az.batch.v1+json', 'x-api-key': pip_key,
            'Accept': 'application/vnd.az.resultset.v1+json',
            'Prefer': 'respond-async'
        }
        return header

    def _create_get_header(self) -> dict:
        pip_key = self._get_enviornment_variable(self._environment_keys.PIP_KEY)
        header = {'Content-Type': 'application/json', 'x-api-key': pip_key}
        return header

    def _get_enviornment_variable(self, variable: str) -> str:
        try:
            return os.environ[variable]
        except KeyError:
            return self._retrieve_pip_key_from_config(variable)

    def _retrieve_pip_key_from_config(self, variable: str) -> str:
        try:
            environmental_variables = reinvent_scoring_config[self._environment_keys.ENVIRONMENTAL_VARIABLES]
            return environmental_variables[variable]
        except KeyError as ex:
            raise KeyError(f"Key {variable} not found in reinvent scoring config")

    def _get_results(self, response):
        async_location = response.headers.get('Location', None)
        url = self._create_get_url(async_location)

        time_threshold = 300
        time_delay = 1

        while time_delay > 0 :

            try:
                response = requests.get(url=url, headers=self._get_header)
                time_delay = int(response.headers.get('retry-after', 0))
                time_threshold-=time_delay
                time.sleep(time_delay)
                if time_threshold <= 0:
                    raise TimeoutError(f'Terminated since retrieving results took too long')
            except:
                raise ValueError(
                    f" Status: {response.status_code} Reason: ({response.reason})."
                    f"Response content: {response.content}\n"
                    f"Response content: {response.text}"
                )
        result = response
        return result