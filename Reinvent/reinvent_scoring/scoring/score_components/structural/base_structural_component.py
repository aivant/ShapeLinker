import io
import subprocess
from abc import abstractmethod
from typing import List

import numpy as np

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary


class BaseStructuralComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def calculate_score_for_step(self, molecules: List, step=-1) -> ComponentSummary:
        return self.calculate_score(molecules, step)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        valid_smiles = self._chemistry.mols_to_smiles(molecules)
        score, raw_score = self._calculate_score(valid_smiles, step)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _get_step_string(self, step) -> str:
        if step == -1:
            return "\"\""
        return "".join(["\"e", str(step).zfill(4), "_\""])

    @abstractmethod
    def _calculate_score(self, smiles: List[str], step) -> np.array:
        raise NotImplementedError("_calculate_score method is not implemented")

    @abstractmethod
    def _create_command(self, input_file, step) -> str:
        raise NotImplementedError("_create_command method is not implemented")

    def _send_request_with_stepwize_read(self, command, data_size: int):
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              shell=True) as proc:
            wrapped_proc_in = io.TextIOWrapper(proc.stdin, 'utf-8')
            wrapped_proc_out = io.TextIOWrapper(proc.stdout, 'utf-8')
            result = [self._parse_result(wrapped_proc_out.readline()) for i in range(data_size)]
            wrapped_proc_in.close()
            wrapped_proc_out.close()
            proc.wait()
            proc.terminate()
        return result

    @abstractmethod
    def _parse_result(self, result) -> str:
        raise NotImplementedError("_parse_result method is not implemented")
