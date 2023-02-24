import json
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
from typing import List, Tuple

from reinvent_scoring.scoring.utils import _is_development_environment

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.console_invoked.base_console_invoked_component import BaseConsoleInvokedComponent


class Icolos(BaseConsoleInvokedComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._executor_path = self.parameters.specific_parameters[self.component_specific_parameters.ICOLOS_EXECUTOR_PATH]
        self._configuration_path = self.parameters.specific_parameters[self.component_specific_parameters.ICOLOS_CONFPATH]
        self._values_key = self.parameters.specific_parameters[self.component_specific_parameters.ICOLOS_VALUES_KEY]

    def _add_debug_mode_if_selected(self, command):
        if self.parameters.specific_parameters.get(self.component_specific_parameters.ICOLOS_DEBUG, False)\
                or _is_development_environment():
            command = ' '.join([command, "-debug"])
        return command

    def _create_command(self, step, input_json_path: str, output_json_path: str):
        # use "step" as well for the write-out
        global_variables = "".join(["\"input_json_path:", input_json_path, "\" ",
                                    "\"output_json_path:", output_json_path, "\" ",
                                    "\"step_id:", str(step), "\""])
        command = ' '.join([self._executor_path,
                            "-conf", self._configuration_path,
                            "--global_variables", global_variables])

        # check, if Icolos is to be executed in debug mode, which will cause its loggers to print out
        # much more detailed information
        command = self._add_debug_mode_if_selected(command)
        return command

    def _prepare_input_data_JSON(self, path: str, smiles: List[str]):
        """Needs to look something like:
           {
               "names": ["0", "1", "3"],
               "smiles": ["C#CCCCn1...", "CCCCn1c...", "CC(C)(C)CCC1(c2..."]
           }"""
        names = [str(idx) for idx in range(len(smiles))]
        input_dict = {"names": names,
                      "smiles": smiles}
        with open(path, 'w') as f:
            json.dump(input_dict, f, indent=4)

    def _select_values(self, data: dict) -> list:
        for value_dict in data["results"]:
            if self._values_key == value_dict[self.component_specific_parameters.ICOLOS_VALUES_KEY]:
                return value_dict["values"]
        return []

    def _parse_output_data_json(self, path: str) -> Tuple[List[str], List[float]]:
        """Needs to look something like:
           {
               "results": [{
                   "values_key": "docking_score",
                   "values": ["-5.88841", "-5.72676", "-7.30167"]},
                           {
                   "values_key": "shape_similarity",
                   "values": ["0.476677", "0.458017", "0.510676"]},
                           {
                   "values_key": "esp_similarity",
                   "values": ["0.107989", "0.119446", "0.100109"]}],
               "names": ["0", "1", "2"]
           }"""
        names_list = []
        values_list = []

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Output file {path} does not exist, indicating that execution of Icolos failed entirely. Check your setup and the log file.")

        with open(path, 'r') as f:
            data = f.read().replace("\r", "").replace("\n", "")
        data = json.loads(data)
        raw_values_list = self._select_values(data=data)

        for idx in range(len(data["names"])):
            names_list.append(data["names"][idx])
            try:
                score = float(raw_values_list[idx])
            except ValueError:
                score = 0
            values_list.append(score)

        return names_list, values_list

    def _execute_command(self, command: str, final_file_path: str = None):
        # execute the pre-defined command
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        # TODO: once logging is available here, check the return code
        # if result.returncode != 0:

        # wait in case the final file has not been written yet or is empty (in case of a filesystem delay / hick-up)
        if final_file_path is not None:
            for _ in range(5):
                if os.path.isfile(final_file_path) and os.path.getsize(final_file_path) > 0:
                    break
                else:
                    time.sleep(3)

    def _calculate_score(self, smiles: List[str], step) -> np.array:
        # make temporary folder and set input and output paths
        tmp_dir = tempfile.mkdtemp()
        input_json_path = os.path.join(tmp_dir, "input.json")
        output_json_path = os.path.join(tmp_dir, "output.json")

        # save the smiles in an Icolos compatible JSON file
        self._prepare_input_data_JSON(path=input_json_path, smiles=smiles)

        # create the external command
        command = self._create_command(step=step,
                                       input_json_path=input_json_path,
                                       output_json_path=output_json_path)

        # execute the Icolos component
        self._execute_command(command=command, final_file_path=output_json_path)

        # parse the output
        smiles_ids, scores = self._parse_output_data_json(path=output_json_path)

        # apply transformation
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)

        # clean up
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

        return np.array(transformed_scores), np.array(scores)
