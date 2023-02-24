import numpy as np
from typing import List

from reinvent_scoring.scoring.utils import _is_development_environment

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.structural.base_structural_component import BaseStructuralComponent


class DockStream(BaseStructuralComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._configuration_path = self.parameters.specific_parameters[self.component_specific_parameters.DOCKSTREAM_CONFPATH]
        self._docker_script_path = self.parameters.specific_parameters[self.component_specific_parameters.DOCKSTREAM_DOCKERSCRIPTPATH]
        self._environment_path = self.parameters.specific_parameters[self.component_specific_parameters.DOCKSTREAM_ENVPATH]

    def _add_debug_mode_if_selected(self, command):
        if self.parameters.specific_parameters.get(self.component_specific_parameters.DOCKSTREAM_DEBUG, False)\
                or _is_development_environment():
            command = ' '.join([command, "-debug"])
        return command

    def _create_command(self, smiles: List[str], step):
        concat_smiles = '"' + ';'.join(smiles) + '"'
        command = ' '.join([self._environment_path,
                            self._docker_script_path,
                            "-conf", self._configuration_path,
                            "-output_prefix", self._get_step_string(step),
                            "-smiles", concat_smiles,
                            "-print_scores"])

        # check, if DockStream is to be executed in debug mode, which will cause its loggers to print out
        # much more detailed information
        command = self._add_debug_mode_if_selected(command)
        return command

    def _calculate_score(self, smiles: List[str], step) -> np.array:
        # create the external command
        command = self._create_command(smiles, step)

        # send the batch smiles and retrieve the result as a list of strings
        results = self._send_request_with_stepwize_read(command, len(smiles))

        # note: some ligands might have failed in DockStream (embedding or docking) although they are valid
        #       RDKit molecules -> "docker.py" will return "NA"'s for failed molecules, as '0' could be a perfectly
        #       normal value; anything that cannot be cast to a floating point number will result in '0'
        scores = []
        for score in results:
            try:
                score = float(score)
            except ValueError:
                score = 0
            scores.append(score)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)

        return np.array(transformed_scores), np.array(scores)

    def _parse_result(self, result):
        return str(result).strip()
