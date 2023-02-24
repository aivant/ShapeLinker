from typing import List
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.pip.pip_prediction_component import PiPPredictionComponent


class QptunaPiPModelComponent(PiPPredictionComponent):

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
                    "artifact": self.parameters.specific_parameters.get(self.component_specific_parameters.ARTIFACT)
                }
            }
        }

        return data
