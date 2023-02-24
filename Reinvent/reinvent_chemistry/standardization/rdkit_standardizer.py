from typing import Dict, List

from rdkit.Chem.rdmolfiles import MolToSmiles

from reinvent_chemistry import Conversions
from reinvent_chemistry.standardization.filter_configuration import FilterConfiguration
from reinvent_chemistry.standardization.filter_registry import FilterRegistry


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)

    import rdkit.rdBase as rkrb
    rkrb.DisableLog('rdApp.error')


disable_rdkit_logging()


class RDKitStandardizer:
    def __init__(self, filter_configs: List[FilterConfiguration], logger):
        self._conversions = Conversions()
        self._filter_configs = self._set_filter_configs(filter_configs)
        self._filters = self._load_filters(self._filter_configs)
        self._logger = logger

    def apply_filter(self, smile: str) -> str:
        molecule = self._conversions.smile_to_mol(smile)

        for config in self._filter_configs:
            if molecule:
                rdkit_filter = self._filters[config.name]
                if config.parameters:
                    molecule = rdkit_filter(molecule, **config.parameters)
                else:
                    molecule = rdkit_filter(molecule)
            else:
                message = f'filter: "{config.name}" detected the following SMILES string {smile} as invalid!'
                self._log_message(message)

        if not molecule:
            message = f'filter: "{self._filter_configs[-1].name}" detected the following SMILES string {smile} as invalid!'
            self._log_message(message)
            return None

        valid_smile = MolToSmiles(molecule, isomericSmiles=False)
        return valid_smile

    def _set_filter_configs(self, filter_configs):
        return filter_configs if filter_configs else [FilterConfiguration(name="default", parameters={})]

    def _load_filters(self, filter_configs: List[FilterConfiguration]) -> Dict:
        registry = FilterRegistry()
        return {filter_config.name: registry.get_filter(filter_config.name) for filter_config in filter_configs}

    def _log_message(self, message: str):
        if self._logger:
            self._logger.log_message(message)
