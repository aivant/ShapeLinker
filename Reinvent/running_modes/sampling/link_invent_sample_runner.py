import os
import pandas as pd
from typing import List
import torch.utils.data as tud

from reinvent_chemistry.library_design.bond_maker import BondMaker
from reinvent_chemistry.library_design.attachment_points import AttachmentPoints
from reinvent_chemistry import Conversions, TransformationTokens

from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.link_invent.dataset.dataset import Dataset
from reinvent_models.link_invent.link_invent_model import LinkInventModel

from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.reinforcement_learning.actions.link_invent_sample_model import LinkInventSampleModel
from running_modes.reinforcement_learning.actions import LinkInventLikelihoodEvaluation
from running_modes.reinforcement_learning.dto.sampled_sequences_dto import SampledSequencesDTO
from running_modes.configurations.sampling.link_invent_sampling_configuration import LinkInventSamplingConfiguration
from running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from running_modes.sampling.logging.sampling_logger import SamplingLogger


class SampleLinkInventModelRunner(BaseRunningMode):

    def __init__(self, main_config: GeneralConfigurationEnvelope, config: LinkInventSamplingConfiguration) -> None:
        super().__init__()
        self.model = LinkInventModel.load_from_file(config.model_path)
        self.output_path = config.output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.sample_size = config.num_samples
        self.batch_size = config.batch_size
        self.randomize = config.randomize_warheads
        self.warheads = config.warheads
        self.logger = SamplingLogger(main_config)
        self.with_likelihood = config.with_likelihood
        self.temperature = config.temperature

        self.conversions = Conversions()
        self.attachment_points = AttachmentPoints()
        self.tokens = TransformationTokens()
        self.bond_maker = BondMaker()

    def _open_output(self, path):
        try:
            os.mkdir(os.path.dirname(path))
        except FileExistsError:
            pass
        return open(path, "wt+")

    def run(self):
        warheads_list = self._randomize_warheads(self.warheads) if self.randomize else self.warheads
        clean_warheads = [self.attachment_points.remove_attachment_point_numbers(warheads) for warheads in warheads_list]
        dataset = Dataset(clean_warheads, self.model.get_vocabulary().input)
        data_loader = tud.DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=dataset.collate_fn)

        for batch in data_loader:
            sampled_sequences = []
            for _ in range(self.sample_size):
                sampled_sequences.extend(self.model.sample(*batch, self.temperature))
        
        molecules = self._join_linker_and_warheads(sampled_sequences)

        mol_smiles = []
        for molecule in molecules:
            try:
                smiles_str = self.conversions.mol_to_smiles(molecule) if molecule else "INVALID"
            except RuntimeError as exception:
                smiles_str = "INVALID"
            finally:
                mol_smiles.append(smiles_str)
        
        df_sampled = pd.DataFrame({'molecules': mol_smiles, 'input': [x.input for x in sampled_sequences], 'linker': [x.output for x in sampled_sequences]})
        if self.with_likelihood:
            df_sampled['likelihood'] = [x.nll for x in sampled_sequences]

        df_sampled.to_csv(self.output_path, index=False)
        
        self.logger.log_out_input_configuration()

    def _randomize_warheads(self, warhead_pair_list: List[str]):
        randomized_warhead_pair_list = []
        for warhead_pair in warhead_pair_list:
            warhead_list = warhead_pair.split(self.tokens.ATTACHMENT_SEPARATOR_TOKEN)
            warhead_mol_list = [self.conversions.smile_to_mol(warhead) for warhead in warhead_list]
            warhead_randomized_list = [self.conversions.mol_to_random_smiles(mol) for mol in warhead_mol_list]
            warhead_pair_randomized = self.tokens.ATTACHMENT_SEPARATOR_TOKEN.join(warhead_randomized_list)
            randomized_warhead_pair_list.append(warhead_pair_randomized)
        return randomized_warhead_pair_list

    def _join_linker_and_warheads(self, sampled_sequences: List[SampledSequencesDTO], keep_labels=False):
        molecules = []
        for sample in sampled_sequences:
            linker = self.attachment_points.add_attachment_point_numbers(sample.output, canonicalize=False)
            molecule = self.bond_maker.join_scaffolds_and_decorations(linker, sample.input,
                                                                       keep_labels_on_atoms=keep_labels)
            molecules.append(molecule)
        return molecules