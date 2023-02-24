from typing import List
import torch

from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.reinvent_core.models.model import Model


class ReinventCoreAdapter(GenerativeModelBase):

    def __init__(self, path_to_file: str, mode: str):
        model_regime = GenerativeModelRegimeEnum()
        mode = mode == model_regime.INFERENCE
        self.generative_model = Model.load_from_file(path_to_file, mode)
        self.vocabulary =  self.generative_model.vocabulary
        self.tokenizer =  self.generative_model.tokenizer
        self.max_sequence_length =  self.generative_model.max_sequence_length
        self.network =  self.generative_model.network

    def save_to_file(self, path):
        self.generative_model.save(path)

    def likelihood(self, sequences):
        return self.generative_model.likelihood(sequences)

    def sample(self, batch_size):
        return self.generative_model.sample_sequences_and_smiles(batch_size, sampling_type = 'multinomial')

    def set_mode(self, mode: str):
        self.generative_model.set_mode(mode)

    def get_network_parameters(self):
        return self.generative_model.get_network_parameters()

    def get_vocabulary(self):
        return self.vocabulary

    def likelihood_smiles(self, smiles: List[str])-> torch.Tensor:
        return self.generative_model.likelihood_smiles(smiles)
