from typing import List

from torch.utils.data import DataLoader

from reinvent_models.lib_invent.models.dataset import DecoratorDataset
from reinvent_models.lib_invent.models.model import DecoratorModel
from reinvent_models.link_invent.dto.linkinvent_batch_dto import LinkInventBatchDTO
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.link_invent.dto import BatchLikelihoodDTO
from reinvent_models.link_invent.dto.sampled_sequence_dto import SampledSequencesDTO


class LibInventAdapter(GenerativeModelBase):

    def __init__(self, path_to_file: str, mode: str):
        self.generative_model = DecoratorModel.load_from_file(path_to_file, mode)
        self.vocabulary = self.generative_model.vocabulary
        self.max_sequence_length = self.generative_model.max_sequence_length
        self.network = self.generative_model.network

    def save_to_file(self, path):
        self.generative_model.save(path)

    def likelihood(self, scaffold_seqs, scaffold_seq_lengths, decoration_seqs, decoration_seq_lengths):
        return self.generative_model.likelihood(scaffold_seqs, scaffold_seq_lengths, decoration_seqs, decoration_seq_lengths)

    def sample(self, scaffold_seqs, scaffold_seq_lengths):
        return self.generative_model.sample_decorations(scaffold_seqs, scaffold_seq_lengths)

    def set_mode(self, mode: str):
        self.generative_model.set_mode(mode)

    def get_network_parameters(self):
        return self.generative_model.get_network_parameters()

    def get_vocabulary(self):
        return self.vocabulary

    def likelihood_smiles(self, sampled_sequence_list: List[SampledSequencesDTO]) -> BatchLikelihoodDTO:
        input_output_list = [[ss.input, ss.output] for ss in sampled_sequence_list]
        dataset = DecoratorDataset(input_output_list, self.vocabulary)
        dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=dataset.collate_fn, shuffle=False)

        for input_batch, output_batch in dataloader:
            likelihood = self.generative_model.likelihood(*input_batch, *output_batch)
            batch = LinkInventBatchDTO(input_batch, output_batch)
            dto = BatchLikelihoodDTO(batch, likelihood)
            return dto