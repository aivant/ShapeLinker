from typing import List, Union, Any

import torch
from dacite import from_dict
from torch import nn as tnn

from reinvent_models.link_invent.dto import LinkInventModelParameterDTO
from reinvent_models.link_invent.dto import SampledSequencesDTO
from reinvent_models.link_invent.model_vocabulary.paired_model_vocabulary import PairedModelVocabulary
from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.link_invent.networks import EncoderDecoder


class LinkInventModel:
    def __init__(self, vocabulary: PairedModelVocabulary, network: EncoderDecoder,
                 max_sequence_length: int = 256, no_cuda: bool = False, mode: str = ModelModeEnum().TRAINING):
        self.vocabulary = vocabulary
        self.network = network
        self.max_sequence_length = max_sequence_length

        self._model_modes = ModelModeEnum()

        self.set_mode(mode)
        if torch.cuda.is_available() and not no_cuda:
            self.device = torch.device("cuda")
            self.network.cuda()
        else:
            self.device = torch.device("cpu")
        self._nll_loss = tnn.NLLLoss(reduction="none", ignore_index=0)

    def set_mode(self, mode: str):
        if mode == self._model_modes.TRAINING:
            self.network.train()
        elif mode == self._model_modes.INFERENCE:
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    @classmethod
    def load_from_file(cls, path_to_file, mode: str = ModelModeEnum().TRAINING) -> Union[Any, GenerativeModelBase] :
        """
        Loads a model from a single file
        :param path_to_file: Path to the saved model
        :param mode: Mode in which the model should be initialized
        :return: An instance of the network
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        data = from_dict(LinkInventModelParameterDTO, torch.load(path_to_file, map_location=device))
        network = EncoderDecoder(**data.network_parameter)
        network.load_state_dict(data.network_state)
        model = LinkInventModel(vocabulary=data.vocabulary, network=network,
                                max_sequence_length=data.max_sequence_length, mode=mode)
        return model

    def save_to_file(self, path_to_file):
        """
        Saves the model to a file.
        :param path_to_file: Path to the file which the model will be saved to.
        """
        data = LinkInventModelParameterDTO(vocabulary=self.vocabulary, max_sequence_length=self.max_sequence_length,
                                           network_parameter=self.network.get_params(),
                                           network_state=self.network.state_dict())
        torch.save(data.__dict__, path_to_file)

    def likelihood(self, warheads_seqs, warheads_seq_lengths, linker_seqs, linker_seq_lengths):
        """
        Retrieves the likelihood of warheads and their respective linker.
        :param warheads_seqs: (batch, seq) A batch of padded scaffold sequences.
        :param warheads_seq_lengths: The length of the scaffold sequences (for packing purposes).
        :param linker_seqs: (batch, seq) A batch of decorator sequences.
        :param linker_seq_lengths: The length of the decorator sequences (for packing purposes).
        :return:  (batch) Log likelihood for each item in the batch.
        """

        # NOTE: the decoration_seq_lengths have a - 1 to prevent the end token to be forward-passed.
        logits = self.network(warheads_seqs, warheads_seq_lengths, linker_seqs,
                              linker_seq_lengths - 1)  # (batch, seq - 1, voc)
        log_probs = logits.log_softmax(dim=2).transpose(1, 2)  # (batch, voc, seq - 1)
        return self._nll_loss(log_probs, linker_seqs[:, 1:]).sum(dim=1)  # (batch)

    @torch.no_grad()
    def sample(self, inputs, input_seq_lengths, temperature = 1.0) -> List[SampledSequencesDTO]:
        """
        Samples as many linker as warhead pairs in the tensor.
        :param inputs: A tensor with the warheads to sample already encoded and padded.
        :param input_seq_lengths: A tensor with the length of the warheads.
        :return: a sampled sequence dto with input_smi, output_smi and nll
        """
        batch_size = inputs.size(0)
        
        input_vector = torch.full((batch_size, 1), self.vocabulary.target.vocabulary["^"],
                                  dtype=torch.long).to(self.device)  # (batch, 1)
        seq_lengths = torch.ones(batch_size)  # (batch)
        encoder_padded_seqs, hidden_states = self.network.forward_encoder(inputs, input_seq_lengths)
        nlls = torch.zeros(batch_size).to(self.device)
        not_finished = torch.ones(batch_size, 1, dtype=torch.long).to(self.device)
        sequences = []
        for _ in range(self.max_sequence_length - 1):
            logits, hidden_states, _ = self.network.forward_decoder(
                input_vector, seq_lengths, encoder_padded_seqs, hidden_states)  # (batch, 1, voc)
            logits = logits / temperature
            probs = logits.softmax(dim=2).squeeze(dim=1)  # (batch, voc)
            log_probs = logits.log_softmax(dim=2).squeeze(dim=1)  # (batch, voc)
            input_vector = torch.multinomial(probs, 1) * not_finished  # (batch, 1)
            sequences.append(input_vector)
            nlls += self._nll_loss(log_probs, input_vector.squeeze(dim=1))
            not_finished = (input_vector > 1).type(torch.long)  # 0 is padding, 1 is end token
            if not_finished.sum() == 0:
                break

        linker_smiles_list = [self.vocabulary.target.decode(seq) for seq in torch.cat(sequences, 1).data.cpu().numpy()]
        warheads_smiles_list = [self.vocabulary.input.decode(seq) for seq in inputs.data.cpu().numpy()]

        result = [SampledSequencesDTO(warheads, linker, nll) for warheads, linker, nll in
                  zip(warheads_smiles_list, linker_smiles_list, nlls.data.cpu().numpy().tolist())]
        return result

    def get_network_parameters(self):
        return self.network.parameters()

    def get_vocabulary(self):
        return self.vocabulary
