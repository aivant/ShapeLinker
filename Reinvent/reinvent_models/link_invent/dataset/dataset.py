# coding=utf-8
from typing import List, Tuple

import torch
import torch.utils.data as tud
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from reinvent_models.link_invent.model_vocabulary.model_vocabulary import ModelVocabulary


class Dataset(tud.Dataset):
    """Dataset that takes a list of SMILES only."""

    def __init__(self, smiles_list, model_vocabulary: ModelVocabulary):
        """
        Instantiates a Dataset.
        :param smiles_list: A list with SMILES strings.
        :param model_vocabulary: A ModelVocabulary object.
        :return:
        """
        self._model_vocabulary = model_vocabulary

        self._encoded_list = []
        for smi in smiles_list:
            enc = self._model_vocabulary.encode(smi)

            if enc is not None:
                self._encoded_list.append(enc)
            else:
                pass
                # TODO log theses cases

    def __getitem__(self, i):
        return torch.tensor(self._encoded_list[i], dtype=torch.long)  # pylint: disable=E1102

    def __len__(self):
        return len(self._encoded_list)

    @classmethod
    def collate_fn(cls, encoded_seqs):
        return cls._pad_batch(encoded_seqs)

    @staticmethod
    def _pad_batch(encoded_seqs: List) -> Tuple[Tensor, Tensor]:
        """
        Pads a batch.
        :param encoded_seqs: A list of encoded sequences.
        :return: A tensor with the sequences correctly padded.
        """
        seq_lengths = torch.tensor([len(seq) for seq in encoded_seqs], dtype=torch.int64)
        if torch.cuda.is_available():
            return pad_sequence(encoded_seqs, batch_first=True).cuda(), seq_lengths.cuda()
        else:
            return pad_sequence(encoded_seqs, batch_first=True), seq_lengths



