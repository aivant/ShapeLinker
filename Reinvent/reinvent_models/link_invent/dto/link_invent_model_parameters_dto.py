from dataclasses import dataclass

from reinvent_models.link_invent.model_vocabulary.paired_model_vocabulary import PairedModelVocabulary


@dataclass
class LinkInventModelParameterDTO:
    vocabulary: PairedModelVocabulary
    max_sequence_length: int
    network_parameter: dict
    network_state: dict