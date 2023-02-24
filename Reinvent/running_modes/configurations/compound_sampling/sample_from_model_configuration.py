from dataclasses import dataclass


@dataclass
class SampleFromModelConfiguration:
    model_path: str
    output_smiles_path: str
    num_smiles: int = 1024
    batch_size: int = 128
    with_likelihood: bool = False
    sampling_mode: str = 'multinomial'
    k: int = None
    p: float = 1.0
    temperature: float = 1.0
