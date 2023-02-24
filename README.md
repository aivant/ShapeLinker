# ShapeLinker
[ShapeLinker](link_to_preprint) is a method for the shape-conditioned ***de novo*** linker design for PROTACs. It is based on [Link-INVENT](https://chemrxiv.org/engage/chemrxiv/article-details/62628b2debac3a61c7debf31) and uses reinforcement learning to steer the linker generation towards a desired query shape. 
The main code is adapted from [Reinvent v3.2](https://github.com/MolecularAI/Reinvent) to include the shape alignment scoring.

## Requirements
* Multi-parameter optimization using shape alignment requires the [point_cloud_methods](link_to_repo) repository.
* Only works on Cuda-enabled GPU.

### Installation
1. Create ShapeLinker conda environment
```
conda env create -f env.yml
```
2. Create ```point_cloud_methods``` environment:
```
conda env create -f shape_align.yml
```

## Data
Download data from [link_to_storage](link) and store in ```utils/data```. This data dump includes:

* ```protacdb_extlinker_model_align.pth```: Trained model for shape alignment
* ```pdb_systems_data.csv```: Processed data for the investigated crystal structures
* ```protacdb_extended_linkers.csv```: Processed PROTAC-DB data
* folder ```xtal_poses```: Processed and fragmented crystal structures

## Notebooks
The notebooks (folder ```notebooks```) used here were adapted from [ReinventCommunity](https://github.com/MolecularAI/ReinventCommunity) and help with preparing runs for RL or sampling. More modes are available on their GitHub repo.

The folder ```utils/postprocessing```contains more useful jupyter notebooks allowing the postprocessing and evaluation of the generated data.