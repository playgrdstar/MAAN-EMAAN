## Learning Semantically Rich Network-Based Multi-Modal Mobile User Interface Embeddings

### Overview:
This repository contains a Pytorch implementation of the MAAN and EMAAN model proposed in the paper Learning Semantically Rich Network-Based Multi-Modal Mobile User Interface Embeddings

### Requirements:

The key libraries required are Pytorch, Numpy, Pandas, Scipy, Networkx, DGL. See requirements.txt.

### Datasets

Due to Github's size constraints (as the data is >100MB), please email me (at playgrdstar(at)gmail(dot)com) for the data files that should be placed in this folder. Data should be placed in the ``data/RICO_N`` and the  ``data/RICO_M_W`` folder. 

### Repository Organization
- ``data/RICO_N`` and ``data/RICO_M_W` should contain the pre-processed datasets. 
- ``models.py`` contains the models
- ``helpers.py`` contains utility functions
- ``maan.py`` is the script that can be used to train the MAAN model 
- ``emaan.py`` is the script that can be used to train the EMAAN model (As the network/graph here is rather large, you may need a GPU with more RAM.

### Running the code
Run either of the commands below.
```
python maan.py 
python emaan.py
```

## Citation

If you use this repository, e.g., the code and the datasets, in your research, please cite the following paper:
```
@article{10.1145/3533856,
    author = {Ang, Gary and Lim, Ee-Peng},
    title = {Learning Semantically Rich Network-Based Multi-Modal Mobile User Interface Embeddings},
    year = {2022},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    issn = {2160-6455},
    url = {https://doi.org/10.1145/3533856},
    doi = {10.1145/3533856},
    journal = {ACM Trans. Interact. Intell. Syst.},
    month = {apr},
}
```