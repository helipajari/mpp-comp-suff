# Evaluating the Faithfulness of GNN Explainers in Molecular Property Prediction with Comprehensiveness and Sufficiency 

 
## Summary
This repository contains the code implemented for the end project of a Bachelor in Computer Science and Engineering at TU Delft.
The associated written work is included in the repository.

The code is an extension of MolRep[1]. It adds an algorithm to extract subgraphs from a GNN explanation 
and a new metric to measure faithfulness, defined by two submetrics:
- Comprehensiveness
- Sufficiency

Comprehensiveness and sufficiency are originally defined in the BAGEL benchmark[2]. The formula modifications are original work.

## MolRep
The original MolRep repository is available at: https://github.com/biomed-AI/MolRep

## Installation
Set up a Python environment and install the following packages:
```
pip install networkx~=3.2.1
pip install numpy~=1.26.4
pip install ogb~=1.3.6
pip install pandas~=2.2.2
pip install PyYAML~=6.0.1
pip install rdkit~=2023.9.5
pip install scikit-learn~=1.4.2
pip install scipy==1.11.4
pip install six~=1.16.0
pip install torch~=2.3.0+cu118
pip install tqdm~=4.66.4
pip install transformers~=4.40.2
pip install xgboost~=2.0.3
pip install setuptools~=65.5.1
```

The datasets used in MolRep are included in this repository, but for a clean installation of MolRep please see README-molrep.md

## Use
All code has been implemented in `Comprehensiveness_Sufficiency.ipynb`, found in the `Examples` folder.

### On the first run, train a model and generate explanations:

1. Set global variables and define what GNN, data and explainers to use 
   1. `LOADING_FROM_FILES = False`
   1. `TRAIN_MODEL = True`
   2. Define sample size, which split method and formula to use 
   3. Define output folder
   2. Choose GNN model, dataset, and explainer(s) to use 
3. Train the model and generate explanations by running the cells 

### For subsequent runs with a pre-existing model and explanations
1. Set global variables
   1. `LOADING_FROM_FILES = True`
   1. `TRAIN_MODEL = False`
   1. change `MODELPATH` to correspond with the trained model
2. Run the cells to load in the model and explanations

### Generating comprehensiveness and sufficiency for explanations
1. Run the cells in order

`N.B.`: If CUDA runs out of memory with generating comprehensiveness and sufficiency for the explainers in method `read_comp_suff`,
restart the Jupyter server and just rerun all cells (set `LOADING_FROM_FILES = True` and `TRAIN_MODEL = False` 
if this occurs on the first run).

2. Repeat until all explainers have comprehensiveness and sufficiency calculated.
3. Run cells until average comprehensiveness and sufficiency per explainer have been output. 

## References
[1] J. Rao, S. Zheng, Y. Song, J. Chen, C. Li, J. Xie, H. Yang, H. Chen, and Y. Yang, 
“Molrep: A deep representation learning library for molecular property prediction”, https://arxiv.org/abs/2107.04119

[2] M. Rathee, T. Funke, A. Anand, and M. Khosla, “Bagel: A benchmark for assessing graph neural network explanations”, https://arxiv.org/abs/2206.13983