# STGAT
This repository represents a framework named STGAT which can be trained on Spatial Transcriptomics (ST) data and applied on TCGA data to generate spot-level gene expression for the TCGA data. The produced gene expression can be leveraged various downstream analysis including subtype classification, survival prediction etc. Moreover, comprehensive experimentation proves that considering only the tumor spots to compute the mean gene expression of a sample results in better performance.

## Required Python libraries
- Python (>= 3.9.7)
- Pytorch (>= 1.11) [with cudatoolkit (>= 11.3.1) if cuda is used]
- scikit-learn (>= 1.0.2)
- scipy (>= 1.7.1)
- pandas (>= 1.3.4)
- numpy (>= 1.20.3)
- scikit-image (>=0.19.2)
